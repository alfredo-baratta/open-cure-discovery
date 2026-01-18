"""
Machine learning-based binding affinity prediction.

This module provides neural network models for predicting
molecule-target binding affinity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np

from loguru import logger

from src.core.ml.fingerprints import FingerprintGenerator, FingerprintConfig, FingerprintType


@dataclass
class PredictorConfig:
    """Configuration for binding predictor."""
    model_path: Optional[Path] = None
    use_gpu: bool = True
    batch_size: int = 256
    fingerprint_type: FingerprintType = FingerprintType.MORGAN
    fingerprint_bits: int = 2048
    fingerprint_radius: int = 2


class BindingPredictorBase(ABC):
    """Abstract base class for binding predictors."""

    @abstractmethod
    def predict(self, smiles: str, target_id: Optional[str] = None) -> float:
        """Predict binding probability for a single molecule."""
        pass

    @abstractmethod
    def predict_batch(
        self,
        smiles_list: list[str],
        target_id: Optional[str] = None,
    ) -> np.ndarray:
        """Predict binding probabilities for a batch of molecules."""
        pass


class NeuralNetworkPredictor(BindingPredictorBase):
    """
    Neural network-based binding affinity predictor.

    Uses a pre-trained neural network to predict binding
    probability from molecular fingerprints.
    """

    def __init__(self, config: Optional[PredictorConfig] = None):
        """
        Initialize predictor.

        Args:
            config: Predictor configuration.
        """
        self.config = config or PredictorConfig()
        self.model = None
        self.device = "cpu"

        # Initialize fingerprint generator
        fp_config = FingerprintConfig(
            fp_type=self.config.fingerprint_type,
            n_bits=self.config.fingerprint_bits,
            radius=self.config.fingerprint_radius,
        )
        self.fingerprint_gen = FingerprintGenerator(fp_config)

        # Try to load model
        if self.config.model_path and self.config.model_path.exists():
            self._load_model(self.config.model_path)

    def _load_model(self, model_path: Path) -> None:
        """Load pre-trained model."""
        try:
            import torch

            self.device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"

            # Try loading as PyTorch model
            if model_path.suffix == ".pt":
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                logger.info(f"Loaded PyTorch model from {model_path}")

            # Try loading as ONNX model
            elif model_path.suffix == ".onnx":
                import onnxruntime as ort

                providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
                self.model = ort.InferenceSession(str(model_path), providers=providers)
                logger.info(f"Loaded ONNX model from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def predict(self, smiles: str, target_id: Optional[str] = None) -> float:
        """
        Predict binding probability for a single molecule.

        Args:
            smiles: SMILES string.
            target_id: Optional target identifier (for target-specific models).

        Returns:
            Binding probability (0-1).
        """
        predictions = self.predict_batch([smiles], target_id)
        return float(predictions[0])

    def predict_batch(
        self,
        smiles_list: list[str],
        target_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict binding probabilities for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings.
            target_id: Optional target identifier.

        Returns:
            NumPy array of binding probabilities.
        """
        # Generate fingerprints
        fingerprints = self.fingerprint_gen.generate_batch(smiles_list)

        if self.model is None:
            # No model loaded - use heuristic based on drug-likeness
            logger.warning("No model loaded, using heuristic prediction")
            return self._heuristic_predict(smiles_list)

        try:
            import torch

            if isinstance(self.model, torch.nn.Module):
                return self._predict_pytorch(fingerprints)
            else:
                # ONNX model
                return self._predict_onnx(fingerprints)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.full(len(smiles_list), 0.5)

    def _predict_pytorch(self, fingerprints: np.ndarray) -> np.ndarray:
        """Run prediction with PyTorch model."""
        import torch

        with torch.no_grad():
            tensor = torch.FloatTensor(fingerprints).to(self.device)

            # Process in batches
            predictions = []
            for i in range(0, len(tensor), self.config.batch_size):
                batch = tensor[i:i + self.config.batch_size]
                output = self.model(batch)
                # Assume output is logits, apply sigmoid
                probs = torch.sigmoid(output).cpu().numpy()
                predictions.append(probs)

            return np.concatenate(predictions).flatten()

    def _predict_onnx(self, fingerprints: np.ndarray) -> np.ndarray:
        """Run prediction with ONNX model."""
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        predictions = []
        for i in range(0, len(fingerprints), self.config.batch_size):
            batch = fingerprints[i:i + self.config.batch_size].astype(np.float32)
            output = self.model.run([output_name], {input_name: batch})[0]
            # Apply sigmoid if needed
            probs = 1 / (1 + np.exp(-output))
            predictions.append(probs)

        return np.concatenate(predictions).flatten()

    def _heuristic_predict(self, smiles_list: list[str]) -> np.ndarray:
        """
        Heuristic binding prediction based on drug-likeness.

        Uses molecular descriptors to estimate binding probability
        when no ML model is available.
        """
        from src.core.ml.fingerprints import MolecularDescriptorCalculator

        calc = MolecularDescriptorCalculator()
        predictions = []

        for smiles in smiles_list:
            desc = calc.calculate(smiles)
            if not desc:
                predictions.append(0.3)  # Unknown
                continue

            # Score based on drug-like properties
            score = 0.5  # Base score

            # QED score contribution (0-1)
            qed = desc.get("qed", 0.5)
            score += 0.2 * qed

            # Penalize for Lipinski violations
            violations = desc.get("lipinski_violations", 0)
            score -= 0.1 * violations

            # Molecular weight preference (200-500 optimal)
            mw = desc.get("mol_weight", 300)
            if 200 <= mw <= 500:
                score += 0.1
            elif mw > 600:
                score -= 0.1

            # LogP preference (-1 to 4 optimal)
            logp = desc.get("logp", 2)
            if -1 <= logp <= 4:
                score += 0.05
            elif logp > 5:
                score -= 0.1

            predictions.append(max(0.0, min(1.0, score)))

        return np.array(predictions, dtype=np.float32)


class EnsemblePredictor(BindingPredictorBase):
    """
    Ensemble of multiple binding predictors.

    Combines predictions from multiple models for more
    robust binding affinity estimation.
    """

    def __init__(self, predictors: list[BindingPredictorBase], weights: Optional[list[float]] = None):
        """
        Initialize ensemble.

        Args:
            predictors: List of binding predictors.
            weights: Optional weights for each predictor (default: equal).
        """
        self.predictors = predictors
        self.weights = weights or [1.0 / len(predictors)] * len(predictors)

        if len(self.weights) != len(self.predictors):
            raise ValueError("Number of weights must match number of predictors")

    def predict(self, smiles: str, target_id: Optional[str] = None) -> float:
        """Predict using ensemble average."""
        predictions = [p.predict(smiles, target_id) for p in self.predictors]
        return sum(w * p for w, p in zip(self.weights, predictions))

    def predict_batch(
        self,
        smiles_list: list[str],
        target_id: Optional[str] = None,
    ) -> np.ndarray:
        """Predict batch using ensemble average."""
        all_predictions = []

        for predictor in self.predictors:
            preds = predictor.predict_batch(smiles_list, target_id)
            all_predictions.append(preds)

        # Weighted average
        stacked = np.stack(all_predictions, axis=0)
        weights = np.array(self.weights).reshape(-1, 1)
        return np.sum(stacked * weights, axis=0)


class BindingPredictor:
    """
    Main binding predictor interface.

    Provides a unified interface for binding affinity prediction,
    automatically selecting the best available method.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize binding predictor.

        Args:
            model_path: Path to pre-trained model.
            use_gpu: Whether to use GPU acceleration.
        """
        config = PredictorConfig(
            model_path=model_path,
            use_gpu=use_gpu,
        )
        self._predictor = NeuralNetworkPredictor(config)

    def predict(
        self,
        smiles: Union[str, list[str]],
        target_id: Optional[str] = None,
    ) -> Union[float, np.ndarray]:
        """
        Predict binding probability.

        Args:
            smiles: Single SMILES or list of SMILES.
            target_id: Optional target identifier.

        Returns:
            Single probability or array of probabilities.
        """
        if isinstance(smiles, str):
            return self._predictor.predict(smiles, target_id)
        else:
            return self._predictor.predict_batch(smiles, target_id)

    @property
    def has_model(self) -> bool:
        """Check if a ML model is loaded."""
        return self._predictor.model is not None
