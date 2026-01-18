"""
DeepChem-based binding affinity prediction.

This module provides professional binding affinity prediction using
DeepChem's graph neural networks trained on the PDBbind dataset.

Features:
- Graph Convolutional Networks for molecular representation
- Training on PDBbind dataset (up to 19,000 protein-ligand complexes)
- SMILES-only prediction (no 3D structure needed for inference)
- Model persistence and loading
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import tempfile
import pickle
import json

from loguru import logger


@dataclass
class DeepChemConfig:
    """Configuration for DeepChem binding predictor."""
    model_type: str = "graphconv"  # "graphconv", "attentivefp"
    model_dir: Optional[Path] = None
    n_tasks: int = 1
    graph_conv_layers: List[int] = field(default_factory=lambda: [128, 128])
    dense_layer_size: int = 256
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    use_gpu: bool = True


class DeepChemBindingPredictor:
    """
    Professional binding affinity predictor using DeepChem.

    This predictor uses Graph Convolutional Networks trained on the
    PDBbind dataset to predict protein-ligand binding affinities from
    SMILES strings alone.

    Attributes:
        model: DeepChem model instance
        featurizer: Molecular featurizer for SMILES conversion
        transformers: Data transformers for normalization
    """

    def __init__(self, config: Optional[DeepChemConfig] = None):
        """
        Initialize predictor.

        Args:
            config: Configuration for the predictor.
        """
        self.config = config or DeepChemConfig()
        self.model = None
        self.featurizer = None
        self.transformers = []
        self._dc = None  # DeepChem module
        self._is_trained = False

        # Check DeepChem availability
        self._check_deepchem()

    def _check_deepchem(self) -> bool:
        """Check if DeepChem is available."""
        try:
            import deepchem as dc
            self._dc = dc
            logger.info(f"DeepChem {dc.__version__} available")
            return True
        except ImportError:
            logger.warning(
                "DeepChem not available. Install with: pip install deepchem[torch]"
            )
            return False

    def _create_model(self):
        """Create DeepChem model based on configuration."""
        if self._dc is None:
            raise RuntimeError("DeepChem not available")

        dc = self._dc

        # Create model directory
        model_dir = self.config.model_dir or Path(tempfile.mkdtemp())
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.config.model_type == "graphconv":
            self.model = dc.models.GraphConvModel(
                n_tasks=self.config.n_tasks,
                mode='regression',
                graph_conv_layers=self.config.graph_conv_layers,
                dense_layer_size=self.config.dense_layer_size,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                model_dir=str(model_dir),
            )
            self.featurizer = dc.feat.ConvMolFeaturizer()

        elif self.config.model_type == "attentivefp":
            self.model = dc.models.AttentiveFPModel(
                n_tasks=self.config.n_tasks,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                model_dir=str(model_dir),
            )
            self.featurizer = dc.feat.MolGraphConvFeaturizer()

        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        logger.info(f"Created {self.config.model_type} model")

    def train_on_pdbbind(
        self,
        dataset_size: str = "core",
        epochs: Optional[int] = None,
        validation_split: float = 0.1,
    ) -> Dict[str, float]:
        """
        Train model on PDBbind dataset.

        Args:
            dataset_size: "core" (195), "refined" (~5000), or "general" (~19000)
            epochs: Number of training epochs (uses config default if None)
            validation_split: Fraction of data for validation

        Returns:
            Dictionary with training metrics (train_r2, valid_r2, test_r2)
        """
        if self._dc is None:
            raise RuntimeError("DeepChem not available")

        dc = self._dc
        epochs = epochs or self.config.epochs

        logger.info(f"Loading PDBbind {dataset_size} dataset...")

        # Load PDBbind dataset
        tasks, datasets, transformers = dc.molnet.load_pdbbind(
            featurizer='GraphConv' if self.config.model_type == "graphconv" else 'GraphConv',
            set_name=dataset_size,
            split='random',
        )

        train_dataset, valid_dataset, test_dataset = datasets
        self.transformers = transformers

        logger.info(f"Training: {len(train_dataset)}, Validation: {len(valid_dataset)}, Test: {len(test_dataset)}")

        # Create model
        self._create_model()

        # Train
        logger.info(f"Training for {epochs} epochs...")
        losses = self.model.fit(
            train_dataset,
            nb_epoch=epochs,
            checkpoint_interval=max(1, epochs // 5),
        )

        # Evaluate
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

        train_scores = self.model.evaluate(train_dataset, [metric], transformers)
        valid_scores = self.model.evaluate(valid_dataset, [metric], transformers)
        test_scores = self.model.evaluate(test_dataset, [metric], transformers)

        results = {
            "train_r2": train_scores.get('mean-pearson_r2_score', 0),
            "valid_r2": valid_scores.get('mean-pearson_r2_score', 0),
            "test_r2": test_scores.get('mean-pearson_r2_score', 0),
            "epochs": epochs,
            "dataset_size": dataset_size,
            "train_samples": len(train_dataset),
        }

        logger.info(f"Training complete. Test R²: {results['test_r2']:.4f}")

        self._is_trained = True
        return results

    def train_on_custom_data(
        self,
        smiles_list: List[str],
        affinities: List[float],
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train model on custom binding data.

        Args:
            smiles_list: List of SMILES strings
            affinities: List of binding affinities (pKd, pKi, or kcal/mol)
            epochs: Number of training epochs

        Returns:
            Dictionary with training metrics
        """
        if self._dc is None:
            raise RuntimeError("DeepChem not available")

        dc = self._dc
        epochs = epochs or self.config.epochs

        # Create model and featurizer
        self._create_model()

        # Featurize molecules
        logger.info(f"Featurizing {len(smiles_list)} molecules...")
        X = self.featurizer.featurize(smiles_list)
        y = np.array(affinities).reshape(-1, 1)

        # Create dataset
        dataset = dc.data.NumpyDataset(X=X, y=y, ids=smiles_list)

        # Split
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
        )

        logger.info(f"Training: {len(train_dataset)}, Validation: {len(valid_dataset)}, Test: {len(test_dataset)}")

        # Train
        logger.info(f"Training for {epochs} epochs...")
        self.model.fit(train_dataset, nb_epoch=epochs)

        # Evaluate
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

        train_scores = self.model.evaluate(train_dataset, [metric])
        test_scores = self.model.evaluate(test_dataset, [metric])

        results = {
            "train_r2": train_scores.get('mean-pearson_r2_score', 0),
            "test_r2": test_scores.get('mean-pearson_r2_score', 0),
            "epochs": epochs,
            "train_samples": len(train_dataset),
        }

        logger.info(f"Training complete. Test R²: {results['test_r2']:.4f}")

        self._is_trained = True
        return results

    def predict(self, smiles: str) -> float:
        """
        Predict binding affinity for a single molecule.

        Args:
            smiles: SMILES string

        Returns:
            Predicted binding affinity
        """
        predictions = self.predict_batch([smiles])
        return float(predictions[0])

    def predict_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Predict binding affinities for multiple molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Array of predicted binding affinities
        """
        if self.model is None or not self._is_trained:
            logger.warning("Model not trained. Using heuristic prediction.")
            return self._heuristic_prediction(smiles_list)

        dc = self._dc

        # Featurize
        X = self.featurizer.featurize(smiles_list)

        # Handle failed featurizations
        valid_indices = []
        valid_X = []
        for i, x in enumerate(X):
            if x is not None:
                valid_indices.append(i)
                valid_X.append(x)

        if not valid_X:
            logger.warning("All molecules failed featurization")
            return np.zeros(len(smiles_list))

        # Create dataset
        dataset = dc.data.NumpyDataset(X=np.array(valid_X), ids=[smiles_list[i] for i in valid_indices])

        # Predict
        predictions = self.model.predict(dataset)

        # Apply inverse transformers if available
        if self.transformers:
            for transformer in reversed(self.transformers):
                predictions = transformer.untransform(predictions)

        # Build full result array
        result = np.zeros(len(smiles_list))
        for i, pred in zip(valid_indices, predictions):
            result[i] = pred[0] if len(pred) > 0 else 0

        return result

    def _heuristic_prediction(self, smiles_list: List[str]) -> np.ndarray:
        """Fallback heuristic prediction based on molecular properties."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED

            predictions = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    predictions.append(0.5)
                    continue

                # Simple heuristic based on drug-likeness
                qed = QED.qed(mol)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)

                # Estimate binding probability
                # Drug-like molecules typically have better binding
                mw_score = 1.0 if 200 < mw < 600 else 0.5
                logp_score = 1.0 if 0 < logp < 5 else 0.5

                score = qed * 0.5 + mw_score * 0.25 + logp_score * 0.25
                predictions.append(score)

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Heuristic prediction failed: {e}")
            return np.full(len(smiles_list), 0.5)

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "config.json"
        config_dict = {
            "model_type": self.config.model_type,
            "n_tasks": self.config.n_tasks,
            "graph_conv_layers": self.config.graph_conv_layers,
            "dense_layer_size": self.config.dense_layer_size,
            "dropout": self.config.dropout,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save transformers
        if self.transformers:
            transformers_path = path / "transformers.pkl"
            with open(transformers_path, "wb") as f:
                pickle.dump(self.transformers, f)

        # Model is already saved by DeepChem in model_dir
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "DeepChemBindingPredictor":
        """
        Load model from disk.

        Args:
            path: Directory containing saved model

        Returns:
            Loaded predictor instance
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        config = DeepChemConfig(
            model_type=config_dict.get("model_type", "graphconv"),
            model_dir=path,
            n_tasks=config_dict.get("n_tasks", 1),
            graph_conv_layers=config_dict.get("graph_conv_layers", [128, 128]),
            dense_layer_size=config_dict.get("dense_layer_size", 256),
            dropout=config_dict.get("dropout", 0.3),
        )

        predictor = cls(config)

        # Create and restore model
        predictor._create_model()
        predictor.model.restore()

        # Load transformers
        transformers_path = path / "transformers.pkl"
        if transformers_path.exists():
            with open(transformers_path, "rb") as f:
                predictor.transformers = pickle.load(f)

        predictor._is_trained = True
        logger.info(f"Model loaded from {path}")

        return predictor


def create_pretrained_predictor(
    dataset_size: str = "core",
    model_dir: Optional[str] = None,
    epochs: int = 50,
) -> DeepChemBindingPredictor:
    """
    Create and train a binding predictor on PDBbind.

    This is a convenience function for creating a production-ready predictor.

    Args:
        dataset_size: "core" (fast), "refined" (recommended), or "general" (best)
        model_dir: Directory to save trained model
        epochs: Number of training epochs

    Returns:
        Trained predictor instance
    """
    config = DeepChemConfig(
        model_type="graphconv",
        model_dir=Path(model_dir) if model_dir else None,
        epochs=epochs,
    )

    predictor = DeepChemBindingPredictor(config)

    # Train on PDBbind
    metrics = predictor.train_on_pdbbind(dataset_size=dataset_size, epochs=epochs)

    logger.info(f"Trained predictor with Test R² = {metrics['test_r2']:.4f}")

    return predictor
