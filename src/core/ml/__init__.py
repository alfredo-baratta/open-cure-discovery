"""
Machine learning module.

This module provides pre-trained models for binding affinity prediction,
molecular property calculation, and other ML-based tasks.

Includes:
- FingerprintGenerator: Generate molecular fingerprints (Morgan, MACCS, etc.)
- BindingPredictor: Basic binding affinity prediction
- DeepChemBindingPredictor: Graph neural network binding prediction (optional)
"""

from src.core.ml.fingerprints import (
    FingerprintGenerator,
    FingerprintConfig,
    FingerprintType,
    MolecularDescriptorCalculator,
)
from src.core.ml.binding import (
    BindingPredictor,
    NeuralNetworkPredictor,
    EnsemblePredictor,
    PredictorConfig,
)

# Optional DeepChem integration
try:
    from src.core.ml.deepchem_binding import (
        DeepChemBindingPredictor,
        DeepChemConfig,
        create_pretrained_predictor,
    )
    _DEEPCHEM_AVAILABLE = True
except ImportError:
    _DEEPCHEM_AVAILABLE = False
    DeepChemBindingPredictor = None
    DeepChemConfig = None
    create_pretrained_predictor = None

__all__ = [
    # Fingerprints
    "FingerprintGenerator",
    "FingerprintConfig",
    "FingerprintType",
    "MolecularDescriptorCalculator",
    # Basic binding prediction
    "BindingPredictor",
    "NeuralNetworkPredictor",
    "EnsemblePredictor",
    "PredictorConfig",
    # DeepChem (optional)
    "DeepChemBindingPredictor",
    "DeepChemConfig",
    "create_pretrained_predictor",
]
