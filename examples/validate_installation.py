#!/usr/bin/env python3
"""
Validate Open Cure Discovery installation.

This script checks that all components are properly installed
and functioning correctly.

Usage:
    python examples/validate_installation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_status(name: str, status: bool, details: str = "") -> None:
    """Print status with colored indicator."""
    indicator = "[OK]" if status else "[FAIL]"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"

    print(f"  {color}{indicator}{reset} {name}")
    if details:
        print(f"       {details}")


def check_python_version() -> bool:
    """Check Python version."""
    version = sys.version_info
    ok = version >= (3, 10)
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    print_status("Python 3.10+", ok, details)
    return ok


def check_core_imports() -> bool:
    """Check core module imports."""
    try:
        from src.core.models import Molecule, ProteinTarget
        from src.core.config import CampaignConfig
        print_status("Core models", True)
        return True
    except ImportError as e:
        print_status("Core models", False, str(e))
        return False


def check_numpy() -> bool:
    """Check NumPy."""
    try:
        import numpy as np
        version = np.__version__
        print_status("NumPy", True, f"v{version}")
        return True
    except ImportError:
        print_status("NumPy", False, "pip install numpy")
        return False


def check_rdkit() -> bool:
    """Check RDKit."""
    try:
        from rdkit import Chem
        from rdkit import __version__ as rdkit_version
        print_status("RDKit", True, f"v{rdkit_version}")
        return True
    except ImportError:
        print_status("RDKit", False, "pip install rdkit")
        return False


def check_pytorch() -> tuple[bool, bool]:
    """Check PyTorch and CUDA."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()

        print_status("PyTorch", True, f"v{version}")

        if cuda_available:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            print_status("CUDA", True, f"v{cuda_version} ({device_name})")
        else:
            print_status("CUDA", False, "GPU acceleration unavailable")

        return True, cuda_available

    except ImportError:
        print_status("PyTorch", False, "pip install torch")
        return False, False


def check_fingerprints() -> bool:
    """Check fingerprint generation."""
    try:
        from src.core.ml.fingerprints import FingerprintGenerator

        gen = FingerprintGenerator()
        fp = gen.generate("CCO")  # Ethanol

        if fp is not None and len(fp) == 2048:
            print_status("Fingerprint generation", True)
            return True
        else:
            print_status("Fingerprint generation", False, "Invalid output")
            return False

    except Exception as e:
        print_status("Fingerprint generation", False, str(e))
        return False


def check_admet() -> bool:
    """Check ADMET calculations."""
    try:
        from src.core.admet import ADMETCalculator

        calc = ADMETCalculator()
        props = calc.calculate("CCO")  # Ethanol

        if props.qed_score is not None:
            print_status("ADMET calculation", True, f"QED={props.qed_score:.3f}")
            return True
        else:
            print_status("ADMET calculation", False, "Missing QED score")
            return False

    except Exception as e:
        print_status("ADMET calculation", False, str(e))
        return False


def check_filters() -> bool:
    """Check molecular filters."""
    try:
        from src.core.admet import DrugLikenessFilter

        filt = DrugLikenessFilter()
        passed, results = filt.filter("CCO")

        print_status("Molecular filters", True, f"Passed={passed}")
        return True

    except Exception as e:
        print_status("Molecular filters", False, str(e))
        return False


def check_binding_predictor() -> bool:
    """Check binding predictor."""
    try:
        from src.core.ml import BindingPredictor

        predictor = BindingPredictor(use_gpu=False)
        score = predictor.predict("CCO")

        print_status("Binding predictor", True, f"Score={score:.3f}")
        return True

    except Exception as e:
        print_status("Binding predictor", False, str(e))
        return False


def check_scoring() -> bool:
    """Check scoring system."""
    try:
        from src.core.models import Molecule
        from src.core.scoring import CompositeScorer

        scorer = CompositeScorer()
        mol = Molecule(id="test", smiles="CCO")
        candidate = scorer.score(mol, "target1", ml_binding_score=0.7)

        print_status("Scoring system", True, f"Score={candidate.final_score:.3f}")
        return True

    except Exception as e:
        print_status("Scoring system", False, str(e))
        return False


def check_pipeline() -> bool:
    """Check pipeline import."""
    try:
        from src.core.pipeline import ScreeningPipeline, PipelineConfig
        print_status("Screening pipeline", True)
        return True
    except Exception as e:
        print_status("Screening pipeline", False, str(e))
        return False


def check_data_loaders() -> bool:
    """Check data loaders."""
    try:
        from src.data.loaders import ChEMBLLoader, PDBLoader, ZINCLoader
        print_status("Data loaders", True)
        return True
    except Exception as e:
        print_status("Data loaders", False, str(e))
        return False


def check_docking_engine() -> bool:
    """Check docking engine."""
    try:
        from src.core.docking import DockingEngine

        engine = DockingEngine(prefer_gpu=False)

        if engine.is_available:
            print_status("Docking engine", True, "AutoDock available")
        else:
            print_status("Docking engine", True, "No docking software found (optional)")

        return True

    except Exception as e:
        print_status("Docking engine", False, str(e))
        return False


def main():
    """Run all validation checks."""
    print()
    print("=" * 60)
    print("Open Cure Discovery - Installation Validation")
    print("=" * 60)
    print()

    results = {}

    # Core checks
    print("Core Dependencies:")
    print("-" * 40)
    results["python"] = check_python_version()
    results["core"] = check_core_imports()
    results["numpy"] = check_numpy()
    results["rdkit"] = check_rdkit()
    pytorch_ok, cuda_ok = check_pytorch()
    results["pytorch"] = pytorch_ok
    results["cuda"] = cuda_ok
    print()

    # Component checks
    print("Components:")
    print("-" * 40)
    results["fingerprints"] = check_fingerprints()
    results["admet"] = check_admet()
    results["filters"] = check_filters()
    results["binding"] = check_binding_predictor()
    results["scoring"] = check_scoring()
    results["pipeline"] = check_pipeline()
    results["loaders"] = check_data_loaders()
    results["docking"] = check_docking_engine()
    print()

    # Summary
    print("=" * 60)
    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    if failed == 0:
        print("\033[92mAll checks passed!\033[0m")
        print("Open Cure Discovery is ready to use.")
    else:
        print(f"\033[93m{passed}/{total} checks passed\033[0m")
        print("\nTo fix issues, install missing dependencies:")
        print("  pip install -e '.[all]'")

    print()

    # Feature summary
    print("Feature Availability:")
    print("-" * 40)

    features = [
        ("Fingerprint generation", results.get("fingerprints", False) and results.get("rdkit", False)),
        ("ADMET prediction", results.get("admet", False) and results.get("rdkit", False)),
        ("ML binding prediction", results.get("binding", False)),
        ("GPU acceleration", results.get("cuda", False)),
        ("Molecular docking", results.get("docking", False)),
        ("Data download", results.get("loaders", False)),
    ]

    for name, available in features:
        status = "\033[92mAvailable\033[0m" if available else "\033[93mLimited\033[0m"
        print(f"  {name}: {status}")

    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
