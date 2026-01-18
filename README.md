# Open Cure Discovery

**Accelerating drug discovery through open-source, GPU-powered virtual screening**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-67%20passed-brightgreen.svg)]()

> **Created by [Alfredo Baratta](https://github.com/alfredo-baratta)** - Democratizing drug discovery for everyone.

---

## The Vision

Every year, millions of people suffer from diseases that lack effective treatments. Traditional drug discovery is a lengthy and expensive process:

- **10-15 years** from initial discovery to market
- **$2-3 billion** average cost per approved drug
- **90% failure rate** in clinical trials
- **Limited accessibility** to expensive computational resources

**Open Cure Discovery** aims to revolutionize this process by putting the power of computational drug screening in everyone's hands. With just a consumer-grade GPU (GTX 1060 6GB or better), anyone can contribute to finding new treatments for cancer, Alzheimer's, infectious diseases, and more.

### Why This Matters

Pharmaceutical companies focus on profitable diseases, leaving many rare conditions ("orphan diseases") without research investment. By democratizing drug discovery:

- **Researchers worldwide** can screen millions of compounds without expensive infrastructure
- **Patient advocacy groups** can drive research for neglected diseases
- **Academic institutions** can participate in cutting-edge drug discovery
- **Citizen scientists** can contribute computational power to find cures

---

## How It Works

Open Cure Discovery uses a multi-stage pipeline that mimics the early phases of pharmaceutical drug discovery:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VIRTUAL SCREENING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  MOLECULES   │    │   FILTERS    │    │   DOCKING    │    │  SCORING  │ │
│  │  ──────────  │ -> │  ──────────  │ -> │  ──────────  │ -> │ ───────── │ │
│  │ SMILES input │    │ PAINS/Lipinski│   │ AutoDock-GPU │    │ Composite │ │
│  │ ChEMBL/ZINC  │    │ Toxicophores │    │ Binding poses│    │ ML+ADMET  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                             │
│                                    ↓                                        │
│                                                                             │
│                        ┌────────────────────────┐                           │
│                        │   RANKED CANDIDATES    │                           │
│                        │   ──────────────────   │                           │
│                        │  CSV, SMILES, JSON     │                           │
│                        │  Top-N with scores     │                           │
│                        │  Ready for validation  │                           │
│                        └────────────────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Molecule Input
- Load molecules from public databases (ChEMBL, ZINC, PDB)
- Support for custom molecule libraries in SMILES format
- Batch processing for millions of compounds

### Stage 2: Pre-filtering
- **PAINS filters**: Remove Pan-Assay Interference Compounds (false positives)
- **Lipinski's Rule of Five**: Ensure drug-likeness properties
- **Toxicophore detection**: Flag potentially toxic substructures
- Eliminates 40-60% of unsuitable candidates early, saving computational time

### Stage 3: Molecular Docking
- **GPU-accelerated docking** using AutoDock-GPU
- Simulates how molecules bind to target proteins
- Calculates binding affinity (how strongly a molecule binds)
- CPU fallback with AutoDock Vina for systems without GPU

### Stage 4: ML Prediction & ADMET
- **Machine Learning binding prediction**: Neural network models predict binding affinity
- **ADMET analysis** (15+ endpoints):
  - **A**bsorption: Can the body absorb the drug?
  - **D**istribution: Where does it go in the body?
  - **M**etabolism: How is it processed?
  - **E**xcretion: How is it eliminated?
  - **T**oxicity: Is it safe?

### Stage 5: Scoring & Ranking
- **Composite scoring**: Combines docking, ML, and ADMET scores
- **Pareto optimization**: Multi-objective ranking for best trade-offs
- **Diversity selection**: Ensures chemically diverse candidates
- Outputs ranked list of promising drug candidates

---

## Key Advantages

### Zero Cost
- **100% free and open-source** - No subscriptions, no cloud fees
- Uses freely available molecular databases (ChEMBL, ZINC, PDB)
- No proprietary software dependencies

### Runs on Consumer Hardware
- **Minimum**: NVIDIA GTX 1060 6GB (or CPU-only mode)
- **Optimized batch processing** to fit in limited GPU memory
- Automatic hardware detection and configuration

### Scientifically Rigorous
- Based on established computational chemistry methods
- Validated against benchmark datasets
- Clear documentation of limitations and assumptions

### Modular & Extensible
- Add custom scoring functions
- Integrate new molecular databases
- Extend with additional ADMET endpoints

### Performance Estimates

| Hardware | Molecules/Day | Use Case |
|----------|---------------|----------|
| GTX 1060 6GB | ~100,000 | Personal research |
| RTX 3060 12GB | ~300,000 | Small lab |
| RTX 4090 24GB | ~1,000,000 | High-throughput screening |
| CPU only (8 cores) | ~10,000 | Testing/Development |

---

## Current Status

**Phase 1 Core Implementation: COMPLETE**

| Component | Status | Description |
|-----------|--------|-------------|
| Docking Engine | ✅ | AutoDock-GPU integration with Vina fallback |
| ML Prediction | ✅ | Binding affinity and fingerprint models |
| ADMET | ✅ | 15+ pharmacokinetic/toxicity endpoints |
| Scoring | ✅ | Composite scoring with Pareto ranking |
| Pipeline | ✅ | Complete screening workflow |
| CLI | ✅ | Command-line interface |
| Receptor Preparation | ✅ | Professional PDB to PDBQT conversion |
| Binding Site Detection | ✅ | Automatic extraction from co-crystallized ligands |
| DeepChem Integration | ✅ | Graph neural network binding prediction (optional) |
| Multi-Target Validation | ✅ | Validated on COX-2, EGFR, HIV-PR, AChE |
| Tests | ✅ | 67 unit and integration tests (100% passing) |

---

## Installation Guide

### TL;DR - Quick Install (Experienced Users)

```bash
# Clone and setup
git clone https://github.com/alfredo-baratta/open-cure-discovery.git
cd open-cure-discovery
python -m venv venv && source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -e ".[all]"

# Download Vina (required for docking)
# Windows: Download vina.exe from https://github.com/ccsb-scripps/AutoDock-Vina/releases
# Linux: wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 -O tools/vina && chmod +x tools/vina

# Validate
python examples/validate_installation.py
python examples/multi_target_validation.py  # Full validation (~10 min)
```

---

### Prerequisites

Before installing Open Cure Discovery, ensure you have:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10+ | [Download](https://www.python.org/downloads/) |
| **Git** | Any | [Download](https://git-scm.com/downloads) |
| **NVIDIA GPU** | Optional | For GPU-accelerated docking |
| **CUDA Toolkit** | 11.0+ | Only if using GPU features |

### Step 1: Clone the Repository

```bash
git clone https://github.com/alfredo-baratta/open-cure-discovery.git
cd open-cure-discovery
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
# Install core dependencies
pip install -e ".[all]"

# Verify RDKit installation (required)
python -c "from rdkit import Chem; print('RDKit OK')"
```

### Step 4: Install AutoDock Vina (Required for Docking)

AutoDock Vina is the molecular docking engine. **You must install it separately.**

#### Windows

1. Download from: https://github.com/ccsb-scripps/AutoDock-Vina/releases
2. Download `vina_1.2.5_windows_x86_64.zip` (or latest version)
3. Extract `vina.exe` to a `tools/` folder in your project:
   ```
   open-cure-discovery/
   └── tools/
       └── vina.exe
   ```
4. Verify installation:
   ```powershell
   .\tools\vina.exe --version
   # Expected: AutoDock Vina 1.2.5
   ```

#### Linux

```bash
# Option 1: Download binary
wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64
chmod +x vina_1.2.5_linux_x86_64
mkdir -p tools && mv vina_1.2.5_linux_x86_64 tools/vina

# Option 2: Install via conda
conda install -c conda-forge autodock-vina

# Verify
./tools/vina --version
```

#### macOS

```bash
# Download binary
wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_mac_x86_64
chmod +x vina_1.2.5_mac_x86_64
mkdir -p tools && mv vina_1.2.5_mac_x86_64 tools/vina

# For Apple Silicon (M1/M2), use Rosetta or compile from source
# Verify
./tools/vina --version
```

### Step 5: Install Optional Dependencies

#### AutoDock-GPU (For NVIDIA GPU Users)

If you have an NVIDIA GPU, you can use AutoDock-GPU for 10-100x faster docking.

**Windows:**
1. Download from: https://github.com/ccsb-scripps/AutoDock-GPU/releases
2. Download `adgpu-v1.5.3-windows.zip` (or latest)
3. Extract to `tools/` folder:
   ```
   open-cure-discovery/
   └── tools/
       ├── vina.exe
       └── autodock_gpu.exe
   ```
4. Verify CUDA is available:
   ```powershell
   nvidia-smi  # Should show your GPU
   .\tools\autodock_gpu.exe --version
   ```

**Linux:**
```bash
# Download binary
wget https://github.com/ccsb-scripps/AutoDock-GPU/releases/download/v1.5.3/adgpu-v1.5.3-linux.tar.gz
tar -xzf adgpu-v1.5.3-linux.tar.gz
mv autodock_gpu_* tools/autodock_gpu

# Verify
./tools/autodock_gpu --version
```

> **Note**: AutoDock-GPU requires NVIDIA GPU with CUDA. Falls back to Vina (CPU) if unavailable.

#### DeepChem (Advanced ML Models)

DeepChem provides graph neural network models for binding affinity prediction.

```bash
# Install DeepChem with PyTorch backend
pip install deepchem[torch]

# Verify installation
python -c "import deepchem; print(f'DeepChem {deepchem.__version__} OK')"
```

> **Note**: DeepChem is optional. The system works without it using heuristic models.

#### Meeko (Ligand Preparation)

Meeko is used for preparing ligands for docking.

```bash
pip install meeko

# Verify
python -c "from meeko import MoleculePreparation; print('Meeko OK')"
```

### Step 6: Validate Installation

Run the validation script to check everything is working:

```bash
python examples/validate_installation.py
```

**Expected output:**
```
============================================================
OPEN CURE DISCOVERY - Installation Validation
============================================================

[1] Checking Python version... OK (3.10.x)
[2] Checking RDKit... OK
[3] Checking NumPy... OK
[4] Checking Meeko... OK
[5] Checking AutoDock Vina... OK (tools/vina.exe)
[6] Checking DeepChem... OK (optional)

============================================================
ALL CHECKS PASSED - Installation successful!
============================================================
```

### Step 7: Run Multi-Target Validation (Recommended)

Verify the docking system works correctly on real drug targets:

```bash
python examples/multi_target_validation.py
```

This tests docking on 4 therapeutic targets (COX-2, EGFR, HIV-PR, AChE) with known drugs.

---

## Quick Start

### Run Demo Screening

```bash
# Quick demo with sample molecules (~1 second, no docking)
python examples/demo_screening.py
```

**Expected output:**
```
RESULTS
============================================================
Total screened: 10
Passed filters: 6
Duration: 0.71 seconds

Top Candidates:
------------------------------------------------------------
Rank   Name            Score      ADMET      ML Binding
------------------------------------------------------------
1      naproxen        0.5034     0.8211     0.8262
2      ibuprofen       0.4873     0.7616     0.8143
3      omeprazole      0.4764     0.7224     0.8055
...
```

### Run Real Docking Validation

```bash
# Test real docking on COX-2 target (~2 minutes)
python examples/real_docking_validation.py
```

### Programmatic Usage

```python
from src.core.models import Molecule, ProteinTarget
from src.core.pipeline import ScreeningPipeline, PipelineConfig
from src.core.docking import ReceptorPreparator

# Prepare receptor automatically from PDB
preparator = ReceptorPreparator(vina_path="tools/vina.exe")  # or "tools/vina" on Linux
receptor_info = preparator.prepare_from_pdb_id(
    pdb_id="1CX2",      # COX-2 structure
    ligand_code="S58",  # Co-crystallized ligand for binding site
)

# Create molecules to screen
molecules = [
    Molecule(id="celecoxib", smiles="CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"),
    Molecule(id="ibuprofen", smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
]

# Define target with auto-detected binding site
target = ProteinTarget(
    id="COX2",
    name="Cyclooxygenase-2",
    pdb_id="1CX2",
    binding_sites=[receptor_info.binding_site],
)

# Configure pipeline with docking enabled
config = PipelineConfig(
    run_docking=True,
    run_ml_prediction=True,
    run_admet=True,
    top_n=100,
    vina_path="tools/vina.exe",
)

# Run screening
pipeline = ScreeningPipeline(config)
results = pipeline.run(molecules, target)

# View top candidates
for candidate in results.candidates[:10]:
    print(f"{candidate.rank}. {candidate.molecule.name}: {candidate.final_score:.3f}")
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: rdkit` | Run `pip install rdkit` or use conda: `conda install -c conda-forge rdkit` |
| `vina.exe not found` | Download Vina and place in `tools/` folder |
| `CUDA out of memory` | Reduce batch size or use CPU mode |
| `DeepChem import error` | DeepChem is optional, system works without it |
| `Meeko preparation failed` | Ensure molecule SMILES is valid |

### Windows-Specific Issues

- **PowerShell execution policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Long path names**: Enable long paths in Windows settings or use shorter directory names

### Linux-Specific Issues

- **Permission denied on vina**: Run `chmod +x tools/vina`
- **Missing libstdc++**: Install with `sudo apt install libstdc++6`

### Component Summary

| Component | Required | Purpose | Installation |
|-----------|----------|---------|--------------|
| **Python 3.10+** | ✅ Yes | Runtime | python.org |
| **RDKit** | ✅ Yes | Cheminformatics | `pip install rdkit` |
| **AutoDock Vina** | ✅ Yes* | Molecular docking | Download binary |
| **Meeko** | ✅ Yes | Ligand preparation | `pip install meeko` |
| **AutoDock-GPU** | ❌ Optional | GPU-accelerated docking | Download binary |
| **DeepChem** | ❌ Optional | GNN binding prediction | `pip install deepchem[torch]` |
| **CUDA Toolkit** | ❌ Optional | GPU acceleration | nvidia.com |

*\*Required only if you want to run molecular docking. ML prediction and ADMET work without it.*

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Docking** | AutoDock-GPU with Vina CPU fallback |
| **Receptor Prep** | Automatic PDB download, cleaning, PDBQT conversion |
| **Binding Site** | Auto-detection from co-crystallized ligands |
| **Fingerprints** | Morgan (ECFP), MACCS, RDKit fingerprints |
| **ML Binding** | Neural network + DeepChem graph neural networks |
| **ADMET** | QED, Lipinski, BBB permeability, hERG inhibition, AMES mutagenicity, hepatotoxicity |
| **Filters** | PAINS, toxicophores, Lipinski, Veber |
| **Scoring** | Weighted composite with normalization |
| **Ranking** | Top-N, Pareto optimization, MaxMin diversity selection |

### Supported Databases

| Database | Content | Access |
|----------|---------|--------|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | 2M+ bioactive molecules | ✅ API integrated |
| [PDB](https://www.rcsb.org/) | 200K+ protein structures | ✅ API integrated |
| [ZINC](https://zinc.docking.org/) | 750M+ purchasable compounds | ✅ API integrated |

---

## Hardware Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GTX 1060 6GB (or CPU-only mode) |
| CPU | 4 cores, 2.5GHz |
| RAM | 16GB |
| Storage | 50GB (for databases and results) |

### Recommended
| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA RTX 3060 12GB or better |
| CPU | 8+ cores |
| RAM | 32GB |
| Storage | 100GB SSD |

---

## Project Structure

```
open-cure-discovery/
├── src/
│   ├── core/
│   │   ├── models.py         # Data models (Molecule, Target, etc.)
│   │   ├── config.py         # Configuration management
│   │   ├── pipeline.py       # Main screening pipeline
│   │   ├── docking/          # Molecular docking engines
│   │   │   ├── engine.py     # AutoDock-GPU and Vina engines
│   │   │   ├── receptor.py   # Professional receptor preparation
│   │   │   └── preparation.py # Ligand preparation
│   │   ├── ml/               # ML prediction
│   │   │   ├── fingerprints.py    # Morgan, MACCS, RDKit fingerprints
│   │   │   ├── binding.py         # Neural network binding prediction
│   │   │   └── deepchem_binding.py # DeepChem GNN models (optional)
│   │   ├── admet/            # ADMET calculation & filters
│   │   └── scoring/          # Scoring & ranking algorithms
│   ├── data/loaders/         # Database loaders (ChEMBL, PDB, ZINC)
│   ├── ui/cli/               # Command-line interface
│   └── utils/                # Utilities (GPU detection, I/O)
├── examples/
│   ├── demo_screening.py          # Demo script
│   ├── validate_installation.py   # Installation validation
│   ├── real_docking_validation.py # Single-target docking test
│   └── multi_target_validation.py # Multi-target validation suite
├── tests/                    # Test suite (67 tests)
├── configs/diseases/         # Disease-specific presets
└── docs/                     # Documentation
```

---

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [Development Roadmap](docs/ROADMAP.md) - Project phases and milestones
- [Task List](docs/TASKS.md) - Implementation status
- [Contributing Guide](CONTRIBUTING.md) - How to contribute

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=src

# Run specific test file
pytest tests/test_integration.py -v
```

---

## Contributing

We welcome contributions from everyone! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- Run validation and report issues
- Improve documentation
- Add new ADMET prediction endpoints
- Implement web dashboard (Phase 2)
- Validate results against experimental data
- Add support for new molecular databases
- Optimize performance for specific hardware

---

## Scientific Validity

This project prioritizes scientific rigor:

1. **Open Methods**: All algorithms are fully documented and open-source
2. **Benchmarking**: Validation against standard datasets (DUD-E, MUV)
3. **Multi-Target Validation**: Tested on 4 real therapeutic targets with known drugs
4. **Limitations**: Clear documentation of computational constraints
5. **Reproducibility**: Deterministic results with fixed random seeds

### Validation Results

The docking system has been validated on 4 therapeutic targets:

| Target | PDB | Indication | Drugs Tested | Status |
|--------|-----|------------|--------------|--------|
| COX-2 | 1CX2 | Inflammation/Pain | Celecoxib, Diclofenac, Naproxen | ✅ PASS |
| EGFR | 1M17 | Lung Cancer | Erlotinib, Gefitinib, Lapatinib | ✅ PASS |
| HIV-1 Protease | 1HVR | HIV/AIDS | Ritonavir, Indinavir, Saquinavir | ✅ PASS |
| Acetylcholinesterase | 1EVE | Alzheimer's | Donepezil, Rivastigmine, Galantamine | ✅ PASS |

All targets correctly rank known drugs above negative controls with significant energy differences (1.5-4.0 kcal/mol).

> **Important Disclaimer**: Computational predictions require experimental validation. This software identifies candidates for further laboratory study—it does not produce ready-to-use drugs. Any promising candidates must undergo rigorous in vitro and in vivo testing before clinical consideration.

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| Phase 0 | ✅ Complete | Foundation & Architecture |
| Phase 1 | ✅ Complete | Core Engine & Pipeline |
| Phase 2 | 🔄 In Progress | User Experience & Web UI |
| Phase 3 | ⏳ Planned | Community & Distributed Computing |

See [ROADMAP.md](docs/ROADMAP.md) for detailed milestones.

---

## Use Cases

### For Researchers
Screen large compound libraries against your target of interest. Export results in standard formats for further analysis.

### For Patient Advocacy Groups
Focus computational resources on neglected diseases. Generate preliminary data to attract research funding.

### For Students & Educators
Learn computational drug discovery with real tools and data. Hands-on experience with molecular docking and ADMET prediction.

### For Citizen Scientists
Contribute to drug discovery from home. Join a global effort to find new treatments.

---

## License

Apache License 2.0 - see [LICENSE](LICENSE)

This means you can:
- Use commercially
- Modify and distribute
- Use privately
- Use patents

---

## Citation

If you use Open Cure Discovery in your research, please cite:

```bibtex
@software{open_cure_discovery,
  author = {Baratta, Alfredo},
  title = {Open Cure Discovery: Democratizing Drug Discovery},
  year = {2025},
  url = {https://github.com/alfredo-baratta/open-cure-discovery}
}
```

---

## Acknowledgments

This project builds on the work of many open-source projects and databases:
- [RDKit](https://www.rdkit.org/) - Cheminformatics toolkit
- [AutoDock Vina](https://vina.scripps.edu/) - Molecular docking
- [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) - GPU-accelerated docking
- [DeepChem](https://deepchem.io/) - Deep learning for chemistry (optional)
- [Meeko](https://github.com/forlilab/Meeko) - Molecular preparation
- [ChEMBL](https://www.ebi.ac.uk/chembl/) - Bioactivity database
- [RCSB PDB](https://www.rcsb.org/) - Protein structure database
- [ZINC](https://zinc.docking.org/) - Compound database

---

<p align="center">
  <strong>Together, we can accelerate the discovery of cures.</strong>
  <br><br>
  <em>"The best time to plant a tree was 20 years ago. The second best time is now."</em>
</p>
