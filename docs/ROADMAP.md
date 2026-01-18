# Open Cure Discovery - Development Roadmap

> **Last Updated**: December 2025
> **Current Phase**: Phase 1 Complete, Phase 2 In Progress

## Project Phases Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DEVELOPMENT PHASES                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PHASE 0          PHASE 1          PHASE 2          PHASE 3            │
│   Foundation       Core Engine      User Experience  Community           │
│   ──────────       ───────────      ───────────────  ─────────           │
│   ✅ COMPLETE      ✅ COMPLETE      🔄 IN PROGRESS   ⏳ PLANNED          │
│                                                                          │
│   • Research       • Docking        • CLI Interface  • Distributed      │
│   • Architecture   • ML Models      • Web Dashboard  • Validation       │
│   • Data Sources   • ADMET          • Documentation  • Publications     │
│   • Setup          • Scoring        • Tutorials      • Partnerships     │
│                    • Pipeline                                            │
│                                                                          │
│   ═══════════════════╗                                                  │
│                      ▼ We are here                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Foundation ✅ COMPLETE

**Goal**: Establish project infrastructure and validate feasibility

### 0.1 Research & Planning ✅
- [x] Surveyed existing open-source drug discovery tools
- [x] Identified gaps in current solutions
- [x] Defined scientific requirements
- [x] Documented all data sources and licenses

### 0.2 Technical Setup ✅
- [x] Initialized repository with proper structure
- [x] Configured testing framework (pytest)
- [x] Established code quality standards (ruff, mypy, black)
- [x] Created pyproject.toml with all dependencies

### 0.3 Data Source Validation ✅
- [x] ChEMBL API loader implemented
- [x] ZINC subset download validated
- [x] PDB structure retrieval working
- [x] Data download scripts created

### 0.4 Proof of Concept ✅
- [x] Basic docking engine working
- [x] GPU detection and benchmarking utility
- [x] Memory usage validated for 6GB GPUs
- [x] Scientific accuracy verified

**Deliverables**:
- ✅ Repository structure
- ✅ Working PoC for screening pipeline
- ✅ Performance benchmarks in GPU detector
- ✅ Data access documentation

---

## Phase 1: Core Engine ✅ COMPLETE

**Goal**: Build the computational heart of the system

### 1.1 Molecular Docking Module ✅

#### 1.1.1 AutoDock-GPU Integration ✅
- [x] AutoDock-GPU wrapper (`src/core/docking/engine.py`)
- [x] Python API for docking operations
- [x] Batch processing with queue management
- [x] Progress tracking and checkpointing
- [x] VRAM management for 6GB cards

#### 1.1.2 Docking Preparation ✅
- [x] Ligand preparation pipeline (`src/core/docking/preparation.py`)
- [x] Professional receptor preparation (`src/core/docking/receptor.py`)
- [x] Automatic PDB download from RCSB
- [x] Binding site detection from co-crystallized ligands
- [x] Grid box automatic configuration

#### 1.1.3 Results Processing ✅
- [x] Parse docking output files
- [x] Extract binding energies and poses
- [x] Store results in data structures
- [x] Resume capability implemented

### 1.2 Machine Learning Module ✅

#### 1.2.1 Pre-trained Models ✅
- [x] Binding affinity predictor (`src/core/ml/binding.py`)
- [x] DeepChem GNN predictor (`src/core/ml/deepchem_binding.py`) - optional
- [x] Molecular fingerprint generators (`src/core/ml/fingerprints.py`)
- [x] ONNX runtime support for inference
- [x] Heuristic fallback when no model available

#### 1.2.2 Model Serving ✅
- [x] Unified prediction API
- [x] Batch inference for efficiency
- [x] GPU/CPU memory management
- [x] Ensemble predictor support

### 1.3 ADMET Prediction Module ✅

#### 1.3.1 Property Calculators ✅
- [x] Lipinski's Rule of Five
- [x] Molecular weight, LogP, TPSA
- [x] Rotatable bonds, H-bond donors/acceptors
- [x] Drug-likeness scores (QED)

#### 1.3.2 Toxicity Prediction ✅
- [x] hERG channel inhibition prediction
- [x] AMES mutagenicity prediction
- [x] Hepatotoxicity prediction
- [x] Skin sensitization prediction

#### 1.3.3 Pharmacokinetics ✅
- [x] Absorption prediction
- [x] Blood-brain barrier permeability
- [x] Plasma protein binding
- [x] CYP450 metabolism prediction

### 1.4 Scoring & Ranking System ✅

#### 1.4.1 Composite Scoring ✅
- [x] Weighted scoring function (`src/core/scoring/scorer.py`)
- [x] Score normalization across metrics
- [x] Configurable weights per disease type
- [x] Novelty scoring (Tanimoto distance)

#### 1.4.2 Filtering Pipeline ✅
- [x] PAINS filter (`src/core/admet/filters.py`)
- [x] Toxicophore filter
- [x] Lipinski filter
- [x] Configurable strictness

#### 1.4.3 Ranking Engine ✅
- [x] Score-based ranking
- [x] Multi-objective Pareto optimization
- [x] Diversity selection (MaxMin algorithm)
- [x] Top-N selection with configurable N

### 1.5 Pipeline Integration ✅

- [x] Complete screening pipeline (`src/core/pipeline.py`)
- [x] Batch processing with progress tracking
- [x] Checkpointing and resume
- [x] Result export (CSV, JSON, SMILES)

**Deliverables**:
- ✅ Working docking pipeline
- ✅ ML prediction module
- ✅ ADMET calculator
- ✅ Unified scoring system
- ✅ Complete screening pipeline
- ✅ Unit and integration tests

---

## Phase 2: User Experience 🔄 IN PROGRESS

**Goal**: Make the system accessible to non-programmers

### 2.1 Command Line Interface ✅

#### 2.1.1 Core Commands ✅
- [x] `ocd init` - Initialize new project
- [x] `ocd check-gpu` - GPU detection utility
- [x] `ocd download` - Download datasets (structure ready)
- [x] `ocd screen` - Run screening campaign (structure ready)
- [x] `ocd results` - View results (structure ready)

#### 2.1.2 Configuration ✅
- [x] YAML-based configuration files
- [x] Disease-specific presets
- [x] Hardware auto-detection
- [x] Resource usage limits

#### 2.1.3 Progress & Monitoring ✅
- [x] Rich terminal progress display
- [x] ETA calculation
- [x] Checkpoint and resume

### 2.2 Web Dashboard (Planned)

#### 2.2.1 Campaign Management
- [ ] Create/manage screening campaigns
- [ ] Configure targets and libraries
- [ ] Monitor progress in real-time
- [ ] Start/stop/pause controls

#### 2.2.2 Results Visualization
- [ ] Interactive molecule viewer (3Dmol.js)
- [ ] Sortable/filterable results table
- [ ] Score distribution charts
- [ ] Binding pose visualization

### 2.3 Documentation 🔄

#### 2.3.1 User Documentation
- [x] README.md with quick start
- [x] ARCHITECTURE.md
- [x] ROADMAP.md
- [x] TASKS.md
- [ ] Full installation guide
- [ ] Disease-specific guides

#### 2.3.2 Developer Documentation
- [x] Code docstrings
- [ ] API reference (Sphinx)
- [x] CONTRIBUTING.md

---

## Phase 3: Community & Distribution ⏳ PLANNED

**Goal**: Build a global community and maximize impact

### 3.1 Distributed Computing Network
- [ ] P2P protocol design
- [ ] Work unit distribution
- [ ] Result aggregation
- [ ] Volunteer computing

### 3.2 Validation & Quality
- [x] Multi-target validation suite (4 targets)
- [x] Known drug vs control discrimination
- [ ] Cross-validation protocols
- [ ] Benchmark against DUD-E
- [ ] Scientific review process

### 3.3 Community Building
- [ ] Scientific paper
- [ ] Conference presentations
- [ ] Lab validation partnerships

---

## Current Milestone Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| M0 | Repository & PoC complete | ✅ Complete |
| M1 | Single-target docking working | ✅ Complete |
| M2 | ML predictions integrated | ✅ Complete |
| M3 | Full scoring pipeline | ✅ Complete |
| M4 | CLI beta release | ✅ Complete |
| M5 | Web dashboard beta | ⏳ Planned |
| M6 | v1.0 public release | ⏳ Planned |

---

## Success Metrics

### Technical Metrics (Current Status)
- [x] Screen molecules on GTX 1060 ✅
- [x] VRAM usage <5GB during operation ✅
- [x] Multi-target validation passed (COX-2, EGFR, HIV-PR, AChE) ✅
- [x] Known drugs correctly ranked above controls on all targets ✅
- [ ] >90% correlation with experimental binding data (needs validation)
- [ ] <1% false positive rate on toxicity (needs validation)

### Community Metrics (Future)
- [ ] 1000+ GitHub stars
- [ ] 100+ active contributors
- [ ] 10+ disease focus campaigns

---

## How to Contribute

The project is now at a stage where contributions are welcome:

1. **Testing**: Run the validation script and report issues
2. **Documentation**: Help improve guides and tutorials
3. **Science**: Validate methods against experimental data
4. **Features**: Implement Phase 2 features (web dashboard)

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
