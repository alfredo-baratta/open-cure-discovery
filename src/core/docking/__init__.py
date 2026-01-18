"""
Molecular docking module.

This module provides GPU-accelerated molecular docking using AutoDock-GPU
and other docking engines for virtual screening campaigns.

Includes:
- DockingEngine: Abstract base class for docking engines
- AutoDockGPUEngine: GPU-accelerated docking with AutoDock-GPU
- VinaEngine: CPU docking with AutoDock Vina
- ReceptorPreparator: Professional receptor preparation for Vina
- MoleculePreparator: Ligand preparation (SMILES to PDBQT)
"""

from src.core.docking.engine import (
    DockingEngine,
    DockingConfig,
    AutoDockGPUEngine,
    VinaEngine,
)
from src.core.docking.preparation import (
    MoleculePreparator,
    ProteinPreparator,
    PreparationConfig,
)
from src.core.docking.receptor import (
    ReceptorPreparator,
    BindingSite,
    ReceptorInfo,
    prepare_receptor_for_vina,
)

__all__ = [
    # Docking engines
    "DockingEngine",
    "DockingConfig",
    "AutoDockGPUEngine",
    "VinaEngine",
    # Preparation
    "MoleculePreparator",
    "ProteinPreparator",
    "PreparationConfig",
    # Receptor preparation (new)
    "ReceptorPreparator",
    "BindingSite",
    "ReceptorInfo",
    "prepare_receptor_for_vina",
]
