"""
Core data models for Open Cure Discovery.

This module defines the fundamental data structures used throughout
the application for representing molecules, proteins, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import hashlib


class MoleculeFormat(str, Enum):
    """Supported molecular file formats."""
    SMILES = "smiles"
    SDF = "sdf"
    MOL2 = "mol2"
    PDB = "pdb"
    PDBQT = "pdbqt"


class MoleculeState(str, Enum):
    """Processing state of a molecule."""
    RAW = "raw"
    PREPARED = "prepared"
    DOCKED = "docked"
    SCORED = "scored"
    FILTERED = "filtered"


@dataclass
class Molecule:
    """
    Represents a small molecule (ligand) for docking.

    Attributes:
        id: Unique identifier for the molecule.
        smiles: SMILES string representation.
        name: Optional human-readable name.
        source: Source database (e.g., 'zinc', 'chembl').
        properties: Dictionary of molecular properties.
    """
    id: str
    smiles: str
    name: Optional[str] = None
    source: Optional[str] = None
    inchi_key: Optional[str] = None
    mol_weight: Optional[float] = None
    properties: dict = field(default_factory=dict)
    state: MoleculeState = MoleculeState.RAW

    # 3D structure data
    conformers: list[bytes] = field(default_factory=list)
    pdbqt_content: Optional[str] = None

    def __post_init__(self):
        """Generate ID from SMILES if not provided."""
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID from SMILES."""
        return hashlib.md5(self.smiles.encode()).hexdigest()[:12]

    @property
    def is_prepared(self) -> bool:
        """Check if molecule is ready for docking."""
        return self.pdbqt_content is not None or len(self.conformers) > 0


@dataclass
class BindingSite:
    """
    Defines a binding site on a protein target.

    Attributes:
        center: (x, y, z) coordinates of binding site center.
        size: (x, y, z) dimensions of the search box.
        residues: Optional list of residue identifiers in the site.
    """
    center: tuple[float, float, float]
    size: tuple[float, float, float] = (20.0, 20.0, 20.0)
    residues: list[str] = field(default_factory=list)
    name: Optional[str] = None

    def to_autodock_config(self) -> dict:
        """Convert to AutoDock grid configuration."""
        return {
            "center_x": self.center[0],
            "center_y": self.center[1],
            "center_z": self.center[2],
            "size_x": self.size[0],
            "size_y": self.size[1],
            "size_z": self.size[2],
        }


@dataclass
class ProteinTarget:
    """
    Represents a protein target for molecular docking.

    Attributes:
        id: Unique identifier (often PDB ID).
        name: Human-readable name.
        pdb_id: PDB database identifier.
        structure_path: Path to structure file.
        binding_sites: List of defined binding sites.
    """
    id: str
    name: str
    pdb_id: Optional[str] = None
    gene_name: Optional[str] = None
    organism: str = "Homo sapiens"
    disease_area: Optional[str] = None

    # Structure data
    structure_path: Optional[Path] = None
    pdbqt_path: Optional[Path] = None
    pdbqt_content: Optional[str] = None

    # Binding sites
    binding_sites: list[BindingSite] = field(default_factory=list)
    primary_binding_site: Optional[int] = 0  # Index of primary site

    # Metadata
    resolution: Optional[float] = None  # Angstroms
    method: Optional[str] = None  # X-ray, Cryo-EM, etc.

    @property
    def primary_site(self) -> Optional[BindingSite]:
        """Get the primary binding site."""
        if self.binding_sites and self.primary_binding_site is not None:
            return self.binding_sites[self.primary_binding_site]
        return None

    @property
    def is_prepared(self) -> bool:
        """Check if target is ready for docking."""
        return self.pdbqt_content is not None or self.pdbqt_path is not None


@dataclass
class DockingPose:
    """
    Represents a single docking pose (binding conformation).

    Attributes:
        rank: Pose ranking (1 = best).
        energy: Binding energy in kcal/mol.
        coordinates: 3D coordinates of the pose.
        rmsd_lb: RMSD lower bound from best pose.
        rmsd_ub: RMSD upper bound from best pose.
    """
    rank: int
    energy: float  # kcal/mol
    rmsd_lb: float = 0.0
    rmsd_ub: float = 0.0
    coordinates: Optional[bytes] = None  # Serialized 3D coordinates
    pdbqt_content: Optional[str] = None

    @property
    def is_favorable(self) -> bool:
        """Check if this is a favorable binding pose."""
        return self.energy < -6.0  # Typical threshold


@dataclass
class DockingResult:
    """
    Complete docking result for a molecule-target pair.

    Attributes:
        molecule_id: ID of the docked molecule.
        target_id: ID of the protein target.
        poses: List of docking poses.
        best_energy: Best (lowest) binding energy.
        computation_time: Time taken for docking in seconds.
    """
    molecule_id: str
    target_id: str
    binding_site_name: Optional[str] = None
    poses: list[DockingPose] = field(default_factory=list)
    computation_time: float = 0.0  # seconds
    engine: str = "autodock-gpu"
    timestamp: datetime = field(default_factory=datetime.now)

    # Status
    success: bool = True
    error_message: Optional[str] = None

    @property
    def best_energy(self) -> Optional[float]:
        """Get the best (lowest) binding energy."""
        if not self.poses:
            return None
        return min(pose.energy for pose in self.poses)

    @property
    def best_pose(self) -> Optional[DockingPose]:
        """Get the best scoring pose."""
        if not self.poses:
            return None
        return min(self.poses, key=lambda p: p.energy)

    @property
    def num_poses(self) -> int:
        """Number of poses generated."""
        return len(self.poses)


@dataclass
class ADMETProperties:
    """
    ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.

    All predictions are normalized to 0-1 scale where higher = better drug-likeness.
    """
    # Absorption
    intestinal_absorption: Optional[float] = None  # 0-1
    caco2_permeability: Optional[float] = None  # 0-1
    pgp_substrate: Optional[float] = None  # Probability

    # Distribution
    vdss: Optional[float] = None  # Volume of distribution
    bbb_permeability: Optional[float] = None  # Blood-brain barrier, 0-1
    plasma_protein_binding: Optional[float] = None  # Fraction bound

    # Metabolism
    cyp2d6_substrate: Optional[float] = None  # Probability
    cyp3a4_substrate: Optional[float] = None  # Probability
    cyp2d6_inhibitor: Optional[float] = None  # Probability
    cyp3a4_inhibitor: Optional[float] = None  # Probability

    # Excretion
    half_life: Optional[float] = None  # Hours
    clearance: Optional[float] = None  # mL/min/kg

    # Toxicity
    herg_inhibition: Optional[float] = None  # Probability (lower = better)
    ames_mutagenicity: Optional[float] = None  # Probability (lower = better)
    hepatotoxicity: Optional[float] = None  # Probability (lower = better)
    skin_sensitization: Optional[float] = None  # Probability (lower = better)

    # Drug-likeness scores
    qed_score: Optional[float] = None  # Quantitative Estimate of Drug-likeness
    lipinski_violations: int = 0

    @property
    def toxicity_score(self) -> float:
        """Calculate composite toxicity score (0-1, lower = less toxic)."""
        scores = []
        if self.herg_inhibition is not None:
            scores.append(self.herg_inhibition)
        if self.ames_mutagenicity is not None:
            scores.append(self.ames_mutagenicity)
        if self.hepatotoxicity is not None:
            scores.append(self.hepatotoxicity)

        if not scores:
            return 0.5  # Unknown
        return sum(scores) / len(scores)

    @property
    def admet_score(self) -> float:
        """Calculate composite ADMET score (0-1, higher = better)."""
        # Start with QED if available
        if self.qed_score is not None:
            base_score = self.qed_score
        else:
            base_score = 0.5

        # Penalize for toxicity
        toxicity_penalty = self.toxicity_score * 0.3

        # Penalize for Lipinski violations
        lipinski_penalty = min(self.lipinski_violations * 0.1, 0.3)

        return max(0.0, base_score - toxicity_penalty - lipinski_penalty)


@dataclass
class ScoredCandidate:
    """
    A fully scored drug candidate with all predictions.

    This represents the final output after docking, ML prediction,
    and ADMET calculation.
    """
    molecule: Molecule
    target_id: str

    # Scores (all normalized to 0-1 where higher = better)
    docking_score: float = 0.0  # Normalized from binding energy
    ml_binding_score: float = 0.0  # ML-predicted binding probability
    admet_score: float = 0.0  # Composite ADMET score
    novelty_score: float = 0.0  # Distance from known drugs

    # Composite score
    final_score: float = 0.0

    # Raw data
    docking_result: Optional[DockingResult] = None
    admet_properties: Optional[ADMETProperties] = None

    # Ranking
    rank: Optional[int] = None

    # Flags
    passes_filters: bool = True
    filter_failures: list[str] = field(default_factory=list)

    @property
    def best_binding_energy(self) -> Optional[float]:
        """Get best binding energy in kcal/mol."""
        if self.docking_result:
            return self.docking_result.best_energy
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            "molecule_id": self.molecule.id,
            "smiles": self.molecule.smiles,
            "name": self.molecule.name,
            "target_id": self.target_id,
            "docking_score": self.docking_score,
            "ml_binding_score": self.ml_binding_score,
            "admet_score": self.admet_score,
            "novelty_score": self.novelty_score,
            "final_score": self.final_score,
            "binding_energy": self.best_binding_energy,
            "rank": self.rank,
            "passes_filters": self.passes_filters,
            "filter_failures": self.filter_failures,
        }
