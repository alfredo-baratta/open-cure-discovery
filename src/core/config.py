"""
Configuration management for Open Cure Discovery.

This module provides Pydantic models for validating and managing
configuration settings for screening campaigns.
"""

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DiseaseArea(str, Enum):
    """Supported disease focus areas."""

    ONCOLOGY = "oncology"
    NEURODEGENERATIVE = "neurodegenerative"
    INFECTIOUS = "infectious"
    CARDIOVASCULAR = "cardiovascular"
    AUTOIMMUNE = "autoimmune"
    METABOLIC = "metabolic"
    CUSTOM = "custom"


class DockingEngine(str, Enum):
    """Supported molecular docking engines."""

    AUTODOCK_GPU = "autodock-gpu"
    VINA_GPU = "vina-gpu"
    AUTODOCK_VINA = "autodock-vina"  # CPU fallback


class OutputFormat(str, Enum):
    """Supported output file formats."""

    CSV = "csv"
    JSON = "json"
    SDF = "sdf"
    MOL2 = "mol2"


class HardwareConfig(BaseModel):
    """Hardware configuration settings."""

    gpu_memory_limit: int = Field(
        default=5000,
        ge=1000,
        le=48000,
        description="Maximum GPU memory to use in MB",
    )
    batch_size: int | Literal["auto"] = Field(
        default="auto",
        description="Batch size for docking (or 'auto' for automatic)",
    )
    num_cpu_workers: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Number of CPU worker threads",
    )
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU acceleration",
    )


class DockingConfig(BaseModel):
    """Molecular docking configuration."""

    engine: DockingEngine = Field(
        default=DockingEngine.AUTODOCK_GPU,
        description="Docking engine to use",
    )
    exhaustiveness: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Search exhaustiveness (higher = slower but more accurate)",
    )
    num_poses: int = Field(
        default=9,
        ge=1,
        le=20,
        description="Number of binding poses to generate",
    )
    energy_range: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Maximum energy range for poses (kcal/mol)",
    )


class ScoringConfig(BaseModel):
    """Scoring and ranking configuration."""

    # Weights for composite score (must sum to 1.0)
    weight_docking: float = Field(default=0.4, ge=0.0, le=1.0)
    weight_ml_binding: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_admet: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_novelty: float = Field(default=0.1, ge=0.0, le=1.0)

    # Filtering thresholds
    score_threshold: float = Field(
        default=-7.0,
        description="Minimum docking score to keep (kcal/mol, more negative = better)",
    )
    top_candidates: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Number of top candidates to keep",
    )

    # Filters
    apply_pains_filter: bool = Field(default=True)
    apply_lipinski_filter: bool = Field(default=True)
    apply_toxicity_filter: bool = Field(default=True)

    @field_validator("weight_docking", "weight_ml_binding", "weight_admet", "weight_novelty")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Ensure weight is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weight must be between 0 and 1")
        return v


class OutputConfig(BaseModel):
    """Output configuration."""

    formats: list[OutputFormat] = Field(
        default=[OutputFormat.CSV, OutputFormat.SDF],
        description="Output file formats",
    )
    save_poses: bool = Field(
        default=True,
        description="Save 3D binding poses",
    )
    save_all_scores: bool = Field(
        default=False,
        description="Save all intermediate scores (increases file size)",
    )
    compress_output: bool = Field(
        default=False,
        description="Compress output files with gzip",
    )


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Project name",
    )
    disease: str = Field(
        ...,
        description="Target disease or disease area",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional project description",
    )


class TargetConfig(BaseModel):
    """Target protein configuration."""

    pdb_id: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9A-Za-z]{4}$",
        description="PDB ID of target structure",
    )
    structure_file: Optional[Path] = Field(
        default=None,
        description="Path to local structure file",
    )
    binding_site_center: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="Center of binding site (x, y, z)",
    )
    binding_site_size: tuple[float, float, float] = Field(
        default=(20.0, 20.0, 20.0),
        description="Size of binding site box (x, y, z) in Angstroms",
    )

    @field_validator("pdb_id", "structure_file")
    @classmethod
    def validate_target_source(cls, v, info):
        """At least one target source must be specified."""
        return v


class MoleculeLibraryConfig(BaseModel):
    """Molecule library configuration."""

    name: str = Field(
        default="custom",
        description="Library name or identifier",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source database (zinc, chembl, pubchem, custom)",
    )
    file_path: Optional[Path] = Field(
        default=None,
        description="Path to local molecule file",
    )
    max_molecules: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum molecules to screen (None = all)",
    )


class CampaignConfig(BaseModel):
    """Complete screening campaign configuration."""

    project: ProjectConfig
    target: TargetConfig
    molecules: MoleculeLibraryConfig
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    docking: DockingConfig = Field(default_factory=DockingConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "CampaignConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)


# Pre-configured disease presets
DISEASE_PRESETS: dict[str, dict] = {
    "lung-cancer": {
        "targets": ["EGFR", "ALK", "KRAS-G12C", "MET"],
        "scoring": {
            "weight_docking": 0.35,
            "weight_ml_binding": 0.35,
            "weight_admet": 0.2,
            "weight_novelty": 0.1,
        },
    },
    "breast-cancer": {
        "targets": ["HER2", "ER-alpha", "CDK4", "CDK6"],
        "scoring": {
            "weight_docking": 0.35,
            "weight_ml_binding": 0.35,
            "weight_admet": 0.2,
            "weight_novelty": 0.1,
        },
    },
    "alzheimer": {
        "targets": ["Beta-Secretase", "Gamma-Secretase", "Tau", "AChE"],
        "scoring": {
            "weight_docking": 0.3,
            "weight_ml_binding": 0.3,
            "weight_admet": 0.3,  # Higher ADMET weight for CNS drugs
            "weight_novelty": 0.1,
        },
    },
    "covid-19": {
        "targets": ["Main-Protease", "PLpro", "RdRp", "Spike-RBD"],
        "scoring": {
            "weight_docking": 0.4,
            "weight_ml_binding": 0.3,
            "weight_admet": 0.2,
            "weight_novelty": 0.1,
        },
    },
}


def get_preset(disease: str) -> Optional[dict]:
    """Get preset configuration for a disease."""
    return DISEASE_PRESETS.get(disease.lower())
