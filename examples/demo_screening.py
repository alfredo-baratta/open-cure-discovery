#!/usr/bin/env python3
"""
Demo screening script for Open Cure Discovery.

This script demonstrates how to run a basic virtual screening
campaign using the Open Cure Discovery pipeline.

Usage:
    python examples/demo_screening.py

Requirements:
    - RDKit (for molecular operations)
    - NumPy
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.core.models import Molecule, ProteinTarget, BindingSite
from src.core.pipeline import ScreeningPipeline, PipelineConfig, PipelineProgress


# Sample drug-like molecules (SMILES)
SAMPLE_MOLECULES = [
    ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("acetaminophen", "CC(=O)NC1=CC=C(C=C1)O"),
    ("naproxen", "COC1=CC=C2C=C(C=CC2=C1)C(C)C(=O)O"),
    ("diclofenac", "OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl"),
    ("celecoxib", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"),
    ("atorvastatin", "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"),
    ("metformin", "CN(C)C(=N)NC(=N)N"),
    ("omeprazole", "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC3=NC=C(C=C3C)OC"),
]


def create_sample_molecules() -> list[Molecule]:
    """Create sample molecules for demo."""
    molecules = []
    for name, smiles in SAMPLE_MOLECULES:
        mol = Molecule(
            id=f"demo_{name}",
            smiles=smiles,
            name=name,
            source="demo",
        )
        molecules.append(mol)
    return molecules


def create_sample_target() -> ProteinTarget:
    """Create a sample protein target."""
    # EGFR kinase - common cancer target
    binding_site = BindingSite(
        center=(21.0, 2.0, -17.0),
        size=(22.0, 22.0, 22.0),
        name="ATP_binding_site",
    )

    target = ProteinTarget(
        id="EGFR_demo",
        name="Epidermal Growth Factor Receptor (Demo)",
        pdb_id="1M17",  # EGFR with erlotinib
        gene_name="EGFR",
        disease_area="oncology",
        binding_sites=[binding_site],
    )

    return target


def progress_callback(progress: PipelineProgress) -> None:
    """Callback to display progress."""
    pct = progress.progress_percent
    stage = progress.current_stage
    rate = progress.molecules_per_second

    print(f"\r[{pct:5.1f}%] Stage: {stage:<20} | "
          f"Processed: {progress.processed_molecules}/{progress.total_molecules} | "
          f"Rate: {rate:.1f} mol/s", end="", flush=True)


def main():
    """Run demo screening."""
    print("=" * 60)
    print("Open Cure Discovery - Demo Screening")
    print("=" * 60)
    print()

    # Create sample data
    print("Creating sample molecules...")
    molecules = create_sample_molecules()
    print(f"  Created {len(molecules)} molecules")

    print("Creating sample target...")
    target = create_sample_target()
    print(f"  Target: {target.name}")
    print()

    # Configure pipeline
    # Note: Docking is disabled in demo since AutoDock-GPU may not be installed
    config = PipelineConfig(
        batch_size=5,
        use_gpu=False,  # Use CPU for demo
        run_docking=False,  # Disable docking (requires AutoDock)
        run_ml_prediction=True,
        run_admet=True,
        run_filters=True,
        top_n=10,
        min_score=0.0,
        output_dir=Path("demo_output"),
    )

    print("Pipeline configuration:")
    print(f"  - Docking: {'Enabled' if config.run_docking else 'Disabled (demo mode)'}")
    print(f"  - ML Prediction: {'Enabled' if config.run_ml_prediction else 'Disabled'}")
    print(f"  - ADMET: {'Enabled' if config.run_admet else 'Disabled'}")
    print(f"  - Filters: {'Enabled' if config.run_filters else 'Disabled'}")
    print()

    # Create and run pipeline
    print("Running screening pipeline...")
    print("-" * 60)

    pipeline = ScreeningPipeline(config)
    pipeline.set_progress_callback(progress_callback)

    try:
        results = pipeline.run(molecules, target, campaign_id="demo_campaign")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nNote: Some features require RDKit. Install with:")
        print("  pip install rdkit")
        return 1

    print()  # New line after progress
    print("-" * 60)
    print()

    # Display results
    print("RESULTS")
    print("=" * 60)
    print(f"Total screened: {results.total_screened}")
    print(f"Passed filters: {results.total_passed}")
    print(f"Duration: {results.duration_seconds:.2f} seconds")
    print()

    print("Top Candidates:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Name':<15} {'Score':<10} {'ADMET':<10} {'ML Binding':<10}")
    print("-" * 60)

    for candidate in results.candidates[:10]:
        name = candidate.molecule.name or candidate.molecule.id
        print(f"{candidate.rank:<6} {name:<15} {candidate.final_score:<10.4f} "
              f"{candidate.admet_score:<10.4f} {candidate.ml_binding_score:<10.4f}")

    print()
    print(f"Results saved to: {config.output_dir}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
