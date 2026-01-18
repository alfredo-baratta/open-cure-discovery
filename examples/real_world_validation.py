"""
Real-World Validation: COX-2 Inhibitors

This script validates the Open Cure Discovery pipeline against a well-known
therapeutic target: Cyclooxygenase-2 (COX-2).

COX-2 is an enzyme involved in inflammation and pain. Several approved drugs
target COX-2, including:
- Celecoxib (Celebrex) - FDA approved 1998
- Naproxen (Aleve) - non-selective, but has COX-2 activity
- Ibuprofen (Advil) - non-selective NSAID
- Aspirin - irreversible COX inhibitor
- Meloxicam (Mobic) - preferential COX-2 inhibitor
- Diclofenac (Voltaren) - non-selective NSAID

The validation checks if the system correctly:
1. Ranks known COX-2 inhibitors higher than random molecules
2. Produces consistent ADMET scores for known drugs
3. Identifies drug-like properties correctly

References:
- PDB ID 1CX2: COX-2 structure with selective inhibitor SC-558
- PDB ID 6COX: COX-2 with celecoxib bound
- ChEMBL Target: CHEMBL230 (Cyclooxygenase-2)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import Descriptors

from src.core.models import Molecule, ProteinTarget, BindingSite
from src.core.pipeline import ScreeningPipeline, PipelineConfig
from src.core.admet.calculator import ADMETCalculator
from src.core.ml.fingerprints import FingerprintGenerator, FingerprintConfig, FingerprintType

# ============================================================
# REAL MOLECULAR DATA
# ============================================================

# Known COX-2 inhibitors with their actual SMILES structures
# Source: PubChem, DrugBank, ChEMBL
COX2_INHIBITORS = [
    # Selective COX-2 inhibitors (Coxibs)
    Molecule(
        id="celecoxib",
        smiles="CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        name="Celecoxib (Celebrex)",
    ),
    Molecule(
        id="rofecoxib",
        smiles="CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3",
        name="Rofecoxib (Vioxx - withdrawn)",
    ),
    Molecule(
        id="valdecoxib",
        smiles="CC1=C(C(=NO1)C2=CC=C(C=C2)S(=O)(=O)N)C3=CC=CC=C3",
        name="Valdecoxib (Bextra - withdrawn)",
    ),
    Molecule(
        id="etoricoxib",
        smiles="CC1=NC=C(C=C1)C2=CC=C(C=C2)S(=O)(=O)C3=CC=C(C=C3)Cl",
        name="Etoricoxib (Arcoxia)",
    ),

    # Non-selective NSAIDs with COX-2 activity
    Molecule(
        id="naproxen",
        smiles="CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
        name="Naproxen (Aleve)",
    ),
    Molecule(
        id="ibuprofen",
        smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        name="Ibuprofen (Advil)",
    ),
    Molecule(
        id="diclofenac",
        smiles="OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl",
        name="Diclofenac (Voltaren)",
    ),
    Molecule(
        id="meloxicam",
        smiles="CC1=CN=C(S1)NC(=O)C2=C(C3=CC=CC=C3S(=O)(=O)N2C)O",
        name="Meloxicam (Mobic)",
    ),
    Molecule(
        id="aspirin",
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        name="Aspirin (acetylsalicylic acid)",
    ),
    Molecule(
        id="indomethacin",
        smiles="CC1=C(C2=CC=CC=C2N1C(=O)C3=CC=C(C=C3)Cl)CC(=O)O",
        name="Indomethacin (Indocin)",
    ),
]

# Negative controls - molecules that should NOT be good COX-2 inhibitors
NEGATIVE_CONTROLS = [
    Molecule(
        id="caffeine",
        smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        name="Caffeine (stimulant)",
    ),
    Molecule(
        id="glucose",
        smiles="OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        name="Glucose (sugar)",
    ),
    Molecule(
        id="nicotine",
        smiles="CN1CCC[C@H]1C2=CN=CC=C2",
        name="Nicotine (alkaloid)",
    ),
    Molecule(
        id="cholesterol",
        smiles="CC(C)CCC[C@@H](C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
        name="Cholesterol (lipid)",
    ),
    Molecule(
        id="ethanol",
        smiles="CCO",
        name="Ethanol (alcohol)",
    ),
]

# COX-2 protein target (real PDB structure)
# 1CX2: Human COX-2 with SC-558 inhibitor bound
# Binding site coordinates from the crystal structure
COX2_TARGET = ProteinTarget(
    id="COX2",
    name="Cyclooxygenase-2 (Prostaglandin G/H synthase 2)",
    pdb_id="1CX2",
    organism="Homo sapiens",
    binding_sites=[
        BindingSite(
            name="Active site",
            center=(23.88, 7.13, 66.91),  # Coordinates from 1CX2 ligand
            size=(20.0, 20.0, 20.0),
        )
    ],
)


def print_header(title: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{title}")
    print("-" * 50)


def validate_admet_properties():
    """Validate ADMET calculation for known drugs."""
    print_header("ADMET PROPERTY VALIDATION")
    print("Checking if known drugs have expected drug-like properties...")

    calculator = ADMETCalculator()

    print_section("Known COX-2 Inhibitors - ADMET Properties")
    print(f"{'Drug':<20} {'MW':<8} {'LogP':<6} {'QED':<6} {'Lipinski':<10} {'BBB':<6}")
    print("-" * 70)

    for mol in COX2_INHIBITORS:
        rdkit_mol = Chem.MolFromSmiles(mol.smiles)
        mw = Descriptors.MolWt(rdkit_mol) if rdkit_mol else 0
        logp = Descriptors.MolLogP(rdkit_mol) if rdkit_mol else 0
        props = calculator.calculate(mol.smiles)
        lipinski = "Pass" if props.lipinski_violations == 0 else f"Fail({props.lipinski_violations})"
        bbb = "Yes" if props.bbb_permeability and props.bbb_permeability > 0.5 else "No"
        qed = props.qed_score if props.qed_score else 0
        print(f"{mol.name[:19]:<20} {mw:<8.1f} {logp:<6.2f} {qed:<6.3f} {lipinski:<10} {bbb:<6}")

    print_section("Negative Controls - ADMET Properties")
    print(f"{'Drug':<20} {'MW':<8} {'LogP':<6} {'QED':<6} {'Lipinski':<10} {'BBB':<6}")
    print("-" * 70)

    for mol in NEGATIVE_CONTROLS:
        rdkit_mol = Chem.MolFromSmiles(mol.smiles)
        mw = Descriptors.MolWt(rdkit_mol) if rdkit_mol else 0
        logp = Descriptors.MolLogP(rdkit_mol) if rdkit_mol else 0
        props = calculator.calculate(mol.smiles)
        lipinski = "Pass" if props.lipinski_violations == 0 else f"Fail({props.lipinski_violations})"
        bbb = "Yes" if props.bbb_permeability and props.bbb_permeability > 0.5 else "No"
        qed = props.qed_score if props.qed_score else 0
        print(f"{mol.name[:19]:<20} {mw:<8.1f} {logp:<6.2f} {qed:<6.3f} {lipinski:<10} {bbb:<6}")


def validate_fingerprint_similarity():
    """Validate that COX-2 inhibitors have similar fingerprints."""
    print_header("FINGERPRINT SIMILARITY VALIDATION")
    print("Checking structural similarity between COX-2 inhibitors...")

    config = FingerprintConfig(fp_type=FingerprintType.MORGAN)
    generator = FingerprintGenerator(config=config)

    # Generate fingerprints for all COX-2 inhibitors
    fps = {}
    for mol in COX2_INHIBITORS:
        fp = generator.generate(mol.smiles)
        if fp is not None:
            fps[mol.id] = fp

    # Calculate similarity between celecoxib (reference) and others
    if "celecoxib" in fps:
        reference = fps["celecoxib"]

        print_section("Similarity to Celecoxib (reference COX-2 inhibitor)")
        print(f"{'Molecule':<25} {'Tanimoto Similarity':<20}")
        print("-" * 50)

        similarities = []
        for mol_id, fp in fps.items():
            if mol_id != "celecoxib":
                sim = generator.similarity(reference, fp, metric="tanimoto")
                similarities.append((mol_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        for mol_id, sim in similarities:
            name = next(m.name for m in COX2_INHIBITORS if m.id == mol_id)
            print(f"{name[:24]:<25} {sim:<20.4f}")

        # Also check negative controls
        print_section("Similarity of Negative Controls to Celecoxib")
        for mol in NEGATIVE_CONTROLS:
            fp = generator.generate(mol.smiles)
            if fp is not None:
                sim = generator.similarity(reference, fp, metric="tanimoto")
                print(f"{mol.name[:24]:<25} {sim:<20.4f}")


def run_screening_validation():
    """Run full screening pipeline and validate ranking."""
    print_header("FULL PIPELINE VALIDATION")
    print("Running complete screening pipeline against COX-2 target...")
    print(f"Target: {COX2_TARGET.name}")
    print(f"PDB ID: {COX2_TARGET.pdb_id}")

    # Combine all molecules
    all_molecules = COX2_INHIBITORS + NEGATIVE_CONTROLS

    # Configure pipeline (without docking for now, ML + ADMET only)
    config = PipelineConfig(
        run_docking=False,  # Would require AutoDock-GPU installed
        run_ml_prediction=True,
        run_admet=True,
        run_filters=True,
        use_gpu=False,
        batch_size=10,
        top_n=20,
        min_score=0.0,
    )

    print_section("Pipeline Configuration")
    print(f"  Docking: {config.run_docking}")
    print(f"  ML Prediction: {config.run_ml_prediction}")
    print(f"  ADMET: {config.run_admet}")
    print(f"  Filters: {config.run_filters}")

    # Run pipeline
    print_section("Running Screening...")
    pipeline = ScreeningPipeline(config)
    results = pipeline.run(all_molecules, COX2_TARGET, campaign_id="cox2_validation")

    # Analyze results
    print_section("SCREENING RESULTS")
    print(f"Total molecules screened: {results.total_screened}")
    print(f"Molecules passed filters: {results.total_passed}")
    print(f"Duration: {results.duration_seconds:.2f} seconds")

    print_section("Ranked Candidates")
    print(f"{'Rank':<6} {'ID':<15} {'Name':<30} {'Score':<8} {'ADMET':<8} {'ML Bind':<8}")
    print("-" * 85)

    # Track how many COX-2 inhibitors are in top positions
    cox2_ids = {m.id for m in COX2_INHIBITORS}
    control_ids = {m.id for m in NEGATIVE_CONTROLS}

    cox2_in_top5 = 0
    cox2_in_top10 = 0
    controls_in_top5 = 0

    for i, candidate in enumerate(results.candidates):
        mol_id = candidate.molecule.id
        name = candidate.molecule.name or mol_id

        # Determine if COX-2 inhibitor or control
        marker = ""
        if mol_id in cox2_ids:
            marker = " [COX-2]"
            if i < 5:
                cox2_in_top5 += 1
            if i < 10:
                cox2_in_top10 += 1
        elif mol_id in control_ids:
            marker = " [CTRL]"
            if i < 5:
                controls_in_top5 += 1

        print(f"{candidate.rank:<6} {mol_id:<15} {name[:28]:<30} {candidate.final_score:<8.4f} {candidate.admet_score:<8.4f} {candidate.ml_binding_score:<8.4f}{marker}")

    # Validation summary
    print_header("VALIDATION SUMMARY")

    total_cox2 = len(COX2_INHIBITORS)
    total_controls = len(NEGATIVE_CONTROLS)

    print(f"COX-2 inhibitors in top 5:  {cox2_in_top5}/{total_cox2}")
    print(f"COX-2 inhibitors in top 10: {cox2_in_top10}/{total_cox2}")
    print(f"Negative controls in top 5: {controls_in_top5}/{total_controls}")

    # Determine if validation passed
    # We expect most COX-2 inhibitors to rank higher than controls
    validation_passed = cox2_in_top5 >= 3 and controls_in_top5 <= 2

    if validation_passed:
        print("\n[PASS] VALIDATION PASSED")
        print("The system correctly prioritizes known COX-2 inhibitors over random molecules.")
    else:
        print("\n[WARN] VALIDATION NEEDS REVIEW")
        print("Results may need investigation - check if scoring weights are appropriate.")

    return results


def main():
    """Run all validation tests."""
    print_header("OPEN CURE DISCOVERY - REAL-WORLD VALIDATION")
    print("Target: Cyclooxygenase-2 (COX-2)")
    print("Therapeutic Area: Inflammation, Pain, Arthritis")
    print()
    print("This validation uses real molecular data from approved drugs")
    print("to verify that the screening system produces meaningful results.")

    # Run validations
    validate_admet_properties()
    validate_fingerprint_similarity()
    results = run_screening_validation()

    print_header("VALIDATION COMPLETE")
    print("See above for detailed results.")
    print("\nKey observations:")
    print("1. Known NSAIDs should have good drug-like (ADMET) properties")
    print("2. Selective COX-2 inhibitors (coxibs) should show structural similarity")
    print("3. COX-2 inhibitors should rank higher than unrelated molecules")

    return results


if __name__ == "__main__":
    main()
