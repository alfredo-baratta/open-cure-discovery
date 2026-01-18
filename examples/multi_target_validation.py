"""
Multi-Target Docking Validation with AutoDock Vina

This script validates the docking system across multiple therapeutic targets
to ensure reliability and accuracy in different scenarios.

Targets tested:
1. COX-2 (Inflammation) - PDB: 1CX2
2. EGFR (Cancer) - PDB: 1M17
3. HIV-1 Protease (Antiviral) - PDB: 1HVR
4. Acetylcholinesterase (Neurological) - PDB: 1EVE

Each target is validated against known drugs with experimental binding data.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import subprocess
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from loguru import logger

# Import our receptor preparation module
from src.core.docking.receptor import ReceptorPreparator, BindingSite


# ============================================================
# CONFIGURATION
# ============================================================

VINA_PATH = Path(__file__).parent.parent / "tools" / "vina.exe"


@dataclass
class TargetDefinition:
    """Definition of a therapeutic target for validation."""
    name: str
    pdb_id: str
    indication: str
    ligand_code: Optional[str]  # Co-crystallized ligand for binding site
    chain: str = "A"
    known_drugs: Dict[str, Dict] = None  # {id: {smiles, name, ic50_nm}}
    negative_controls: Dict[str, Dict] = None  # {id: {smiles, name}}


# ============================================================
# TARGET DEFINITIONS WITH KNOWN DRUGS
# ============================================================

TARGETS = {
    "COX2": TargetDefinition(
        name="Cyclooxygenase-2",
        pdb_id="1CX2",
        indication="Inflammation, Pain",
        ligand_code="S58",  # SC-558 inhibitor
        chain="A",
        known_drugs={
            "celecoxib": {
                "smiles": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
                "name": "Celecoxib (Celebrex)",
                "ic50_nm": 40,
            },
            "diclofenac": {
                "smiles": "OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl",
                "name": "Diclofenac (Voltaren)",
                "ic50_nm": 50,
            },
            "naproxen": {
                "smiles": "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
                "name": "Naproxen (Aleve)",
                "ic50_nm": 1800,
            },
            "ibuprofen": {
                "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "name": "Ibuprofen (Advil)",
                "ic50_nm": 13000,
            },
        },
        negative_controls={
            "caffeine": {
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "name": "Caffeine",
            },
            "metformin": {
                "smiles": "CN(C)C(=N)NC(=N)N",
                "name": "Metformin",
            },
        },
    ),

    "EGFR": TargetDefinition(
        name="Epidermal Growth Factor Receptor",
        pdb_id="1M17",
        indication="Non-Small Cell Lung Cancer",
        ligand_code="AQ4",  # Erlotinib analog
        chain="A",
        known_drugs={
            "erlotinib": {
                "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
                "name": "Erlotinib (Tarceva)",
                "ic50_nm": 2,
            },
            "gefitinib": {
                "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
                "name": "Gefitinib (Iressa)",
                "ic50_nm": 33,
            },
            "lapatinib": {
                "smiles": "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)Cl)Cl",
                "name": "Lapatinib (Tykerb)",
                "ic50_nm": 10,
            },
        },
        negative_controls={
            "aspirin": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "name": "Aspirin",
            },
            "glucose": {
                "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                "name": "Glucose",
            },
        },
    ),

    "HIV_PR": TargetDefinition(
        name="HIV-1 Protease",
        pdb_id="1HVR",
        indication="HIV/AIDS",
        ligand_code="A77",  # A77003 inhibitor
        chain="A",
        known_drugs={
            "ritonavir": {
                "smiles": "CC(C)C(NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)NC(CC(O)C(CC2=CC=CC=C2)NC(=O)OCC3=CN=CS3)CC4=CC=CC=C4",
                "name": "Ritonavir (Norvir)",
                "ic50_nm": 15,
            },
            "indinavir": {
                "smiles": "CC(C)(C)NC(=O)C1CN(CCN1CC(CC(CC2=CC=CC=C2)C(=O)NC3C(CC4=CC=CC=C34)O)O)CC5=CC=CN=C5",
                "name": "Indinavir (Crixivan)",
                "ic50_nm": 5,
            },
            "saquinavir": {
                "smiles": "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(O)C(CC3=CC=CC=C3)NC(=O)C(CC(=O)N)NC(=O)C4=NC5=CC=CC=C5C=C4",
                "name": "Saquinavir (Invirase)",
                "ic50_nm": 10,
            },
        },
        negative_controls={
            "ethanol": {
                "smiles": "CCO",
                "name": "Ethanol",
            },
            "caffeine": {
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "name": "Caffeine",
            },
        },
    ),

    "AChE": TargetDefinition(
        name="Acetylcholinesterase",
        pdb_id="1EVE",
        indication="Alzheimer's Disease",
        ligand_code="E20",  # Donepezil-like
        chain="A",
        known_drugs={
            "donepezil": {
                "smiles": "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC",
                "name": "Donepezil (Aricept)",
                "ic50_nm": 6,
            },
            "rivastigmine": {
                "smiles": "CCN(C)C(=O)OC1=CC=CC(=C1)C(C)N(C)C",
                "name": "Rivastigmine (Exelon)",
                "ic50_nm": 4150,
            },
            "galantamine": {
                "smiles": "CN1CCC23C=CC(CC2OC4=C(C=CC(=C34)C1)OC)O",
                "name": "Galantamine (Razadyne)",
                "ic50_nm": 500,
            },
        },
        negative_controls={
            "sucrose": {
                "smiles": "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
                "name": "Sucrose",
            },
            "nicotine": {
                "smiles": "CN1CCCC1C2=CN=CC=C2",
                "name": "Nicotine",
            },
        },
    ),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def smiles_to_pdbqt(smiles: str, output_path: Path, name: str = "ligand") -> bool:
    """Convert SMILES to PDBQT using Meeko."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                return False

        # Optimize
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass

        # Convert with Meeko
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        if not mol_setups:
            return False

        pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]

        with open(output_path, "w") as f:
            f.write(pdbqt_string)

        return True

    except Exception as e:
        logger.error(f"PDBQT conversion failed: {e}")
        return False


def run_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    binding_site: BindingSite,
    output_path: Path,
    exhaustiveness: int = 8,
) -> Optional[float]:
    """Run AutoDock Vina docking."""
    try:
        cmd = [
            str(VINA_PATH),
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--center_x", str(binding_site.center_x),
            "--center_y", str(binding_site.center_y),
            "--center_z", str(binding_site.center_z),
            "--size_x", str(binding_site.size_x),
            "--size_y", str(binding_site.size_y),
            "--size_z", str(binding_site.size_z),
            "--out", str(output_path),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse output for binding energy
        lines = result.stdout.split("\n")
        found_separator = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("-----"):
                found_separator = True
                continue
            if found_separator and stripped and stripped[0].isdigit():
                parts = stripped.split()
                if len(parts) >= 2:
                    return float(parts[1])

        return None

    except Exception as e:
        logger.error(f"Vina docking failed: {e}")
        return None


def validate_target(
    target_id: str,
    target: TargetDefinition,
    output_dir: Path,
) -> Dict[str, any]:
    """
    Validate docking against a single target.

    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*70}")
    print(f"TARGET: {target.name}")
    print(f"PDB: {target.pdb_id} | Indication: {target.indication}")
    print(f"{'='*70}")

    results = {
        "target_id": target_id,
        "target_name": target.name,
        "pdb_id": target.pdb_id,
        "indication": target.indication,
        "drugs": {},
        "controls": {},
        "passed": False,
    }

    # Prepare receptor
    print(f"\n[1] Preparing receptor...")
    preparator = ReceptorPreparator(output_dir=output_dir)

    try:
        pdbqt_path, info, binding_site = preparator.prepare_from_pdb_id(
            target.pdb_id,
            chain=target.chain,
            ligand_code=target.ligand_code,
        )
    except Exception as e:
        print(f"    ERROR: Failed to prepare receptor: {e}")
        return results

    print(f"    Receptor: {info.num_atoms} atoms")
    print(f"    Binding site center: ({binding_site.center_x:.1f}, {binding_site.center_y:.1f}, {binding_site.center_z:.1f})")
    print(f"    Box size: ({binding_site.size_x:.0f}, {binding_site.size_y:.0f}, {binding_site.size_z:.0f})")

    # Dock known drugs
    print(f"\n[2] Docking known drugs...")
    for drug_id, drug_data in target.known_drugs.items():
        ligand_pdbqt = output_dir / f"{drug_id}.pdbqt"
        output_pdbqt = output_dir / f"{drug_id}_out.pdbqt"

        if not smiles_to_pdbqt(drug_data["smiles"], ligand_pdbqt, drug_id):
            print(f"    {drug_data['name']}: FAILED (preparation)")
            continue

        energy = run_vina(pdbqt_path, ligand_pdbqt, binding_site, output_pdbqt)

        if energy is not None:
            results["drugs"][drug_id] = {
                "name": drug_data["name"],
                "energy": energy,
                "ic50_nm": drug_data.get("ic50_nm"),
            }
            print(f"    {drug_data['name']}: {energy:.2f} kcal/mol (IC50: {drug_data.get('ic50_nm', 'N/A')} nM)")
        else:
            print(f"    {drug_data['name']}: FAILED (docking)")

    # Dock negative controls
    print(f"\n[3] Docking negative controls...")
    for ctrl_id, ctrl_data in target.negative_controls.items():
        ligand_pdbqt = output_dir / f"{ctrl_id}.pdbqt"
        output_pdbqt = output_dir / f"{ctrl_id}_out.pdbqt"

        if not smiles_to_pdbqt(ctrl_data["smiles"], ligand_pdbqt, ctrl_id):
            print(f"    {ctrl_data['name']}: FAILED (preparation)")
            continue

        energy = run_vina(pdbqt_path, ligand_pdbqt, binding_site, output_pdbqt)

        if energy is not None:
            results["controls"][ctrl_id] = {
                "name": ctrl_data["name"],
                "energy": energy,
            }
            print(f"    {ctrl_data['name']}: {energy:.2f} kcal/mol")
        else:
            print(f"    {ctrl_data['name']}: FAILED (docking)")

    # Analyze results
    if results["drugs"] and results["controls"]:
        drug_energies = [d["energy"] for d in results["drugs"].values()]
        ctrl_energies = [c["energy"] for c in results["controls"].values()]

        avg_drug = np.mean(drug_energies)
        avg_ctrl = np.mean(ctrl_energies)

        # Sort all results
        all_results = []
        for drug_id, data in results["drugs"].items():
            all_results.append((drug_id, data["energy"], "drug"))
        for ctrl_id, data in results["controls"].items():
            all_results.append((ctrl_id, data["energy"], "control"))

        all_results.sort(key=lambda x: x[1])

        # Check validation criteria
        top3_types = [r[2] for r in all_results[:3]]
        drugs_in_top3 = sum(1 for t in top3_types if t == "drug")

        results["avg_drug_energy"] = avg_drug
        results["avg_ctrl_energy"] = avg_ctrl
        results["energy_difference"] = avg_ctrl - avg_drug
        results["drugs_in_top3"] = drugs_in_top3

        # Validation passed if drugs have better (more negative) energies
        results["passed"] = (avg_drug < avg_ctrl) and (drugs_in_top3 >= 2)

        print(f"\n[4] Results:")
        print(f"    Average drug energy: {avg_drug:.2f} kcal/mol")
        print(f"    Average control energy: {avg_ctrl:.2f} kcal/mol")
        print(f"    Difference: {avg_ctrl - avg_drug:.2f} kcal/mol")
        print(f"    Drugs in top 3: {drugs_in_top3}/3")
        print(f"    Status: {'PASS' if results['passed'] else 'FAIL'}")

    return results


def main():
    """Run multi-target validation."""
    print("=" * 70)
    print("MULTI-TARGET DOCKING VALIDATION")
    print("Open Cure Discovery - Production Validation Suite")
    print("=" * 70)

    # Check Vina
    if not VINA_PATH.exists():
        print(f"\nERROR: AutoDock Vina not found at {VINA_PATH}")
        print("Please download from: https://github.com/ccsb-scripps/AutoDock-Vina/releases")
        return

    print(f"\nUsing Vina: {VINA_PATH}")
    print(f"Targets to validate: {len(TARGETS)}")

    all_results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for target_id, target in TARGETS.items():
            target_dir = tmpdir / target_id
            target_dir.mkdir(exist_ok=True)

            results = validate_target(target_id, target, target_dir)
            all_results[target_id] = results

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0

    print(f"\n{'Target':<20} {'PDB':<8} {'Indication':<25} {'Status':<10}")
    print("-" * 70)

    for target_id, results in all_results.items():
        status = "PASS" if results["passed"] else "FAIL"
        if results["passed"]:
            passed += 1
        else:
            failed += 1

        print(f"{results['target_name']:<20} {results['pdb_id']:<8} {results['indication']:<25} {status:<10}")

    print("-" * 70)
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(TARGETS)} targets")

    if passed == len(TARGETS):
        print("\n[SUCCESS] ALL TARGETS VALIDATED")
        print("The docking system is working correctly across multiple therapeutic areas.")
    elif passed > 0:
        print(f"\n[PARTIAL] {passed}/{len(TARGETS)} targets validated")
        print("Some targets may need investigation or parameter tuning.")
    else:
        print("\n[FAILURE] No targets validated")
        print("The docking system needs debugging.")

    return all_results


if __name__ == "__main__":
    main()
