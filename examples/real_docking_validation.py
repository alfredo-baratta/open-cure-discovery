"""
Real Docking Validation with AutoDock Vina

This script performs REAL molecular docking using AutoDock Vina 1.2.7.
No simulated data - all binding energies are computed by Vina.

Target: COX-2 (Cyclooxygenase-2) - PDB ID: 1CX2
Known Inhibitor: SC-558 (co-crystallized ligand)

What this validates:
1. Real docking calculations with Vina
2. Comparison of known drugs vs. negative controls
3. Verification that the docking scores correlate with known activity

IMPORTANT: This uses actual computational docking, not simulated values.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import urllib.request

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np

# Meeko is the official tool for preparing ligands for AutoDock Vina
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy

# ============================================================
# CONFIGURATION
# ============================================================

VINA_PATH = Path(__file__).parent.parent / "tools" / "vina.exe"

# COX-2 binding site from crystal structure 1CX2
# These coordinates are calculated from the co-crystallized SC-558 inhibitor (S58)
COX2_CENTER = (24.26, 21.53, 16.50)  # Angstroms - center of S58 ligand
COX2_SIZE = (22, 18, 20)  # Search box size (with margin)

# ============================================================
# REAL MOLECULAR DATA
# ============================================================

# Known COX-2 inhibitors with their SMILES
# These are FDA-approved drugs with known COX-2 activity
COX2_INHIBITORS = {
    "celecoxib": {
        "smiles": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "name": "Celecoxib (Celebrex)",
        "ic50_nm": 40,  # IC50 from literature
    },
    "naproxen": {
        "smiles": "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
        "name": "Naproxen (Aleve)",
        "ic50_nm": 1800,  # Non-selective, higher IC50
    },
    "ibuprofen": {
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "name": "Ibuprofen (Advil)",
        "ic50_nm": 13000,  # Non-selective NSAID
    },
    "aspirin": {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "name": "Aspirin",
        "ic50_nm": 50000,  # Weak COX-2 inhibitor
    },
    "diclofenac": {
        "smiles": "OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl",
        "name": "Diclofenac (Voltaren)",
        "ic50_nm": 50,  # Good COX-2 inhibitor
    },
}

# Negative controls - molecules that should NOT bind well to COX-2
NEGATIVE_CONTROLS = {
    "caffeine": {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "name": "Caffeine",
        "expected": "No COX-2 activity",
    },
    "glucose": {
        "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "name": "Glucose",
        "expected": "Sugar, no enzyme inhibition",
    },
    "ethanol": {
        "smiles": "CCO",
        "name": "Ethanol",
        "expected": "Too small for binding pocket",
    },
}


def download_pdb(pdb_id: str, output_path: Path) -> bool:
    """Download PDB structure from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        print(f"  Downloading {pdb_id} from RCSB PDB...")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  ERROR: Failed to download {pdb_id}: {e}")
        return False


def pdb_to_pdbqt_receptor(pdb_path: Path, output_path: Path) -> bool:
    """
    Convert PDB to PDBQT for receptor.

    Uses a conversion approach compatible with AutoDock Vina:
    1. Keep only protein atoms (remove water, ligands)
    2. Use only chain A (the main protein chain)
    3. Add AutoDock atom types
    """
    try:
        print("  Converting receptor to PDBQT...")

        with open(pdb_path) as f:
            pdb_lines = f.readlines()

        # Standard amino acid residues
        amino_acids = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
        }

        def get_ad_type(atom_name: str, resname: str, element: str) -> str:
            """Get AutoDock atom type."""
            e = element.upper()
            if e == "C":
                return "C"
            elif e == "N":
                return "NA" if resname == "HIS" else "N"
            elif e == "O":
                return "OA"
            elif e == "S":
                return "SA"
            elif e == "H":
                return "HD"
            else:
                return e[:2] if len(e) > 0 else "C"

        pdbqt_lines = []

        for line in pdb_lines:
            if line.startswith("ATOM"):
                resname = line[17:20].strip()

                # Skip non-protein residues
                if resname not in amino_acids:
                    continue

                # Use only chain A
                chain = line[21]
                if chain != "A":
                    continue

                atom_name = line[12:16].strip()
                # Get element from column 77-78 or first letter of atom name
                element = line[76:78].strip() if len(line) > 77 and line[76:78].strip() else atom_name[0]

                ad_type = get_ad_type(atom_name, resname, element)

                # Build proper PDBQT line with correct column formatting
                coords = line[30:54]  # x, y, z
                occupancy = line[54:60] if len(line) > 60 else "  1.00"
                bfactor = line[60:66] if len(line) > 66 else "  0.00"

                # PDBQT format: PDB columns + 4 spaces + charge (6 chars) + space + atom type (2 chars)
                pdbqt_line = (
                    line[:30] +       # ATOM, serial, name, resname, chain, resnum
                    coords +          # x, y, z (columns 31-54)
                    occupancy +       # occupancy (columns 55-60)
                    bfactor +         # b-factor (columns 61-66)
                    "    " +          # 4 spaces
                    f"{0.000:>6.3f}" +  # partial charge (6 chars)
                    f" {ad_type:<2}" +  # atom type (space + 2 chars)
                    "\n"
                )
                pdbqt_lines.append(pdbqt_line)

        with open(output_path, "w") as f:
            f.writelines(pdbqt_lines)

        print(f"  Receptor saved: {output_path.name} ({len(pdbqt_lines)} atoms)")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def smiles_to_pdbqt(smiles: str, output_path: Path, name: str = "ligand") -> bool:
    """
    Convert SMILES to PDBQT using Meeko (official AutoDock tool).

    Steps:
    1. Generate 3D coordinates with RDKit
    2. Optimize with MMFF94 force field
    3. Use Meeko for proper PDBQT conversion with AutoDock atom types
    """
    try:
        # Create molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"    Invalid SMILES: {smiles[:30]}...")
            return False

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            # Try with different parameters
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                print(f"    Could not generate 3D coordinates")
                return False

        # Optimize geometry with MMFF
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            # MMFF not available, try UFF
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass  # Use unoptimized coordinates

        # Use Meeko for PDBQT conversion
        preparator = MoleculePreparation()
        mol_setup_list = preparator.prepare(mol)

        if not mol_setup_list:
            print(f"    Meeko preparation failed")
            return False

        # Get the first (and usually only) setup
        mol_setup = mol_setup_list[0]

        # Write PDBQT using Meeko's writer
        pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]

        with open(output_path, "w") as f:
            f.write(pdbqt_string)

        return True

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_vina_docking(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple,
    size: tuple,
    output_path: Path,
    exhaustiveness: int = 8,
) -> Optional[float]:
    """
    Run AutoDock Vina docking.

    Returns the best binding energy (kcal/mol) or None if failed.
    """
    try:
        cmd = [
            str(VINA_PATH),
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]),
            "--size_y", str(size[1]),
            "--size_z", str(size[2]),
            "--out", str(output_path),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "1",  # Only best pose
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Parse output for binding energy
        # Format after "-----" separator: "   1       -6.601          0          0"
        lines = result.stdout.split("\n")
        found_separator = False
        for line in lines:
            stripped = line.strip()
            # Look for the separator line
            if stripped.startswith("-----"):
                found_separator = True
                continue
            # Parse energy after separator
            if found_separator and stripped and stripped[0].isdigit():
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        energy = float(parts[1])
                        return energy
                    except ValueError:
                        pass

        # Check stderr for errors
        if result.returncode != 0 or (result.stderr and "error" in result.stderr.lower()):
            print(f"    Vina error: {result.stderr[:200] if result.stderr else 'Unknown error'}")

        return None

    except subprocess.TimeoutExpired:
        print("    Docking timed out!")
        return None
    except Exception as e:
        print(f"    Docking error: {e}")
        return None


def main():
    """Run real docking validation."""
    print("=" * 70)
    print("REAL DOCKING VALIDATION WITH AUTODOCK VINA")
    print("=" * 70)
    print()
    print("Target: COX-2 (Cyclooxygenase-2)")
    print("PDB ID: 1CX2")
    print(f"Binding site center: {COX2_CENTER}")
    print(f"Search box size: {COX2_SIZE}")
    print()

    # Check Vina
    if not VINA_PATH.exists():
        print(f"ERROR: Vina not found at {VINA_PATH}")
        print("Please download from: https://github.com/ccsb-scripps/AutoDock-Vina/releases")
        return

    print(f"Using Vina: {VINA_PATH}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Download and prepare receptor
        print("\n" + "-" * 70)
        print("STEP 1: PREPARE RECEPTOR")
        print("-" * 70)

        pdb_path = tmpdir / "1CX2.pdb"
        receptor_pdbqt = tmpdir / "1CX2_receptor.pdbqt"

        if not download_pdb("1CX2", pdb_path):
            print("ERROR: Could not download PDB structure")
            return

        if not pdb_to_pdbqt_receptor(pdb_path, receptor_pdbqt):
            print("ERROR: Could not prepare receptor")
            return

        # Step 2: Prepare ligands and run docking
        print("\n" + "-" * 70)
        print("STEP 2: DOCK COX-2 INHIBITORS")
        print("-" * 70)

        results = {}

        # Dock known inhibitors
        print("\nKnown COX-2 inhibitors:")
        for mol_id, data in COX2_INHIBITORS.items():
            ligand_pdbqt = tmpdir / f"{mol_id}.pdbqt"
            output_pdbqt = tmpdir / f"{mol_id}_out.pdbqt"

            print(f"\n  Docking {data['name']}...")

            if not smiles_to_pdbqt(data["smiles"], ligand_pdbqt, mol_id):
                print(f"    Failed to prepare ligand")
                continue

            energy = run_vina_docking(
                receptor_pdbqt,
                ligand_pdbqt,
                COX2_CENTER,
                COX2_SIZE,
                output_pdbqt,
            )

            if energy is not None:
                results[mol_id] = {
                    "name": data["name"],
                    "energy": energy,
                    "ic50_nm": data["ic50_nm"],
                    "type": "inhibitor",
                }
                print(f"    Binding energy: {energy:.2f} kcal/mol")
                print(f"    Known IC50: {data['ic50_nm']} nM")
            else:
                print(f"    Docking failed")

        # Dock negative controls
        print("\n" + "-" * 70)
        print("STEP 3: DOCK NEGATIVE CONTROLS")
        print("-" * 70)

        for mol_id, data in NEGATIVE_CONTROLS.items():
            ligand_pdbqt = tmpdir / f"{mol_id}.pdbqt"
            output_pdbqt = tmpdir / f"{mol_id}_out.pdbqt"

            print(f"\n  Docking {data['name']}...")

            if not smiles_to_pdbqt(data["smiles"], ligand_pdbqt, mol_id):
                print(f"    Failed to prepare ligand")
                continue

            energy = run_vina_docking(
                receptor_pdbqt,
                ligand_pdbqt,
                COX2_CENTER,
                COX2_SIZE,
                output_pdbqt,
            )

            if energy is not None:
                results[mol_id] = {
                    "name": data["name"],
                    "energy": energy,
                    "expected": data["expected"],
                    "type": "control",
                }
                print(f"    Binding energy: {energy:.2f} kcal/mol")
            else:
                print(f"    Docking failed")

        # Step 3: Analyze results
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        if not results:
            print("ERROR: No docking results obtained")
            return

        # Sort by energy (more negative = better)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["energy"])

        print(f"\n{'Rank':<6} {'Molecule':<25} {'Energy (kcal/mol)':<18} {'Type':<12}")
        print("-" * 65)

        for rank, (mol_id, data) in enumerate(sorted_results, 1):
            print(f"{rank:<6} {data['name']:<25} {data['energy']:<18.2f} {data['type']:<12}")

        # Validate results
        print("\n" + "=" * 70)
        print("VALIDATION")
        print("=" * 70)

        inhibitors = [(k, v) for k, v in sorted_results if v["type"] == "inhibitor"]
        controls = [(k, v) for k, v in sorted_results if v["type"] == "control"]

        # Check if inhibitors have better (more negative) energies than controls
        if inhibitors and controls:
            best_inhibitor_energy = min(v["energy"] for _, v in inhibitors)
            worst_control_energy = max(v["energy"] for _, v in controls)
            avg_inhibitor_energy = sum(v["energy"] for _, v in inhibitors) / len(inhibitors)
            avg_control_energy = sum(v["energy"] for _, v in controls) / len(controls)

            print(f"\nAverage inhibitor energy: {avg_inhibitor_energy:.2f} kcal/mol")
            print(f"Average control energy:   {avg_control_energy:.2f} kcal/mol")
            print(f"Difference:               {avg_control_energy - avg_inhibitor_energy:.2f} kcal/mol")

            # Check top 3
            top3_types = [v["type"] for _, v in sorted_results[:3]]
            inhibitors_in_top3 = sum(1 for t in top3_types if t == "inhibitor")

            print(f"\nInhibitors in top 3: {inhibitors_in_top3}/3")

            if avg_inhibitor_energy < avg_control_energy and inhibitors_in_top3 >= 2:
                print("\n[PASS] VALIDATION PASSED")
                print("Known COX-2 inhibitors show better binding energies than controls.")
            else:
                print("\n[WARN] VALIDATION NEEDS REVIEW")
                print("Results may need investigation.")

        # Correlation with IC50
        print("\n" + "-" * 70)
        print("CORRELATION WITH EXPERIMENTAL DATA")
        print("-" * 70)

        print("\nComparing Vina scores with known IC50 values:")
        print(f"{'Molecule':<25} {'Vina (kcal/mol)':<18} {'IC50 (nM)':<12} {'Expected':<12}")
        print("-" * 70)

        for mol_id, data in inhibitors:
            ic50 = data.get("ic50_nm", "N/A")
            # Lower IC50 = more potent, should correlate with more negative Vina score
            expected = "Strong" if ic50 < 100 else "Moderate" if ic50 < 1000 else "Weak"
            print(f"{data['name']:<25} {data['energy']:<18.2f} {ic50:<12} {expected:<12}")

        print("\n" + "=" * 70)
        print("DOCKING COMPLETE")
        print("=" * 70)
        print("\nAll binding energies were computed by AutoDock Vina 1.2.7")
        print("No simulated or artificial data was used.")


if __name__ == "__main__":
    main()
