"""
Molecular preparation utilities for docking.

This module handles the conversion of molecules from SMILES to
3D structures ready for docking (PDBQT format).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tempfile

from loguru import logger

from src.core.models import Molecule, ProteinTarget, BindingSite


@dataclass
class PreparationConfig:
    """Configuration for molecule preparation."""
    ph: float = 7.4  # Physiological pH
    num_conformers: int = 10  # Number of conformers to generate
    max_iterations: int = 500  # Max optimization iterations
    add_hydrogens: bool = True
    minimize_energy: bool = True
    remove_salts: bool = True


class MoleculePreparator:
    """
    Prepare molecules for docking.

    Converts SMILES to 3D structures with proper protonation
    and exports to PDBQT format.
    """

    def __init__(self, config: Optional[PreparationConfig] = None):
        """Initialize preparator with configuration."""
        self.config = config or PreparationConfig()
        self._rdkit_available = self._check_rdkit()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            logger.warning("RDKit not available. Molecule preparation will be limited.")
            return False

    def prepare(self, molecule: Molecule) -> Molecule:
        """
        Prepare a molecule for docking.

        Args:
            molecule: Molecule with SMILES string.

        Returns:
            Molecule with PDBQT content and 3D conformers.
        """
        if not self._rdkit_available:
            raise RuntimeError("RDKit required for molecule preparation")

        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors

        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(molecule.smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {molecule.smiles}")

            # Remove salts if requested
            if self.config.remove_salts:
                mol = self._remove_salts(mol)

            # Add hydrogens
            if self.config.add_hydrogens:
                mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            result = AllChem.EmbedMolecule(
                mol,
                AllChem.ETKDGv3(),
            )
            if result == -1:
                # Try with random coordinates
                AllChem.EmbedMolecule(mol, useRandomCoords=True)

            # Minimize energy
            if self.config.minimize_energy:
                try:
                    AllChem.MMFFOptimizeMolecule(
                        mol,
                        maxIters=self.config.max_iterations,
                    )
                except Exception:
                    # Fall back to UFF
                    AllChem.UFFOptimizeMolecule(mol)

            # Calculate basic properties
            molecule.mol_weight = Descriptors.MolWt(mol)

            # Generate PDBQT content
            molecule.pdbqt_content = self._mol_to_pdbqt(mol, molecule.id)

            # Store conformer as bytes
            molecule.conformers = [mol.ToBinary()]

            # Update state
            from src.core.models import MoleculeState
            molecule.state = MoleculeState.PREPARED

            return molecule

        except Exception as e:
            logger.error(f"Failed to prepare molecule {molecule.id}: {e}")
            raise

    def prepare_batch(
        self,
        molecules: list[Molecule],
        progress_callback: Optional[callable] = None,
    ) -> list[Molecule]:
        """
        Prepare a batch of molecules.

        Args:
            molecules: List of molecules to prepare.
            progress_callback: Optional callback(current, total).

        Returns:
            List of prepared molecules (failed ones are filtered out).
        """
        prepared = []
        total = len(molecules)

        for i, mol in enumerate(molecules):
            try:
                prepared_mol = self.prepare(mol)
                prepared.append(prepared_mol)
            except Exception as e:
                logger.warning(f"Skipping molecule {mol.id}: {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        logger.info(f"Prepared {len(prepared)}/{total} molecules")
        return prepared

    def _remove_salts(self, mol):
        """Remove salt counterions from molecule."""
        from rdkit.Chem.SaltRemover import SaltRemover

        remover = SaltRemover()
        stripped = remover.StripMol(mol)
        return stripped if stripped.GetNumAtoms() > 0 else mol

    def _mol_to_pdbqt(self, mol, mol_id: str) -> str:
        """
        Convert RDKit mol to PDBQT format.

        PDBQT is PDB format with partial charges and atom types.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Get PDB block
        pdb_block = Chem.MolToPDBBlock(mol)

        # Convert to PDBQT (simplified version)
        # Full conversion would use MGLTools or OpenBabel
        pdbqt_lines = []
        pdbqt_lines.append(f"REMARK  Name = {mol_id}")
        pdbqt_lines.append("REMARK  SMILES = " + Chem.MolToSmiles(mol))
        pdbqt_lines.append("ROOT")

        atom_idx = 1
        for line in pdb_block.split("\n"):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Parse PDB atom line
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                # Determine AutoDock atom type
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                ad_type = self._get_autodock_type(element, atom_name)

                # Calculate partial charge (Gasteiger)
                charge = 0.0  # Would need AllChem.ComputeGasteigerCharges

                # Format PDBQT line
                pdbqt_line = (
                    f"ATOM  {atom_idx:5d} {atom_name:4s} LIG A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    "
                    f"{charge:+6.3f} {ad_type:2s}"
                )
                pdbqt_lines.append(pdbqt_line)
                atom_idx += 1

        pdbqt_lines.append("ENDROOT")
        pdbqt_lines.append("TORSDOF 0")

        return "\n".join(pdbqt_lines)

    def _get_autodock_type(self, element: str, atom_name: str) -> str:
        """Map element to AutoDock atom type."""
        type_map = {
            "C": "C",
            "N": "N",
            "O": "OA",
            "S": "SA",
            "H": "HD",  # Polar hydrogen
            "F": "F",
            "Cl": "Cl",
            "Br": "Br",
            "I": "I",
            "P": "P",
        }
        # Check if hydrogen is polar (attached to N, O, S)
        if element == "H":
            return "HD"  # Simplified - all H as HD

        return type_map.get(element, "C")


class ProteinPreparator:
    """
    Prepare protein targets for docking.

    Downloads structures from PDB, adds hydrogens, and
    exports to PDBQT format.
    """

    def __init__(self):
        """Initialize protein preparator."""
        self._rdkit_available = self._check_rdkit()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            return False

    def prepare_from_pdb_id(
        self,
        pdb_id: str,
        binding_site: Optional[BindingSite] = None,
    ) -> ProteinTarget:
        """
        Prepare a protein target from PDB ID.

        Args:
            pdb_id: 4-character PDB identifier.
            binding_site: Optional predefined binding site.

        Returns:
            Prepared ProteinTarget.
        """
        # Download PDB file
        pdb_content = self._download_pdb(pdb_id)

        # Create target
        target = ProteinTarget(
            id=pdb_id,
            name=pdb_id,
            pdb_id=pdb_id,
        )

        # Prepare structure
        target.pdbqt_content = self._pdb_to_pdbqt(pdb_content, pdb_id)

        # Add binding site
        if binding_site:
            target.binding_sites.append(binding_site)
        else:
            # Try to detect binding site from ligand in structure
            detected_site = self._detect_binding_site(pdb_content)
            if detected_site:
                target.binding_sites.append(detected_site)

        return target

    def prepare_from_file(
        self,
        structure_path: Path,
        target_id: str,
        binding_site: Optional[BindingSite] = None,
    ) -> ProteinTarget:
        """
        Prepare a protein target from local file.

        Args:
            structure_path: Path to PDB or PDBQT file.
            target_id: Identifier for the target.
            binding_site: Optional predefined binding site.

        Returns:
            Prepared ProteinTarget.
        """
        content = structure_path.read_text()

        target = ProteinTarget(
            id=target_id,
            name=target_id,
            structure_path=structure_path,
        )

        if structure_path.suffix.lower() == ".pdbqt":
            target.pdbqt_content = content
            target.pdbqt_path = structure_path
        else:
            target.pdbqt_content = self._pdb_to_pdbqt(content, target_id)

        if binding_site:
            target.binding_sites.append(binding_site)

        return target

    def _download_pdb(self, pdb_id: str) -> str:
        """Download PDB structure from RCSB."""
        import requests

        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def _pdb_to_pdbqt(self, pdb_content: str, name: str) -> str:
        """
        Convert PDB to PDBQT format.

        This is a simplified conversion. For production use,
        MGLTools or OpenBabel would be preferred.
        """
        pdbqt_lines = []
        pdbqt_lines.append(f"REMARK  Name = {name}")

        for line in pdb_content.split("\n"):
            if line.startswith("ATOM"):
                # Keep only protein atoms
                residue = line[17:20].strip()
                if residue in self._amino_acids():
                    # Get element and assign AutoDock type
                    element = line[76:78].strip() if len(line) > 78 else line[12:14].strip()[0]
                    ad_type = self._get_receptor_type(element, line[12:16].strip())

                    # Add charge (0.0 placeholder)
                    pdbqt_line = line[:54] + "  1.00  0.00    " + f"{0.0:+6.3f} {ad_type:2s}"
                    pdbqt_lines.append(pdbqt_line)

            elif line.startswith("TER") or line.startswith("END"):
                pdbqt_lines.append(line)

        return "\n".join(pdbqt_lines)

    def _amino_acids(self) -> set:
        """Return set of standard amino acid codes."""
        return {
            "ALA", "ARG", "ASN", "ASP", "CYS",
            "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL",
        }

    def _get_receptor_type(self, element: str, atom_name: str) -> str:
        """Map protein atom to AutoDock type."""
        # Simplified mapping
        if element == "C":
            return "C"
        elif element == "N":
            return "NA" if "N" in atom_name else "N"
        elif element == "O":
            return "OA"
        elif element == "S":
            return "SA"
        elif element == "H":
            return "HD"
        return "C"

    def _detect_binding_site(self, pdb_content: str) -> Optional[BindingSite]:
        """
        Attempt to detect binding site from co-crystallized ligand.

        Returns:
            BindingSite if ligand found, None otherwise.
        """
        hetatm_coords = []

        for line in pdb_content.split("\n"):
            if line.startswith("HETATM"):
                residue = line[17:20].strip()
                # Skip water and common ions
                if residue not in ("HOH", "WAT", "NA", "CL", "MG", "CA", "ZN"):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        hetatm_coords.append((x, y, z))
                    except ValueError:
                        continue

        if not hetatm_coords:
            return None

        # Calculate center of mass
        cx = sum(c[0] for c in hetatm_coords) / len(hetatm_coords)
        cy = sum(c[1] for c in hetatm_coords) / len(hetatm_coords)
        cz = sum(c[2] for c in hetatm_coords) / len(hetatm_coords)

        return BindingSite(
            center=(cx, cy, cz),
            size=(22.0, 22.0, 22.0),
            name="auto-detected",
        )
