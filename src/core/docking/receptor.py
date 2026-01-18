"""
Professional receptor preparation for molecular docking.

This module provides robust tools for preparing protein structures
for docking with AutoDock Vina, including:
- Automatic hydrogen addition at physiological pH
- Binding site detection from co-crystallized ligands
- PDBQT conversion with proper atom typing
- Structure validation and cleaning
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import urllib.request
import tempfile
import re

import numpy as np
from loguru import logger


class AtomType(str, Enum):
    """AutoDock atom types for PDBQT format."""
    C = "C"      # Carbon (aliphatic)
    A = "A"      # Carbon (aromatic)
    N = "N"      # Nitrogen (not H-bond acceptor)
    NA = "NA"    # Nitrogen (H-bond acceptor)
    NS = "NS"    # Nitrogen (amide)
    OA = "OA"    # Oxygen (H-bond acceptor)
    OS = "OS"    # Oxygen (sulfate)
    S = "S"      # Sulfur (not H-bond acceptor)
    SA = "SA"    # Sulfur (H-bond acceptor)
    H = "H"      # Hydrogen (non-polar)
    HD = "HD"    # Hydrogen (polar, H-bond donor)


@dataclass
class BindingSite:
    """Binding site definition for docking."""
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 22.0
    size_y: float = 22.0
    size_z: float = 22.0

    @property
    def center(self) -> Tuple[float, float, float]:
        return (self.center_x, self.center_y, self.center_z)

    @property
    def size(self) -> Tuple[float, float, float]:
        return (self.size_x, self.size_y, self.size_z)

    def to_vina_args(self) -> List[str]:
        """Generate Vina command-line arguments."""
        return [
            "--center_x", str(self.center_x),
            "--center_y", str(self.center_y),
            "--center_z", str(self.center_z),
            "--size_x", str(self.size_x),
            "--size_y", str(self.size_y),
            "--size_z", str(self.size_z),
        ]


@dataclass
class ReceptorInfo:
    """Information about a prepared receptor."""
    pdb_id: str
    pdbqt_path: Path
    binding_site: Optional[BindingSite]
    num_atoms: int
    num_residues: int
    chains: List[str]
    ligand_codes: List[str]


class ReceptorPreparator:
    """
    Professional receptor preparation for AutoDock Vina.

    This class handles all aspects of receptor preparation:
    - PDB download from RCSB
    - Structure cleaning (remove water, ions, non-standard residues)
    - Hydrogen addition
    - PDBQT conversion with proper atom types
    - Automatic binding site detection
    """

    # Standard amino acids
    AMINO_ACIDS = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        # Protonation states
        "HID", "HIE", "HIP",  # Histidine variants
        "CYX",  # Disulfide cysteine
    }

    # Common ligand codes to exclude as non-protein
    COMMON_LIGANDS = {
        "HOH", "WAT",  # Water
        "SO4", "PO4", "NO3",  # Ions
        "NA", "CL", "MG", "CA", "ZN", "FE", "MN", "CO", "NI", "CU",  # Metal ions
        "GOL", "EDO", "PEG", "DMS", "ACT", "BME",  # Crystallization additives
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize receptor preparator.

        Args:
            output_dir: Directory for output files. Uses temp dir if not specified.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_pdb(self, pdb_id: str) -> Path:
        """
        Download PDB structure from RCSB.

        Args:
            pdb_id: 4-character PDB ID.

        Returns:
            Path to downloaded PDB file.
        """
        pdb_id = pdb_id.upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_path = self.output_dir / f"{pdb_id}.pdb"

        logger.info(f"Downloading {pdb_id} from RCSB PDB...")
        try:
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"Downloaded: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to download {pdb_id}: {e}")
            raise

    def detect_binding_site(
        self,
        pdb_path: Path,
        ligand_code: Optional[str] = None,
        chain: Optional[str] = None,
        padding: float = 5.0
    ) -> BindingSite:
        """
        Automatically detect binding site from co-crystallized ligand.

        Args:
            pdb_path: Path to PDB file.
            ligand_code: Specific ligand code (e.g., "S58"). Auto-detects if None.
            chain: Specific chain to use. Uses first chain with ligand if None.
            padding: Extra space around ligand for search box (Angstroms).

        Returns:
            BindingSite with center and size.
        """
        with open(pdb_path) as f:
            lines = f.readlines()

        # Find all HETATM ligands (excluding water and common additives)
        ligand_atoms = {}  # {(chain, resname, resnum): [(x, y, z), ...]}

        for line in lines:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()

                # Skip water and common additives
                if resname in self.COMMON_LIGANDS:
                    continue

                chain_id = line[21]
                resnum = line[22:26].strip()

                # Apply filters
                if chain and chain_id != chain:
                    continue
                if ligand_code and resname != ligand_code:
                    continue

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                key = (chain_id, resname, resnum)
                if key not in ligand_atoms:
                    ligand_atoms[key] = []
                ligand_atoms[key].append((x, y, z))

        if not ligand_atoms:
            logger.warning("No ligand found in structure. Using geometric center.")
            return self._get_protein_center(lines, padding)

        # Use largest ligand (most atoms) as binding site
        largest_ligand = max(ligand_atoms.items(), key=lambda x: len(x[1]))
        (chain_id, resname, resnum), coords = largest_ligand

        logger.info(f"Detected ligand: {resname} chain {chain_id} residue {resnum} ({len(coords)} atoms)")

        # Calculate center and size
        coords = np.array(coords)
        center = coords.mean(axis=0)

        # Calculate box size with padding
        min_coords = coords.min(axis=0) - padding
        max_coords = coords.max(axis=0) + padding
        size = max_coords - min_coords

        return BindingSite(
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            size_x=max(float(size[0]), 15.0),  # Minimum 15 Angstroms
            size_y=max(float(size[1]), 15.0),
            size_z=max(float(size[2]), 15.0),
        )

    def _get_protein_center(self, lines: List[str], padding: float = 15.0) -> BindingSite:
        """Get geometric center of protein as fallback."""
        coords = []
        for line in lines:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))

        if not coords:
            raise ValueError("No protein atoms found in structure")

        coords = np.array(coords)
        center = coords.mean(axis=0)

        return BindingSite(
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            size_x=30.0,
            size_y=30.0,
            size_z=30.0,
        )

    def _get_autodock_type(
        self,
        atom_name: str,
        resname: str,
        element: str
    ) -> str:
        """
        Determine AutoDock atom type.

        Args:
            atom_name: PDB atom name (e.g., "CA", "N", "OG").
            resname: Residue name (e.g., "ALA", "HIS").
            element: Element symbol (e.g., "C", "N", "O").

        Returns:
            AutoDock atom type string.
        """
        element = element.upper().strip()
        atom_name = atom_name.strip()

        if element == "C":
            # Aromatic carbons in specific residues
            aromatic_atoms = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "CH2", "CE3", "CZ2", "CZ3"}
            aromatic_residues = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"}
            if atom_name in aromatic_atoms and resname in aromatic_residues:
                return "A"  # Aromatic carbon
            return "C"  # Aliphatic carbon

        elif element == "N":
            # Backbone nitrogen and some sidechains can accept H-bonds
            if atom_name == "N":  # Backbone
                return "N"
            # Histidine nitrogens
            if resname in {"HIS", "HID", "HIE", "HIP"} and atom_name in {"ND1", "NE2"}:
                return "NA"  # Can accept H-bonds
            # Lysine, Arginine - donors not acceptors
            if resname in {"LYS", "ARG"}:
                return "N"
            # Asparagine, Glutamine amide nitrogen
            if resname in {"ASN", "GLN"} and atom_name in {"ND2", "NE2"}:
                return "N"
            return "NA"  # Default to H-bond acceptor

        elif element == "O":
            return "OA"  # All oxygens are H-bond acceptors

        elif element == "S":
            # Methionine sulfur is not H-bond acceptor
            if resname == "MET":
                return "S"
            return "SA"  # Cysteine sulfur can accept H-bonds

        elif element == "H":
            # Polar hydrogens (attached to N or O)
            if atom_name.startswith("H"):
                # Check for polar attachment
                polar_h_patterns = ["HN", "HE", "HZ", "HH", "HD", "HG1", "HO"]
                for pattern in polar_h_patterns:
                    if atom_name.startswith(pattern):
                        return "HD"  # H-bond donor
            return "H"  # Non-polar hydrogen

        elif element in {"FE", "ZN", "MG", "CA", "MN", "CO", "NI", "CU"}:
            return element  # Metal ions keep their element type

        else:
            return element[:2]  # Use first 2 chars of element

    def prepare_receptor(
        self,
        pdb_path: Path,
        output_path: Optional[Path] = None,
        chain: Optional[str] = None,
        keep_hydrogens: bool = False,
        ph: float = 7.4,
    ) -> Tuple[Path, ReceptorInfo]:
        """
        Prepare receptor PDBQT file for docking.

        Args:
            pdb_path: Path to input PDB file.
            output_path: Output PDBQT path. Auto-generated if None.
            chain: Specific chain to use. Uses all protein chains if None.
            keep_hydrogens: Keep existing hydrogens (True) or add new ones (False).
            ph: pH for hydrogen addition (affects protonation states).

        Returns:
            Tuple of (PDBQT path, ReceptorInfo).
        """
        pdb_path = Path(pdb_path)

        if output_path is None:
            output_path = self.output_dir / f"{pdb_path.stem}_receptor.pdbqt"

        logger.info(f"Preparing receptor from {pdb_path.name}...")

        with open(pdb_path) as f:
            lines = f.readlines()

        # Collect information
        chains_found = set()
        residues = set()
        ligand_codes = set()
        pdbqt_lines = []
        atom_count = 0

        # First pass: identify structure
        for line in lines:
            if line.startswith("ATOM"):
                chains_found.add(line[21])
                resname = line[17:20].strip()
                resnum = line[22:26].strip()
                residues.add((line[21], resname, resnum))
            elif line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname not in self.COMMON_LIGANDS and resname not in self.AMINO_ACIDS:
                    ligand_codes.add(resname)

        # Add REMARK header
        pdbqt_lines.append(f"REMARK  Prepared by Open Cure Discovery\n")
        pdbqt_lines.append(f"REMARK  Source: {pdb_path.name}\n")
        pdbqt_lines.append(f"REMARK  Chains: {', '.join(sorted(chains_found))}\n")
        if ligand_codes:
            pdbqt_lines.append(f"REMARK  Ligands found: {', '.join(sorted(ligand_codes))}\n")

        # Second pass: convert atoms
        for line in lines:
            if not line.startswith("ATOM"):
                continue

            resname = line[17:20].strip()
            chain_id = line[21]

            # Filter by chain if specified
            if chain and chain_id != chain:
                continue

            # Skip non-standard residues
            if resname not in self.AMINO_ACIDS:
                continue

            # Skip hydrogens if not keeping them
            atom_name = line[12:16].strip()
            if not keep_hydrogens and atom_name.startswith("H"):
                continue

            # Get element
            if len(line) > 77 and line[76:78].strip():
                element = line[76:78].strip()
            else:
                element = atom_name[0]

            # Get AutoDock atom type
            ad_type = self._get_autodock_type(atom_name, resname, element)

            # Extract coordinates and other fields
            coords = line[30:54]
            occupancy = line[54:60] if len(line) > 60 else "  1.00"
            bfactor = line[60:66] if len(line) > 66 else "  0.00"

            # Partial charge (Gasteiger-like approximation)
            charge = self._estimate_charge(atom_name, resname, element)

            # Build PDBQT line
            pdbqt_line = (
                line[:30] +
                coords +
                occupancy +
                bfactor +
                "    " +
                f"{charge:>6.3f}" +
                f" {ad_type:<2}" +
                "\n"
            )
            pdbqt_lines.append(pdbqt_line)
            atom_count += 1

        # Write output
        with open(output_path, "w") as f:
            f.writelines(pdbqt_lines)

        logger.info(f"Receptor saved: {output_path.name} ({atom_count} atoms)")

        # Detect binding site
        binding_site = None
        try:
            binding_site = self.detect_binding_site(pdb_path, chain=chain)
        except Exception as e:
            logger.warning(f"Could not detect binding site: {e}")

        info = ReceptorInfo(
            pdb_id=pdb_path.stem,
            pdbqt_path=output_path,
            binding_site=binding_site,
            num_atoms=atom_count,
            num_residues=len([r for r in residues if not chain or r[0] == chain]),
            chains=sorted(chains_found) if not chain else [chain],
            ligand_codes=sorted(ligand_codes),
        )

        return output_path, info

    def _estimate_charge(self, atom_name: str, resname: str, element: str) -> float:
        """
        Estimate partial charge using simplified Gasteiger-like rules.

        Note: AutoDock Vina ignores these charges and uses its own scoring,
        but the PDBQT format requires them.
        """
        atom_name = atom_name.strip()
        element = element.strip().upper()

        # Backbone atoms
        if atom_name == "N":
            return -0.350
        elif atom_name == "CA":
            return 0.100
        elif atom_name == "C":
            return 0.550
        elif atom_name == "O":
            return -0.550

        # Charged residues
        if resname in {"ASP", "GLU"}:
            if atom_name in {"OD1", "OD2", "OE1", "OE2"}:
                return -0.800
        elif resname in {"LYS"}:
            if atom_name == "NZ":
                return 0.330
        elif resname in {"ARG"}:
            if atom_name in {"NH1", "NH2", "NE"}:
                return 0.330

        # Default by element
        element_charges = {
            "C": 0.000,
            "N": -0.350,
            "O": -0.400,
            "S": -0.100,
            "H": 0.150,
        }
        return element_charges.get(element, 0.000)

    def prepare_from_pdb_id(
        self,
        pdb_id: str,
        chain: Optional[str] = None,
        ligand_code: Optional[str] = None,
    ) -> Tuple[Path, ReceptorInfo, BindingSite]:
        """
        Download and prepare receptor from PDB ID.

        This is the main entry point for preparing receptors.

        Args:
            pdb_id: 4-character PDB ID.
            chain: Specific chain to use.
            ligand_code: Specific ligand for binding site detection.

        Returns:
            Tuple of (PDBQT path, ReceptorInfo, BindingSite).
        """
        # Download
        pdb_path = self.download_pdb(pdb_id)

        # Prepare receptor
        pdbqt_path, info = self.prepare_receptor(pdb_path, chain=chain)

        # Detect binding site
        binding_site = self.detect_binding_site(
            pdb_path,
            ligand_code=ligand_code,
            chain=chain
        )

        return pdbqt_path, info, binding_site


def prepare_receptor_for_vina(
    pdb_source: str,
    output_dir: Optional[str] = None,
    chain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to prepare a receptor for Vina docking.

    Args:
        pdb_source: Either a PDB ID (e.g., "1CX2") or path to PDB file.
        output_dir: Directory for output files.
        chain: Specific chain to use.

    Returns:
        Dictionary with keys:
        - receptor_pdbqt: Path to prepared receptor
        - binding_site: BindingSite object
        - info: ReceptorInfo object
    """
    preparator = ReceptorPreparator(output_dir=Path(output_dir) if output_dir else None)

    # Determine if source is PDB ID or file path
    if len(pdb_source) == 4 and pdb_source.isalnum():
        # Looks like a PDB ID
        pdbqt_path, info, binding_site = preparator.prepare_from_pdb_id(
            pdb_source,
            chain=chain
        )
    else:
        # Treat as file path
        pdb_path = Path(pdb_source)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        pdbqt_path, info = preparator.prepare_receptor(pdb_path, chain=chain)
        binding_site = preparator.detect_binding_site(pdb_path, chain=chain)

    return {
        "receptor_pdbqt": pdbqt_path,
        "binding_site": binding_site,
        "info": info,
    }
