"""
PDB (Protein Data Bank) loader.

This module provides functionality to download protein
structures from the RCSB PDB database.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import requests
from loguru import logger

from src.core.models import ProteinTarget, BindingSite


@dataclass
class PDBConfig:
    """Configuration for PDB data loading."""
    base_url: str = "https://files.rcsb.org"
    api_url: str = "https://data.rcsb.org/rest/v1/core"
    cache_dir: Optional[Path] = None
    timeout: int = 30


class PDBLoader:
    """
    Load protein structures from the RCSB PDB database.

    PDB is the primary repository for 3D structural data
    of biological macromolecules.
    """

    def __init__(self, config: Optional[PDBConfig] = None):
        """Initialize PDB loader."""
        self.config = config or PDBConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "OpenCureDiscovery/0.1",
        })

        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_structure(
        self,
        pdb_id: str,
        output_path: Optional[Path] = None,
        file_format: str = "pdb",
    ) -> Path:
        """
        Download a PDB structure file.

        Args:
            pdb_id: 4-character PDB identifier.
            output_path: Where to save the file. If None, uses cache dir.
            file_format: Format to download ("pdb", "cif", "xml").

        Returns:
            Path to downloaded file.

        Raises:
            ValueError: If PDB ID is invalid.
            requests.HTTPError: If download fails.
        """
        pdb_id = pdb_id.upper()

        if not re.match(r"^[0-9A-Z]{4}$", pdb_id):
            raise ValueError(f"Invalid PDB ID: {pdb_id}")

        # Determine output path
        if output_path is None:
            if self.config.cache_dir:
                output_path = self.config.cache_dir / f"{pdb_id}.{file_format}"
            else:
                output_path = Path(f"{pdb_id}.{file_format}")

        # Check cache
        if output_path.exists():
            logger.debug(f"Using cached structure: {output_path}")
            return output_path

        # Download
        url = f"{self.config.base_url}/download/{pdb_id}.{file_format}"

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()

            output_path.write_text(response.text)
            logger.info(f"Downloaded PDB structure: {pdb_id}")

            return output_path

        except requests.HTTPError as e:
            logger.error(f"Failed to download PDB {pdb_id}: {e}")
            raise

    def get_structure_info(self, pdb_id: str) -> dict:
        """
        Get metadata about a PDB structure.

        Args:
            pdb_id: PDB identifier.

        Returns:
            Dictionary with structure metadata.
        """
        pdb_id = pdb_id.upper()
        url = f"{self.config.api_url}/entry/{pdb_id}"

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to get PDB info for {pdb_id}: {e}")
            return {}

    def search_by_gene(self, gene_name: str, organism: str = "Homo sapiens") -> list[str]:
        """
        Search for PDB structures by gene name.

        Args:
            gene_name: Gene symbol (e.g., "EGFR").
            organism: Organism name.

        Returns:
            List of PDB IDs.
        """
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entity_source_organism.rcsb_gene_name.value",
                            "operator": "exact_match",
                            "value": gene_name,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
                            "operator": "exact_match",
                            "value": organism,
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "results_content_type": ["experimental"],
                "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
                "pager": {"start": 0, "rows": 50},
            },
        }

        try:
            response = self.session.post(
                search_url,
                json=query,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

            return [hit["identifier"] for hit in data.get("result_set", [])]

        except Exception as e:
            logger.error(f"PDB search failed for gene {gene_name}: {e}")
            return []

    def load_target(
        self,
        pdb_id: str,
        binding_site: Optional[BindingSite] = None,
    ) -> ProteinTarget:
        """
        Load a protein target from PDB.

        Args:
            pdb_id: PDB identifier.
            binding_site: Predefined binding site (optional).

        Returns:
            ProteinTarget object.
        """
        pdb_id = pdb_id.upper()

        # Download structure
        structure_path = self.download_structure(pdb_id)

        # Get metadata
        info = self.get_structure_info(pdb_id)

        # Extract relevant info
        name = info.get("struct", {}).get("title", pdb_id)
        resolution = info.get("rcsb_entry_info", {}).get("resolution_combined")
        method = info.get("exptl", [{}])[0].get("method")

        # Create target
        target = ProteinTarget(
            id=pdb_id,
            name=name[:100] if name else pdb_id,  # Truncate long titles
            pdb_id=pdb_id,
            structure_path=structure_path,
            resolution=resolution,
            method=method,
        )

        # Add binding site if provided
        if binding_site:
            target.binding_sites.append(binding_site)
        else:
            # Try to detect from ligand
            detected = self._detect_binding_site_from_structure(structure_path)
            if detected:
                target.binding_sites.append(detected)

        return target

    def _detect_binding_site_from_structure(self, pdb_path: Path) -> Optional[BindingSite]:
        """
        Detect binding site from co-crystallized ligand.

        Args:
            pdb_path: Path to PDB file.

        Returns:
            BindingSite if ligand found, None otherwise.
        """
        hetatm_coords = []
        ligand_residues = set()

        # Common solvent/buffer molecules to skip
        skip_residues = {
            "HOH", "WAT", "DOD",  # Water
            "SO4", "PO4", "CL", "NA", "MG", "CA", "ZN", "K",  # Ions
            "GOL", "EDO", "PEG", "MPD", "DMS",  # Solvents
            "ACT", "FMT", "TRS",  # Buffers
        }

        try:
            with open(pdb_path) as f:
                for line in f:
                    if line.startswith("HETATM"):
                        residue = line[17:20].strip()
                        if residue in skip_residues:
                            continue

                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            hetatm_coords.append((x, y, z))
                            ligand_residues.add(residue)
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
                name=f"ligand-{'-'.join(sorted(ligand_residues))}",
            )

        except Exception as e:
            logger.warning(f"Failed to detect binding site: {e}")
            return None

    def get_ligand_smiles(self, pdb_id: str, ligand_id: str) -> Optional[str]:
        """
        Get SMILES for a ligand in a PDB structure.

        Args:
            pdb_id: PDB identifier.
            ligand_id: 3-letter ligand code.

        Returns:
            SMILES string or None.
        """
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}"

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()

            # Try different SMILES fields
            descriptors = data.get("rcsb_chem_comp_descriptor", {})
            smiles = descriptors.get("smiles") or descriptors.get("smiles_stereo")

            return smiles

        except Exception as e:
            logger.warning(f"Failed to get ligand SMILES for {ligand_id}: {e}")
            return None
