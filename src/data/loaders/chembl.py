"""
ChEMBL database loader.

This module provides functionality to download and query
bioactivity data from the ChEMBL database.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import json

import requests
from loguru import logger

from src.core.models import Molecule


@dataclass
class ChEMBLConfig:
    """Configuration for ChEMBL data loading."""
    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    cache_dir: Optional[Path] = None
    timeout: int = 30
    batch_size: int = 100


class ChEMBLLoader:
    """
    Load molecules and bioactivity data from ChEMBL.

    ChEMBL is a manually curated database of bioactive molecules
    with drug-like properties.
    """

    def __init__(self, config: Optional[ChEMBLConfig] = None):
        """Initialize ChEMBL loader."""
        self.config = config or ChEMBLConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "OpenCureDiscovery/0.1",
        })

    def search_target(self, query: str) -> list[dict]:
        """
        Search for protein targets by name or identifier.

        Args:
            query: Search query (gene name, protein name, ChEMBL ID).

        Returns:
            List of matching target records.
        """
        endpoint = f"{self.config.base_url}/target/search"
        params = {"q": query, "format": "json"}

        try:
            response = self.session.get(
                endpoint,
                params=params,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("targets", [])

        except Exception as e:
            logger.error(f"ChEMBL target search failed: {e}")
            return []

    def get_target_molecules(
        self,
        target_chembl_id: str,
        activity_type: str = "IC50",
        max_value: float = 10000,  # nM
        limit: Optional[int] = None,
    ) -> Iterator[Molecule]:
        """
        Get molecules with activity against a target.

        Args:
            target_chembl_id: ChEMBL target ID (e.g., "CHEMBL203").
            activity_type: Type of activity measurement.
            max_value: Maximum activity value (nM).
            limit: Maximum molecules to return.

        Yields:
            Molecule objects with activity data.
        """
        endpoint = f"{self.config.base_url}/activity"

        offset = 0
        count = 0

        while True:
            params = {
                "target_chembl_id": target_chembl_id,
                "standard_type": activity_type,
                "standard_value__lte": max_value,
                "format": "json",
                "limit": self.config.batch_size,
                "offset": offset,
            }

            try:
                response = self.session.get(
                    endpoint,
                    params=params,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()

                activities = data.get("activities", [])
                if not activities:
                    break

                for activity in activities:
                    smiles = activity.get("canonical_smiles")
                    if not smiles:
                        continue

                    mol = Molecule(
                        id=activity.get("molecule_chembl_id", ""),
                        smiles=smiles,
                        name=activity.get("molecule_pref_name"),
                        source="chembl",
                        properties={
                            "activity_type": activity.get("standard_type"),
                            "activity_value": activity.get("standard_value"),
                            "activity_units": activity.get("standard_units"),
                            "assay_chembl_id": activity.get("assay_chembl_id"),
                        },
                    )

                    yield mol
                    count += 1

                    if limit and count >= limit:
                        return

                offset += self.config.batch_size

                # Check if more pages available
                if len(activities) < self.config.batch_size:
                    break

            except Exception as e:
                logger.error(f"ChEMBL activity query failed: {e}")
                break

    def get_approved_drugs(self, limit: Optional[int] = None) -> Iterator[Molecule]:
        """
        Get FDA-approved drugs from ChEMBL.

        Args:
            limit: Maximum number of drugs to return.

        Yields:
            Molecule objects for approved drugs.
        """
        endpoint = f"{self.config.base_url}/molecule"

        offset = 0
        count = 0

        while True:
            params = {
                "max_phase": 4,  # Phase 4 = approved
                "format": "json",
                "limit": self.config.batch_size,
                "offset": offset,
            }

            try:
                response = self.session.get(
                    endpoint,
                    params=params,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()

                molecules = data.get("molecules", [])
                if not molecules:
                    break

                for mol_data in molecules:
                    structures = mol_data.get("molecule_structures", {})
                    smiles = structures.get("canonical_smiles") if structures else None

                    if not smiles:
                        continue

                    mol = Molecule(
                        id=mol_data.get("molecule_chembl_id", ""),
                        smiles=smiles,
                        name=mol_data.get("pref_name"),
                        source="chembl",
                        mol_weight=mol_data.get("molecule_properties", {}).get("full_mwt"),
                        properties={
                            "max_phase": mol_data.get("max_phase"),
                            "molecule_type": mol_data.get("molecule_type"),
                            "first_approval": mol_data.get("first_approval"),
                        },
                    )

                    yield mol
                    count += 1

                    if limit and count >= limit:
                        return

                offset += self.config.batch_size

                if len(molecules) < self.config.batch_size:
                    break

            except Exception as e:
                logger.error(f"ChEMBL approved drugs query failed: {e}")
                break

    def get_molecule_by_id(self, chembl_id: str) -> Optional[Molecule]:
        """
        Get a single molecule by ChEMBL ID.

        Args:
            chembl_id: ChEMBL molecule ID.

        Returns:
            Molecule object or None if not found.
        """
        endpoint = f"{self.config.base_url}/molecule/{chembl_id}"

        try:
            response = self.session.get(
                endpoint,
                params={"format": "json"},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

            structures = data.get("molecule_structures", {})
            smiles = structures.get("canonical_smiles") if structures else None

            if not smiles:
                return None

            return Molecule(
                id=data.get("molecule_chembl_id", ""),
                smiles=smiles,
                name=data.get("pref_name"),
                source="chembl",
            )

        except Exception as e:
            logger.error(f"ChEMBL molecule lookup failed: {e}")
            return None

    def download_target_set(
        self,
        target_chembl_id: str,
        output_path: Path,
        activity_type: str = "IC50",
        max_value: float = 10000,
    ) -> int:
        """
        Download all molecules for a target to a file.

        Args:
            target_chembl_id: ChEMBL target ID.
            output_path: Path to save molecules (SMILES file).
            activity_type: Activity type filter.
            max_value: Maximum activity value.

        Returns:
            Number of molecules downloaded.
        """
        count = 0

        with open(output_path, "w") as f:
            f.write("smiles\tid\tname\tactivity_value\n")

            for mol in self.get_target_molecules(
                target_chembl_id,
                activity_type,
                max_value,
            ):
                activity = mol.properties.get("activity_value", "")
                f.write(f"{mol.smiles}\t{mol.id}\t{mol.name or ''}\t{activity}\n")
                count += 1

        logger.info(f"Downloaded {count} molecules for target {target_chembl_id}")
        return count
