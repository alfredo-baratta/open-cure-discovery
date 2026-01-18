"""
ZINC database loader.

This module provides functionality to download and process
molecules from the ZINC database of commercially available compounds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import gzip
import io

import requests
from loguru import logger

from src.core.models import Molecule


@dataclass
class ZINCConfig:
    """Configuration for ZINC data loading."""
    base_url: str = "https://zinc.docking.org"
    cache_dir: Optional[Path] = None
    timeout: int = 60
    chunk_size: int = 8192


class ZINCSubset:
    """ZINC predefined subsets."""
    DRUGLIKE = "druglike"
    LEADLIKE = "leadlike"
    FRAGMENTS = "fragments"
    GOLDILOCKS = "goldilocks"
    BIG_N_GREASY = "big-n-greasy"
    LUGS = "lugs"
    SHARDS = "shards"
    WORLD = "world"


class ZINCLoader:
    """
    Load molecules from the ZINC database.

    ZINC is a free database of commercially-available compounds
    for virtual screening.
    """

    def __init__(self, config: Optional[ZINCConfig] = None):
        """Initialize ZINC loader."""
        self.config = config or ZINCConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "OpenCureDiscovery/0.1",
        })

        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_subset(
        self,
        subset: str,
        output_path: Optional[Path] = None,
        tranches: Optional[list[str]] = None,
    ) -> Path:
        """
        Download a ZINC subset.

        Args:
            subset: Subset name (see ZINCSubset).
            output_path: Where to save the file.
            tranches: Specific tranches to download (e.g., ["AAAA", "AAAB"]).

        Returns:
            Path to downloaded file.
        """
        if output_path is None:
            output_path = self.config.cache_dir or Path(".")
            output_path = output_path / f"zinc_{subset}.smi"

        # ZINC 22 provides tranches
        # Format: /tranches/download?subset={subset}&format=smiles
        if tranches:
            # Download specific tranches
            all_molecules = []
            for tranche in tranches:
                url = f"{self.config.base_url}/tranches/{tranche}.smi"
                try:
                    response = self.session.get(url, timeout=self.config.timeout)
                    response.raise_for_status()
                    all_molecules.append(response.text)
                except Exception as e:
                    logger.warning(f"Failed to download tranche {tranche}: {e}")

            output_path.write_text("\n".join(all_molecules))

        else:
            # Download full subset (can be very large)
            url = f"{self.config.base_url}/tranches/download"
            params = {
                "subset": subset,
                "format": "smiles",
            }

            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout * 10,  # Longer timeout for large files
                    stream=True,
                )
                response.raise_for_status()

                with open(output_path, "w") as f:
                    for chunk in response.iter_content(
                        chunk_size=self.config.chunk_size,
                        decode_unicode=True,
                    ):
                        if chunk:
                            f.write(chunk)

            except Exception as e:
                logger.error(f"Failed to download ZINC subset {subset}: {e}")
                raise

        logger.info(f"Downloaded ZINC {subset} to {output_path}")
        return output_path

    def load_from_file(
        self,
        file_path: Path,
        limit: Optional[int] = None,
    ) -> Iterator[Molecule]:
        """
        Load molecules from a ZINC SMILES file.

        Args:
            file_path: Path to SMILES file.
            limit: Maximum molecules to load.

        Yields:
            Molecule objects.
        """
        count = 0

        # Handle gzipped files
        if file_path.suffix == ".gz":
            opener = lambda p: gzip.open(p, "rt")
        else:
            opener = lambda p: open(p, "r")

        try:
            with opener(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # ZINC format: SMILES ZINC_ID
                    parts = line.split()
                    if len(parts) >= 2:
                        smiles, zinc_id = parts[0], parts[1]
                    else:
                        smiles = parts[0]
                        zinc_id = f"ZINC_{count}"

                    mol = Molecule(
                        id=zinc_id,
                        smiles=smiles,
                        source="zinc",
                    )

                    yield mol
                    count += 1

                    if limit and count >= limit:
                        return

        except Exception as e:
            logger.error(f"Failed to load ZINC file {file_path}: {e}")
            raise

    def search_similar(
        self,
        smiles: str,
        threshold: float = 0.7,
        limit: int = 100,
    ) -> list[Molecule]:
        """
        Search for similar molecules in ZINC.

        Args:
            smiles: Query SMILES.
            threshold: Minimum Tanimoto similarity.
            limit: Maximum results.

        Returns:
            List of similar molecules.
        """
        # Note: ZINC API for similarity search
        url = f"{self.config.base_url}/search/similarity"
        params = {
            "smiles": smiles,
            "threshold": threshold,
            "count": limit,
        }

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            molecules = []
            for line in response.text.strip().split("\n"):
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    mol = Molecule(
                        id=parts[1],
                        smiles=parts[0],
                        source="zinc",
                    )
                    molecules.append(mol)

            return molecules

        except Exception as e:
            logger.error(f"ZINC similarity search failed: {e}")
            return []

    def get_molecule(self, zinc_id: str) -> Optional[Molecule]:
        """
        Get a single molecule by ZINC ID.

        Args:
            zinc_id: ZINC identifier.

        Returns:
            Molecule object or None.
        """
        url = f"{self.config.base_url}/substances/{zinc_id}.smi"

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()

            parts = response.text.strip().split()
            if parts:
                return Molecule(
                    id=zinc_id,
                    smiles=parts[0],
                    source="zinc",
                )

        except Exception as e:
            logger.warning(f"Failed to get ZINC molecule {zinc_id}: {e}")

        return None

    def get_purchasable_info(self, zinc_id: str) -> dict:
        """
        Get purchasability information for a molecule.

        Args:
            zinc_id: ZINC identifier.

        Returns:
            Dictionary with vendor information.
        """
        url = f"{self.config.base_url}/substances/{zinc_id}/purchasability"

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.warning(f"Failed to get purchasability for {zinc_id}: {e}")
            return {}

    def download_druglike_subset(
        self,
        output_dir: Path,
        max_molecules: Optional[int] = None,
    ) -> list[Path]:
        """
        Download the drug-like subset in manageable chunks.

        Args:
            output_dir: Directory to save files.
            max_molecules: Maximum molecules to download.

        Returns:
            List of downloaded file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Drug-like tranches (example - actual tranches would need to be queried)
        # ZINC uses a 2D tranche system: LogP (-1 to 6) x MW (200-500)
        sample_tranches = [
            "AA", "AB", "AC", "AD",  # Low LogP
            "BA", "BB", "BC", "BD",  # Medium LogP
            "CA", "CB", "CC", "CD",  # Higher LogP
        ]

        downloaded = []
        total_molecules = 0

        for tranche in sample_tranches:
            if max_molecules and total_molecules >= max_molecules:
                break

            output_path = output_dir / f"zinc_druglike_{tranche}.smi"

            try:
                url = f"{self.config.base_url}/tranches/{tranche}.smi"
                response = self.session.get(url, timeout=self.config.timeout)

                if response.status_code == 200:
                    output_path.write_text(response.text)
                    downloaded.append(output_path)

                    # Count molecules
                    n_mols = len([l for l in response.text.split("\n") if l.strip()])
                    total_molecules += n_mols
                    logger.info(f"Downloaded tranche {tranche}: {n_mols} molecules")

            except Exception as e:
                logger.warning(f"Failed to download tranche {tranche}: {e}")

        logger.info(f"Downloaded {total_molecules} total molecules from ZINC")
        return downloaded
