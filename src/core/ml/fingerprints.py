"""
Molecular fingerprint generation.

This module provides various molecular fingerprint generators
for use in similarity search and machine learning models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import numpy as np

from loguru import logger


class FingerprintType(str, Enum):
    """Supported fingerprint types."""
    MORGAN = "morgan"  # Circular fingerprints (ECFP-like)
    RDKIT = "rdkit"  # RDKit topological fingerprints
    MACCS = "maccs"  # MACCS keys (166 bits)
    ATOM_PAIR = "atom_pair"  # Atom pair fingerprints
    TOPOLOGICAL_TORSION = "topological_torsion"


@dataclass
class FingerprintConfig:
    """Configuration for fingerprint generation."""
    fp_type: FingerprintType = FingerprintType.MORGAN
    radius: int = 2  # For Morgan fingerprints
    n_bits: int = 2048  # Bit vector length
    use_features: bool = False  # Use feature invariants
    use_chirality: bool = False  # Include chirality


class FingerprintGenerator:
    """
    Generate molecular fingerprints from SMILES.

    Supports various fingerprint types commonly used in
    cheminformatics and drug discovery.
    """

    def __init__(self, config: Optional[FingerprintConfig] = None):
        """Initialize fingerprint generator."""
        self.config = config or FingerprintConfig()
        self._rdkit_available = self._check_rdkit()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            logger.warning("RDKit not available for fingerprint generation")
            return False

    def generate(self, smiles: str) -> Optional[np.ndarray]:
        """
        Generate fingerprint for a single molecule.

        Args:
            smiles: SMILES string.

        Returns:
            NumPy array with fingerprint bits, or None if failed.
        """
        if not self._rdkit_available:
            raise RuntimeError("RDKit required for fingerprint generation")

        from rdkit import Chem
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
        from rdkit import DataStructs

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        try:
            if self.config.fp_type == FingerprintType.MORGAN:
                # Use the new MorganGenerator API (RDKit 2022+)
                generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.config.radius,
                    fpSize=self.config.n_bits,
                    includeChirality=self.config.use_chirality,
                )
                fp = generator.GetFingerprint(mol)
            elif self.config.fp_type == FingerprintType.RDKIT:
                generator = rdFingerprintGenerator.GetRDKitFPGenerator(
                    fpSize=self.config.n_bits
                )
                fp = generator.GetFingerprint(mol)
            elif self.config.fp_type == FingerprintType.MACCS:
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif self.config.fp_type == FingerprintType.ATOM_PAIR:
                generator = rdFingerprintGenerator.GetAtomPairGenerator(
                    fpSize=self.config.n_bits
                )
                fp = generator.GetFingerprint(mol)
            elif self.config.fp_type == FingerprintType.TOPOLOGICAL_TORSION:
                generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                    fpSize=self.config.n_bits
                )
                fp = generator.GetFingerprint(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {self.config.fp_type}")

            # Convert to numpy array
            arr = np.zeros(len(fp), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr

        except Exception as e:
            logger.error(f"Fingerprint generation failed: {e}")
            return None

    def generate_batch(
        self,
        smiles_list: list[str],
        progress_callback: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Generate fingerprints for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings.
            progress_callback: Optional callback(current, total).

        Returns:
            2D NumPy array of shape (n_molecules, n_bits).
        """
        fingerprints = []
        total = len(smiles_list)

        for i, smiles in enumerate(smiles_list):
            fp = self.generate(smiles)
            if fp is not None:
                fingerprints.append(fp)
            else:
                # Use zero vector for failed molecules
                n_bits = (
                    167 if self.config.fp_type == FingerprintType.MACCS
                    else self.config.n_bits
                )
                fingerprints.append(np.zeros(n_bits, dtype=np.float32))

            if progress_callback:
                progress_callback(i + 1, total)

        return np.vstack(fingerprints)

    def similarity(
        self,
        fp1: np.ndarray,
        fp2: np.ndarray,
        metric: str = "tanimoto",
    ) -> float:
        """
        Calculate similarity between two fingerprints.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.
            metric: Similarity metric ("tanimoto", "dice", "cosine").

        Returns:
            Similarity score (0-1).
        """
        if metric == "tanimoto":
            intersection = np.sum(fp1 * fp2)
            union = np.sum(fp1) + np.sum(fp2) - intersection
            return intersection / union if union > 0 else 0.0

        elif metric == "dice":
            intersection = np.sum(fp1 * fp2)
            total = np.sum(fp1) + np.sum(fp2)
            return 2 * intersection / total if total > 0 else 0.0

        elif metric == "cosine":
            dot = np.dot(fp1, fp2)
            norm1 = np.linalg.norm(fp1)
            norm2 = np.linalg.norm(fp2)
            return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def bulk_similarity(
        self,
        query_fp: np.ndarray,
        database_fps: np.ndarray,
        metric: str = "tanimoto",
        top_k: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate similarity of query against database.

        Args:
            query_fp: Query fingerprint.
            database_fps: Database of fingerprints (2D array).
            metric: Similarity metric.
            top_k: Return only top K results.

        Returns:
            Tuple of (indices, similarities) sorted by similarity.
        """
        n_compounds = database_fps.shape[0]
        similarities = np.zeros(n_compounds, dtype=np.float32)

        for i in range(n_compounds):
            similarities[i] = self.similarity(query_fp, database_fps[i], metric)

        # Sort by similarity (descending)
        sorted_indices = np.argsort(-similarities)

        if top_k:
            sorted_indices = sorted_indices[:top_k]

        return sorted_indices, similarities[sorted_indices]


class MolecularDescriptorCalculator:
    """
    Calculate molecular descriptors for ML models.

    Provides various 2D and 3D molecular descriptors.
    """

    def __init__(self):
        """Initialize descriptor calculator."""
        self._rdkit_available = self._check_rdkit()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            return False

    def calculate(self, smiles: str) -> dict:
        """
        Calculate molecular descriptors.

        Args:
            smiles: SMILES string.

        Returns:
            Dictionary of descriptor name -> value.
        """
        if not self._rdkit_available:
            raise RuntimeError("RDKit required for descriptor calculation")

        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, QED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        try:
            descriptors = {
                # Basic properties
                "mol_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),

                # Lipinski properties
                "num_h_donors": Lipinski.NumHDonors(mol),
                "num_h_acceptors": Lipinski.NumHAcceptors(mol),

                # Ring systems
                "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
                "num_aliphatic_rings": Descriptors.NumAliphaticRings(mol),
                "num_rings": Descriptors.RingCount(mol),

                # Complexity
                "num_heavy_atoms": Descriptors.HeavyAtomCount(mol),
                "num_heteroatoms": Descriptors.NumHeteroatoms(mol),
                "fraction_csp3": Descriptors.FractionCSP3(mol),

                # Charge
                "formal_charge": Chem.GetFormalCharge(mol),

                # Drug-likeness
                "qed": QED.qed(mol),
            }

            # Lipinski rule of 5 violations
            violations = 0
            if descriptors["mol_weight"] > 500:
                violations += 1
            if descriptors["logp"] > 5:
                violations += 1
            if descriptors["num_h_donors"] > 5:
                violations += 1
            if descriptors["num_h_acceptors"] > 10:
                violations += 1
            descriptors["lipinski_violations"] = violations

            return descriptors

        except Exception as e:
            logger.error(f"Descriptor calculation failed: {e}")
            return {}

    def calculate_batch(
        self,
        smiles_list: list[str],
        descriptor_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Calculate descriptors for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings.
            descriptor_names: Specific descriptors to calculate.

        Returns:
            2D NumPy array of descriptors.
        """
        all_descriptors = []

        for smiles in smiles_list:
            desc = self.calculate(smiles)
            all_descriptors.append(desc)

        if not all_descriptors:
            return np.array([])

        # Determine columns
        if descriptor_names:
            columns = descriptor_names
        else:
            columns = list(all_descriptors[0].keys())

        # Build array
        data = []
        for desc in all_descriptors:
            row = [desc.get(col, 0.0) for col in columns]
            data.append(row)

        return np.array(data, dtype=np.float32)
