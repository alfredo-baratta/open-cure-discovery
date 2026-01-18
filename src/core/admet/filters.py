"""
Molecular filters for drug-likeness and safety.

This module provides PAINS filters, toxicophore detection,
and other molecular quality filters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re

from loguru import logger


class FilterType(str, Enum):
    """Types of molecular filters."""
    PAINS = "pains"
    LIPINSKI = "lipinski"
    VEBER = "veber"
    TOXICOPHORE = "toxicophore"
    REACTIVE = "reactive"
    AGGREGATOR = "aggregator"


@dataclass
class FilterResult:
    """Result of applying a molecular filter."""
    passed: bool
    filter_type: FilterType
    alerts: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class PAINSFilter:
    """
    Pan-Assay Interference Compounds (PAINS) filter.

    Identifies compounds that frequently cause false positives
    in biochemical screens.
    """

    # PAINS SMARTS patterns (simplified subset)
    # Full PAINS filter has ~480 patterns
    PAINS_PATTERNS = {
        # PAINS A (most promiscuous)
        "quinone_A": "[#6]1([#6]=[#6][#6](=[O])[#6]=[#6]1=[O])",
        "catechol": "c1(O)c(O)cccc1",
        "rhodanine": "S=C1SC(=O)NC1=O",
        "2_amino_thiazole": "c1csc(N)n1",
        "enone": "[#6]=[#6]-[#6]=[O]",

        # PAINS B
        "thiophene_amino": "c1cc(N)sc1",
        "anil_alk_A": "[CH2]Nc1ccc(N)cc1",
        "indol_3yl_alk": "c1ccc2c(c1)[nH]cc2[CH2]",

        # PAINS C
        "ene_cyano_A": "[CH]=[CH][C]#N",
        "imine_one": "[#6]=[#7]-[#6]=[O]",
        "mannich_A": "[#7]-[CH2]-[#7]",
        "azo": "[#7]=[#7]",
        "diazo": "[#6]=[#7]=[#7]",
    }

    def __init__(self):
        """Initialize PAINS filter."""
        self._rdkit_available = self._check_rdkit()
        self._patterns = {}

        if self._rdkit_available:
            self._compile_patterns()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            return False

    def _compile_patterns(self) -> None:
        """Compile SMARTS patterns."""
        from rdkit import Chem

        for name, smarts in self.PAINS_PATTERNS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    self._patterns[name] = pattern
            except Exception as e:
                logger.warning(f"Failed to compile PAINS pattern {name}: {e}")

    def filter(self, smiles: str) -> FilterResult:
        """
        Check if molecule passes PAINS filter.

        Args:
            smiles: SMILES string.

        Returns:
            FilterResult with pass/fail and alerts.
        """
        if not self._rdkit_available:
            return FilterResult(passed=True, filter_type=FilterType.PAINS)

        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return FilterResult(
                passed=False,
                filter_type=FilterType.PAINS,
                alerts=["Invalid SMILES"],
            )

        alerts = []
        for name, pattern in self._patterns.items():
            if mol.HasSubstructMatch(pattern):
                alerts.append(f"PAINS: {name}")

        return FilterResult(
            passed=len(alerts) == 0,
            filter_type=FilterType.PAINS,
            alerts=alerts,
        )


class LipinskiFilter:
    """
    Lipinski's Rule of Five filter.

    Compounds violating more than one rule have poor oral bioavailability.
    """

    def __init__(self, max_violations: int = 1):
        """
        Initialize Lipinski filter.

        Args:
            max_violations: Maximum allowed violations (default: 1).
        """
        self.max_violations = max_violations
        self._rdkit_available = self._check_rdkit()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            return False

    def filter(self, smiles: str) -> FilterResult:
        """
        Check if molecule passes Lipinski filter.

        Args:
            smiles: SMILES string.

        Returns:
            FilterResult with violations detail.
        """
        if not self._rdkit_available:
            return FilterResult(passed=True, filter_type=FilterType.LIPINSKI)

        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return FilterResult(
                passed=False,
                filter_type=FilterType.LIPINSKI,
                alerts=["Invalid SMILES"],
            )

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        violations = []
        if mw > 500:
            violations.append(f"MW > 500 ({mw:.1f})")
        if logp > 5:
            violations.append(f"LogP > 5 ({logp:.2f})")
        if hbd > 5:
            violations.append(f"HBD > 5 ({hbd})")
        if hba > 10:
            violations.append(f"HBA > 10 ({hba})")

        return FilterResult(
            passed=len(violations) <= self.max_violations,
            filter_type=FilterType.LIPINSKI,
            alerts=violations,
            details={
                "mw": mw,
                "logp": logp,
                "hbd": hbd,
                "hba": hba,
                "num_violations": len(violations),
            },
        )


class ToxicophoreFilter:
    """
    Filter for known toxic structural alerts.

    Identifies structural features associated with toxicity.
    """

    TOXICOPHORE_PATTERNS = {
        # Reactive groups
        "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
        "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",
        "aldehyde": "[CH1](=O)",
        "michael_acceptor_1": "[CH2]=[CH][C,S,N](=O)",
        "epoxide": "C1OC1",
        "aziridine": "C1NC1",
        "alkyl_halide": "[CX4][Cl,Br,I]",

        # Mutagenic alerts
        "nitro_aromatic": "[$(c1ccccc1[N+](=O)[O-])]",
        "aromatic_amine": "[$(c1ccccc1N),$(c1ccc(N)cc1)]",
        "aromatic_nitro": "c[N+](=O)[O-]",
        "azide": "[N-]=[N+]=[N-]",
        "triazene": "[#7]=[#7]=[#7]",

        # Hepatotoxic alerts
        "hydrazine": "[NX3][NX3]",
        "hydroxylamine": "[NX3][OX2H1]",
        "quinone": "C1(=O)C=CC(=O)C=C1",

        # Skin sensitization
        "isocyanate": "[NX2]=[CX2]=[OX1]",
        "isothiocyanate": "[NX2]=[CX2]=[SX1]",
    }

    def __init__(self):
        """Initialize toxicophore filter."""
        self._rdkit_available = self._check_rdkit()
        self._patterns = {}

        if self._rdkit_available:
            self._compile_patterns()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            return False

    def _compile_patterns(self) -> None:
        """Compile SMARTS patterns."""
        from rdkit import Chem

        for name, smarts in self.TOXICOPHORE_PATTERNS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    self._patterns[name] = pattern
            except Exception as e:
                logger.warning(f"Failed to compile toxicophore pattern {name}: {e}")

    def filter(self, smiles: str) -> FilterResult:
        """
        Check if molecule passes toxicophore filter.

        Args:
            smiles: SMILES string.

        Returns:
            FilterResult with toxicophore alerts.
        """
        if not self._rdkit_available:
            return FilterResult(passed=True, filter_type=FilterType.TOXICOPHORE)

        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return FilterResult(
                passed=False,
                filter_type=FilterType.TOXICOPHORE,
                alerts=["Invalid SMILES"],
            )

        alerts = []
        for name, pattern in self._patterns.items():
            if mol.HasSubstructMatch(pattern):
                alerts.append(f"Toxicophore: {name}")

        return FilterResult(
            passed=len(alerts) == 0,
            filter_type=FilterType.TOXICOPHORE,
            alerts=alerts,
        )


class DrugLikenessFilter:
    """
    Combined filter for drug-likeness assessment.

    Combines multiple filters for comprehensive drug-likeness evaluation.
    """

    def __init__(
        self,
        apply_pains: bool = True,
        apply_lipinski: bool = True,
        apply_toxicophore: bool = True,
        lipinski_max_violations: int = 1,
    ):
        """
        Initialize combined filter.

        Args:
            apply_pains: Apply PAINS filter.
            apply_lipinski: Apply Lipinski filter.
            apply_toxicophore: Apply toxicophore filter.
            lipinski_max_violations: Max Lipinski violations allowed.
        """
        self.filters = []

        if apply_pains:
            self.filters.append(PAINSFilter())
        if apply_lipinski:
            self.filters.append(LipinskiFilter(lipinski_max_violations))
        if apply_toxicophore:
            self.filters.append(ToxicophoreFilter())

    def filter(self, smiles: str) -> tuple[bool, list[FilterResult]]:
        """
        Apply all filters to a molecule.

        Args:
            smiles: SMILES string.

        Returns:
            Tuple of (passed_all, list of FilterResults).
        """
        results = []
        passed_all = True

        for f in self.filters:
            result = f.filter(smiles)
            results.append(result)
            if not result.passed:
                passed_all = False

        return passed_all, results

    def filter_batch(
        self,
        smiles_list: list[str],
        progress_callback: Optional[callable] = None,
    ) -> list[tuple[bool, list[FilterResult]]]:
        """
        Apply filters to a batch of molecules.

        Args:
            smiles_list: List of SMILES strings.
            progress_callback: Optional callback(current, total).

        Returns:
            List of (passed_all, results) tuples.
        """
        results = []
        total = len(smiles_list)

        for i, smiles in enumerate(smiles_list):
            result = self.filter(smiles)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results
