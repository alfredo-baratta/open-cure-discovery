"""
ADMET property calculator.

This module provides calculators for Absorption, Distribution,
Metabolism, Excretion, and Toxicity (ADMET) properties.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from loguru import logger

from src.core.models import ADMETProperties


@dataclass
class ADMETConfig:
    """Configuration for ADMET calculations."""
    use_ml_predictions: bool = True  # Use ML models when available
    strict_mode: bool = False  # Fail on calculation errors


class ADMETCalculator:
    """
    Calculate ADMET properties for drug candidates.

    Provides both rule-based and ML-based predictions for
    various pharmacokinetic and toxicity endpoints.
    """

    def __init__(self, config: Optional[ADMETConfig] = None):
        """Initialize ADMET calculator."""
        self.config = config or ADMETConfig()
        self._rdkit_available = self._check_rdkit()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            logger.warning("RDKit not available for ADMET calculations")
            return False

    def calculate(self, smiles: str) -> ADMETProperties:
        """
        Calculate ADMET properties for a molecule.

        Args:
            smiles: SMILES string.

        Returns:
            ADMETProperties object with all predictions.
        """
        if not self._rdkit_available:
            logger.warning("RDKit not available, returning empty properties")
            return ADMETProperties()

        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, QED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return ADMETProperties()

        props = ADMETProperties()

        try:
            # Calculate basic descriptors first
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rotatable = Descriptors.NumRotatableBonds(mol)

            # Drug-likeness (QED)
            props.qed_score = QED.qed(mol)

            # Lipinski violations
            violations = 0
            if mw > 500:
                violations += 1
            if logp > 5:
                violations += 1
            if hbd > 5:
                violations += 1
            if hba > 10:
                violations += 1
            props.lipinski_violations = violations

            # Absorption predictions
            props.intestinal_absorption = self._predict_absorption(
                mw, logp, tpsa, hbd, hba
            )
            props.caco2_permeability = self._predict_caco2(logp, tpsa, mw)
            props.pgp_substrate = self._predict_pgp_substrate(mol, mw, logp)

            # Distribution predictions
            props.bbb_permeability = self._predict_bbb(logp, tpsa, mw, hbd)
            props.plasma_protein_binding = self._predict_ppb(logp, mw)
            props.vdss = self._predict_vdss(logp, mw)

            # Metabolism predictions
            props.cyp2d6_substrate = self._predict_cyp_substrate(mol, "2D6")
            props.cyp3a4_substrate = self._predict_cyp_substrate(mol, "3A4")
            props.cyp2d6_inhibitor = self._predict_cyp_inhibitor(mol, "2D6")
            props.cyp3a4_inhibitor = self._predict_cyp_inhibitor(mol, "3A4")

            # Excretion predictions
            props.half_life = self._predict_half_life(mw, logp)
            props.clearance = self._predict_clearance(mw, logp)

            # Toxicity predictions
            props.herg_inhibition = self._predict_herg(mol, logp, mw)
            props.ames_mutagenicity = self._predict_ames(mol)
            props.hepatotoxicity = self._predict_hepatotoxicity(mol, mw, logp)
            props.skin_sensitization = self._predict_skin_sensitization(mol)

        except Exception as e:
            logger.error(f"ADMET calculation failed: {e}")
            if self.config.strict_mode:
                raise

        return props

    def calculate_batch(
        self,
        smiles_list: list[str],
        progress_callback: Optional[callable] = None,
    ) -> list[ADMETProperties]:
        """
        Calculate ADMET properties for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings.
            progress_callback: Optional callback(current, total).

        Returns:
            List of ADMETProperties objects.
        """
        results = []
        total = len(smiles_list)

        for i, smiles in enumerate(smiles_list):
            props = self.calculate(smiles)
            results.append(props)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    # ==================== Absorption Predictions ====================

    def _predict_absorption(
        self,
        mw: float,
        logp: float,
        tpsa: float,
        hbd: int,
        hba: int,
    ) -> float:
        """
        Predict intestinal absorption (HIA).

        Based on Lipinski's rules and TPSA model.
        Returns probability of high absorption (0-1).
        """
        score = 1.0

        # MW penalty
        if mw > 500:
            score -= 0.15 * (mw - 500) / 200
        if mw > 700:
            score -= 0.2

        # LogP penalty (too low or too high)
        if logp < -1:
            score -= 0.1 * abs(logp + 1)
        if logp > 5:
            score -= 0.15 * (logp - 5)

        # TPSA penalty (high TPSA = low absorption)
        if tpsa > 140:
            score -= 0.3
        elif tpsa > 100:
            score -= 0.1

        # H-bond penalty
        if hbd > 5:
            score -= 0.1 * (hbd - 5)
        if hba > 10:
            score -= 0.1 * (hba - 10)

        return max(0.0, min(1.0, score))

    def _predict_caco2(self, logp: float, tpsa: float, mw: float) -> float:
        """
        Predict Caco-2 cell permeability.

        Returns probability of high permeability (0-1).
        """
        # Simplified model based on TPSA and LogP
        # High permeability: TPSA < 60, moderate LogP

        score = 0.5

        # TPSA contribution
        if tpsa < 60:
            score += 0.25
        elif tpsa < 90:
            score += 0.1
        elif tpsa > 120:
            score -= 0.2

        # LogP contribution
        if 1 <= logp <= 3:
            score += 0.2
        elif logp < 0:
            score -= 0.15
        elif logp > 4:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _predict_pgp_substrate(self, mol, mw: float, logp: float) -> float:
        """
        Predict P-glycoprotein substrate probability.

        P-gp substrates are actively effluxed, reducing absorption.
        """
        from rdkit.Chem import Descriptors

        # Larger, more polar molecules tend to be P-gp substrates
        score = 0.3  # Base probability

        if mw > 400:
            score += 0.1
        if mw > 600:
            score += 0.15

        tpsa = Descriptors.TPSA(mol)
        if tpsa > 100:
            score += 0.1

        # H-bond acceptors
        hba = Descriptors.NumHAcceptors(mol)
        if hba > 8:
            score += 0.1

        return max(0.0, min(1.0, score))

    # ==================== Distribution Predictions ====================

    def _predict_bbb(self, logp: float, tpsa: float, mw: float, hbd: int) -> float:
        """
        Predict blood-brain barrier permeability.

        Returns probability of CNS penetration (0-1).
        """
        score = 0.5

        # Optimal LogP for BBB (2-4)
        if 2 <= logp <= 4:
            score += 0.2
        elif logp < 1 or logp > 5:
            score -= 0.2

        # TPSA must be low for BBB (<90)
        if tpsa < 60:
            score += 0.25
        elif tpsa < 90:
            score += 0.1
        else:
            score -= 0.25

        # MW should be < 450 for BBB
        if mw < 400:
            score += 0.1
        elif mw > 500:
            score -= 0.2

        # Few H-bond donors
        if hbd <= 2:
            score += 0.1
        elif hbd > 4:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _predict_ppb(self, logp: float, mw: float) -> float:
        """
        Predict plasma protein binding.

        Returns fraction bound (0-1).
        """
        # Lipophilic compounds bind more to plasma proteins
        ppb = 0.5 + 0.08 * logp

        # Larger molecules bind more
        if mw > 500:
            ppb += 0.1

        return max(0.0, min(0.99, ppb))

    def _predict_vdss(self, logp: float, mw: float) -> float:
        """
        Predict volume of distribution at steady state (L/kg).

        Higher LogP = higher tissue distribution.
        """
        # Simplified empirical model
        vdss = 0.5 + 0.3 * logp

        if mw > 500:
            vdss *= 0.8  # Large molecules distribute less

        return max(0.1, min(10.0, vdss))

    # ==================== Metabolism Predictions ====================

    def _predict_cyp_substrate(self, mol, isoform: str) -> float:
        """
        Predict CYP450 substrate probability.

        Args:
            mol: RDKit molecule.
            isoform: CYP isoform (e.g., "2D6", "3A4").

        Returns:
            Probability of being a substrate (0-1).
        """
        from rdkit.Chem import Descriptors

        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        # Base probability
        prob = 0.4

        if isoform == "2D6":
            # CYP2D6 prefers basic amines
            if self._has_basic_nitrogen(mol):
                prob += 0.2
        elif isoform == "3A4":
            # CYP3A4 handles large lipophilic compounds
            if mw > 400 and logp > 3:
                prob += 0.2

        return max(0.0, min(1.0, prob))

    def _predict_cyp_inhibitor(self, mol, isoform: str) -> float:
        """
        Predict CYP450 inhibitor probability.

        Inhibitors can cause drug-drug interactions.
        """
        from rdkit.Chem import Descriptors

        logp = Descriptors.MolLogP(mol)

        # Lipophilic compounds more likely to inhibit
        prob = 0.2 + 0.05 * max(0, logp - 2)

        # Check for nitrogen heterocycles (common inhibitors)
        if self._has_nitrogen_heterocycle(mol):
            prob += 0.15

        return max(0.0, min(1.0, prob))

    def _has_basic_nitrogen(self, mol) -> bool:
        """Check if molecule has basic nitrogen."""
        from rdkit import Chem

        pattern = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]")
        return mol.HasSubstructMatch(pattern) if pattern else False

    def _has_nitrogen_heterocycle(self, mol) -> bool:
        """Check if molecule has nitrogen heterocycle."""
        from rdkit import Chem

        pattern = Chem.MolFromSmarts("[nR]")
        return mol.HasSubstructMatch(pattern) if pattern else False

    # ==================== Excretion Predictions ====================

    def _predict_half_life(self, mw: float, logp: float) -> float:
        """
        Predict elimination half-life (hours).

        Simplified model based on MW and LogP.
        """
        # Base half-life
        t_half = 4.0

        # Larger molecules clear slower
        t_half += (mw - 300) / 100

        # Lipophilic compounds accumulate
        t_half += logp * 2

        return max(0.5, min(100.0, t_half))

    def _predict_clearance(self, mw: float, logp: float) -> float:
        """
        Predict clearance (mL/min/kg).

        Simplified hepatic clearance model.
        """
        # Base clearance
        cl = 10.0

        # Lipophilic compounds metabolized faster
        cl += logp * 2

        # Very large molecules clear slower
        if mw > 500:
            cl *= 0.8

        return max(1.0, min(100.0, cl))

    # ==================== Toxicity Predictions ====================

    def _predict_herg(self, mol, logp: float, mw: float) -> float:
        """
        Predict hERG channel inhibition risk.

        hERG inhibition causes cardiac arrhythmias.
        Returns probability of inhibition (lower = safer).
        """
        prob = 0.2

        # Lipophilic compounds more likely to block hERG
        if logp > 3.5:
            prob += 0.15
        if logp > 5:
            prob += 0.2

        # Basic amines are common hERG blockers
        if self._has_basic_nitrogen(mol):
            prob += 0.15

        # MW in danger zone (250-500)
        if 250 < mw < 500:
            prob += 0.05

        return max(0.0, min(1.0, prob))

    def _predict_ames(self, mol) -> float:
        """
        Predict Ames mutagenicity.

        Returns probability of being mutagenic (lower = safer).
        """
        from rdkit import Chem

        prob = 0.15  # Base rate

        # Check for known mutagenic alerts
        mutagenic_patterns = [
            "[N+](=O)[O-]",  # Nitro group
            "[N;H1,H2;!$(NC=O)]c1ccccc1",  # Aromatic amine
            "N=N",  # Azo group
            "[CH2]Cl",  # Alkyl halide
        ]

        for smarts in mutagenic_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                prob += 0.2

        return max(0.0, min(1.0, prob))

    def _predict_hepatotoxicity(self, mol, mw: float, logp: float) -> float:
        """
        Predict hepatotoxicity risk.

        Returns probability of liver toxicity (lower = safer).
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        prob = 0.15

        # High daily dose increases risk
        # (We don't have dose here, use MW as proxy)
        if mw > 400:
            prob += 0.1

        # Reactive metabolite alerts
        reactive_patterns = [
            "[CH2;!R]Br",
            "[CH2;!R]I",
            "C(=O)Cl",  # Acyl halide
            "[N]=[N]=[N]",  # Azide
        ]

        for smarts in reactive_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                prob += 0.15

        # TPSA < 75 increases hepatotox risk
        tpsa = Descriptors.TPSA(mol)
        if tpsa < 75:
            prob += 0.1

        return max(0.0, min(1.0, prob))

    def _predict_skin_sensitization(self, mol) -> float:
        """
        Predict skin sensitization potential.

        Returns probability of being a sensitizer (lower = safer).
        """
        from rdkit import Chem

        prob = 0.1

        # Electrophilic centers that react with skin proteins
        sensitizer_patterns = [
            "[CH2]=C",  # Michael acceptor
            "[C;H1,H2]=[O]",  # Aldehyde
            "C(=O)Cl",  # Acyl chloride
            "[N;H1,H2]c1c(Cl)cccc1",  # Chloroaniline
        ]

        for smarts in sensitizer_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                prob += 0.2

        return max(0.0, min(1.0, prob))
