"""Tests for scoring module."""

import numpy as np
import pytest

from src.core.models import Molecule, DockingResult, DockingPose, ADMETProperties
from src.core.scoring import (
    CompositeScorer,
    CandidateRanker,
    ScoringConfig,
    ScoringWeights,
    DockingScoreNormalizer,
    NoveltyScorer,
)


class TestDockingScoreNormalizer:
    """Tests for DockingScoreNormalizer."""

    def test_normalize_best_energy(self):
        """Test normalization of best binding energy."""
        normalizer = DockingScoreNormalizer(best_energy=-12.0, worst_energy=-4.0)
        assert normalizer.normalize(-12.0) == 1.0
        assert normalizer.normalize(-13.0) == 1.0  # Better than best

    def test_normalize_worst_energy(self):
        """Test normalization of worst binding energy."""
        normalizer = DockingScoreNormalizer(best_energy=-12.0, worst_energy=-4.0)
        assert normalizer.normalize(-4.0) == 0.0
        assert normalizer.normalize(-3.0) == 0.0  # Worse than worst

    def test_normalize_mid_energy(self):
        """Test normalization of intermediate binding energy."""
        normalizer = DockingScoreNormalizer(best_energy=-12.0, worst_energy=-4.0)
        # -8 is midpoint: (-4 - (-8)) / (-4 - (-12)) = 4/8 = 0.5
        assert normalizer.normalize(-8.0) == pytest.approx(0.5)

    def test_normalize_batch(self):
        """Test batch normalization."""
        normalizer = DockingScoreNormalizer(best_energy=-12.0, worst_energy=-4.0)
        energies = np.array([-12.0, -8.0, -4.0, -2.0])
        scores = normalizer.normalize_batch(energies)

        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.5)
        assert scores[2] == pytest.approx(0.0)
        assert scores[3] == pytest.approx(0.0)  # Clipped


class TestNoveltyScorer:
    """Tests for NoveltyScorer."""

    def test_no_reference(self):
        """Test novelty with no reference compounds."""
        scorer = NoveltyScorer()
        fp = np.array([1, 0, 1, 0, 1])
        assert scorer.score(fp) == 0.5  # Neutral score

    def test_identical_compound(self):
        """Test novelty of compound identical to reference."""
        reference = np.array([[1, 0, 1, 0, 1]])
        scorer = NoveltyScorer(reference)

        # Same fingerprint = 0 novelty
        fp = np.array([1, 0, 1, 0, 1])
        assert scorer.score(fp) == 0.0

    def test_novel_compound(self):
        """Test novelty of completely different compound."""
        reference = np.array([[1, 1, 0, 0, 0]])
        scorer = NoveltyScorer(reference)

        # Completely different fingerprint
        fp = np.array([0, 0, 1, 1, 1])
        assert scorer.score(fp) == 1.0  # Maximum novelty

    def test_partial_novelty(self):
        """Test partial novelty."""
        reference = np.array([[1, 1, 1, 0, 0]])
        scorer = NoveltyScorer(reference)

        # 2 bits overlap out of 4 total bits
        fp = np.array([1, 1, 0, 1, 0])
        score = scorer.score(fp)
        assert 0.0 < score < 1.0


class TestScoringWeights:
    """Tests for ScoringWeights."""

    def test_default_weights_sum(self):
        """Test that default weights sum to 1."""
        weights = ScoringWeights()
        total = weights.docking + weights.ml_binding + weights.admet + weights.novelty
        assert total == pytest.approx(1.0)

    def test_weight_normalization(self):
        """Test that weights are normalized if they don't sum to 1."""
        weights = ScoringWeights(
            docking=0.5,
            ml_binding=0.5,
            admet=0.5,
            novelty=0.5,
        )
        total = weights.docking + weights.ml_binding + weights.admet + weights.novelty
        assert total == pytest.approx(1.0)


class TestCompositeScorer:
    """Tests for CompositeScorer."""

    def test_score_with_docking(self):
        """Test scoring with docking results."""
        scorer = CompositeScorer()
        mol = Molecule(id="mol1", smiles="CCO")

        docking = DockingResult(
            molecule_id="mol1",
            target_id="1M17",
            poses=[DockingPose(rank=1, energy=-10.0)],
        )

        candidate = scorer.score(
            molecule=mol,
            target_id="1M17",
            docking_result=docking,
        )

        assert candidate.docking_score > 0.5  # Good binding energy
        assert candidate.final_score > 0

    def test_score_with_all_components(self):
        """Test scoring with all components."""
        scorer = CompositeScorer()
        mol = Molecule(id="mol1", smiles="CCO")

        docking = DockingResult(
            molecule_id="mol1",
            target_id="1M17",
            poses=[DockingPose(rank=1, energy=-10.0)],
        )

        admet = ADMETProperties(
            qed_score=0.8,
            lipinski_violations=0,
        )

        candidate = scorer.score(
            molecule=mol,
            target_id="1M17",
            docking_result=docking,
            ml_binding_score=0.7,
            admet_properties=admet,
            fingerprint=np.array([1, 0, 1, 0, 1]),
        )

        assert candidate.docking_score > 0
        assert candidate.ml_binding_score == 0.7
        assert candidate.admet_score > 0
        assert candidate.novelty_score > 0
        assert candidate.final_score > 0


class TestCandidateRanker:
    """Tests for CandidateRanker."""

    def test_rank_by_score(self):
        """Test ranking by final score."""
        ranker = CandidateRanker()

        candidates = [
            self._make_candidate("mol1", 0.5),
            self._make_candidate("mol2", 0.8),
            self._make_candidate("mol3", 0.3),
            self._make_candidate("mol4", 0.9),
        ]

        ranked = ranker.rank(candidates, top_n=10, min_score=0.0)

        assert len(ranked) == 4
        assert ranked[0].molecule.id == "mol4"
        assert ranked[0].rank == 1
        assert ranked[1].molecule.id == "mol2"
        assert ranked[1].rank == 2

    def test_rank_with_min_score(self):
        """Test ranking with minimum score filter."""
        ranker = CandidateRanker()

        candidates = [
            self._make_candidate("mol1", 0.2),
            self._make_candidate("mol2", 0.5),
            self._make_candidate("mol3", 0.8),
        ]

        ranked = ranker.rank(candidates, min_score=0.4)

        assert len(ranked) == 2
        assert all(c.final_score >= 0.4 for c in ranked)

    def test_rank_top_n(self):
        """Test top-N selection."""
        ranker = CandidateRanker()

        candidates = [self._make_candidate(f"mol{i}", i / 10) for i in range(10)]

        ranked = ranker.rank(candidates, top_n=3, min_score=0.0)

        assert len(ranked) == 3
        assert ranked[0].final_score == 0.9
        assert ranked[2].final_score == 0.7

    def _make_candidate(self, mol_id: str, score: float):
        """Helper to create a scored candidate."""
        from src.core.models import ScoredCandidate
        mol = Molecule(id=mol_id, smiles="CCO")
        return ScoredCandidate(
            molecule=mol,
            target_id="1M17",
            final_score=score,
        )


class TestParetoRanking:
    """Tests for Pareto ranking."""

    def test_pareto_single_objective(self):
        """Test Pareto ranking with single objective."""
        ranker = CandidateRanker()

        candidates = [
            self._make_multi_candidate("mol1", 0.8, 0.5, 0.5),
            self._make_multi_candidate("mol2", 0.5, 0.5, 0.5),
        ]

        pareto = ranker.rank_pareto(
            candidates,
            objectives=["docking_score"],
        )

        # Both should be on Pareto front for single objective
        # (no one dominates the other)
        assert len(pareto) >= 1

    def test_pareto_multi_objective(self):
        """Test Pareto ranking with multiple objectives."""
        ranker = CandidateRanker()

        candidates = [
            self._make_multi_candidate("mol1", 0.9, 0.3, 0.3),  # Best docking
            self._make_multi_candidate("mol2", 0.3, 0.9, 0.3),  # Best ML
            self._make_multi_candidate("mol3", 0.3, 0.3, 0.9),  # Best ADMET
            self._make_multi_candidate("mol4", 0.5, 0.5, 0.5),  # Dominated
        ]

        pareto = ranker.rank_pareto(
            candidates,
            objectives=["docking_score", "ml_binding_score", "admet_score"],
        )

        # First three should be Pareto optimal (not dominated)
        pareto_ids = {c.molecule.id for c in pareto}
        assert "mol1" in pareto_ids
        assert "mol2" in pareto_ids
        assert "mol3" in pareto_ids

    def _make_multi_candidate(
        self,
        mol_id: str,
        docking: float,
        ml: float,
        admet: float,
    ):
        """Helper to create candidate with multiple scores."""
        from src.core.models import ScoredCandidate
        mol = Molecule(id=mol_id, smiles="CCO")
        return ScoredCandidate(
            molecule=mol,
            target_id="1M17",
            docking_score=docking,
            ml_binding_score=ml,
            admet_score=admet,
            final_score=(docking + ml + admet) / 3,
        )
