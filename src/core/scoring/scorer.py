"""
Composite scoring system for drug candidates.

This module provides the main scoring engine that combines
docking, ML, and ADMET scores into a final ranking.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from loguru import logger

from src.core.models import (
    DockingResult,
    Molecule,
    ScoredCandidate,
    ADMETProperties,
)


@dataclass
class ScoringWeights:
    """Weights for composite scoring."""
    docking: float = 0.35
    ml_binding: float = 0.35
    admet: float = 0.20
    novelty: float = 0.10

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.docking + self.ml_binding + self.admet + self.novelty
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            self.docking /= total
            self.ml_binding /= total
            self.admet /= total
            self.novelty /= total


@dataclass
class ScoringConfig:
    """Configuration for scoring engine."""
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # Docking score normalization
    docking_best: float = -12.0  # Best expected binding energy (kcal/mol)
    docking_worst: float = -4.0  # Worst acceptable binding energy

    # Thresholds
    min_score: float = 0.3  # Minimum final score to keep
    top_n: int = 1000  # Top N candidates to return


class DockingScoreNormalizer:
    """
    Normalize docking scores to 0-1 scale.

    Binding energies are typically negative, with more negative
    values indicating stronger binding.
    """

    def __init__(
        self,
        best_energy: float = -12.0,
        worst_energy: float = -4.0,
    ):
        """
        Initialize normalizer.

        Args:
            best_energy: Best expected binding energy (most negative).
            worst_energy: Worst acceptable binding energy.
        """
        self.best = best_energy
        self.worst = worst_energy

    def normalize(self, energy: float) -> float:
        """
        Normalize binding energy to 0-1 score.

        Args:
            energy: Binding energy in kcal/mol.

        Returns:
            Normalized score (0-1, higher = better).
        """
        if energy <= self.best:
            return 1.0
        if energy >= self.worst:
            return 0.0

        # Linear interpolation
        return (self.worst - energy) / (self.worst - self.best)

    def normalize_batch(self, energies: np.ndarray) -> np.ndarray:
        """Normalize a batch of binding energies."""
        scores = (self.worst - energies) / (self.worst - self.best)
        return np.clip(scores, 0.0, 1.0)


class NoveltyScorer:
    """
    Score molecules based on novelty (dissimilarity to known drugs).

    More novel compounds get higher scores, encouraging discovery
    of new chemical scaffolds.
    """

    def __init__(self, reference_fingerprints: Optional[np.ndarray] = None):
        """
        Initialize novelty scorer.

        Args:
            reference_fingerprints: Fingerprints of known drugs/compounds.
        """
        self.reference_fps = reference_fingerprints

    def score(self, fingerprint: np.ndarray) -> float:
        """
        Calculate novelty score for a molecule.

        Args:
            fingerprint: Molecular fingerprint.

        Returns:
            Novelty score (0-1, higher = more novel).
        """
        if self.reference_fps is None or len(self.reference_fps) == 0:
            return 0.5  # No reference, neutral score

        # Calculate max similarity to any reference compound
        max_similarity = 0.0

        for ref_fp in self.reference_fps:
            # Tanimoto similarity
            intersection = np.sum(fingerprint * ref_fp)
            union = np.sum(fingerprint) + np.sum(ref_fp) - intersection
            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of similarity
        return 1.0 - max_similarity

    def score_batch(self, fingerprints: np.ndarray) -> np.ndarray:
        """Score novelty for a batch of molecules."""
        return np.array([self.score(fp) for fp in fingerprints])


class CompositeScorer:
    """
    Main scoring engine that combines all scores.

    Produces a final weighted score for ranking drug candidates.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        """Initialize composite scorer."""
        self.config = config or ScoringConfig()
        self.docking_normalizer = DockingScoreNormalizer(
            best_energy=self.config.docking_best,
            worst_energy=self.config.docking_worst,
        )
        self.novelty_scorer = NoveltyScorer()

    def set_reference_compounds(self, fingerprints: np.ndarray) -> None:
        """Set reference compounds for novelty scoring."""
        self.novelty_scorer = NoveltyScorer(fingerprints)

    def score(
        self,
        molecule: Molecule,
        target_id: str,
        docking_result: Optional[DockingResult] = None,
        ml_binding_score: Optional[float] = None,
        admet_properties: Optional[ADMETProperties] = None,
        fingerprint: Optional[np.ndarray] = None,
    ) -> ScoredCandidate:
        """
        Calculate composite score for a single molecule.

        Args:
            molecule: The molecule to score.
            target_id: Target identifier.
            docking_result: Docking results (optional).
            ml_binding_score: ML binding prediction (optional).
            admet_properties: ADMET properties (optional).
            fingerprint: Molecular fingerprint for novelty (optional).

        Returns:
            ScoredCandidate with all scores and final ranking score.
        """
        weights = self.config.weights

        # Initialize scores
        docking_score = 0.0
        binding_score = ml_binding_score or 0.5
        admet_score = 0.5
        novelty_score = 0.5

        # Calculate docking score
        if docking_result and docking_result.best_energy is not None:
            docking_score = self.docking_normalizer.normalize(
                docking_result.best_energy
            )
        elif docking_result and not docking_result.success:
            docking_score = 0.0

        # Calculate ADMET score
        if admet_properties:
            admet_score = admet_properties.admet_score

        # Calculate novelty score
        if fingerprint is not None:
            novelty_score = self.novelty_scorer.score(fingerprint)

        # Calculate weighted final score
        final_score = (
            weights.docking * docking_score
            + weights.ml_binding * binding_score
            + weights.admet * admet_score
            + weights.novelty * novelty_score
        )

        return ScoredCandidate(
            molecule=molecule,
            target_id=target_id,
            docking_score=docking_score,
            ml_binding_score=binding_score,
            admet_score=admet_score,
            novelty_score=novelty_score,
            final_score=final_score,
            docking_result=docking_result,
            admet_properties=admet_properties,
        )

    def score_batch(
        self,
        molecules: list[Molecule],
        target_id: str,
        docking_results: Optional[list[DockingResult]] = None,
        ml_binding_scores: Optional[np.ndarray] = None,
        admet_properties_list: Optional[list[ADMETProperties]] = None,
        fingerprints: Optional[np.ndarray] = None,
    ) -> list[ScoredCandidate]:
        """
        Score a batch of molecules.

        Args:
            molecules: List of molecules.
            target_id: Target identifier.
            docking_results: List of docking results.
            ml_binding_scores: Array of ML binding scores.
            admet_properties_list: List of ADMET properties.
            fingerprints: Array of molecular fingerprints.

        Returns:
            List of ScoredCandidate objects.
        """
        n = len(molecules)
        results = []

        for i in range(n):
            docking = docking_results[i] if docking_results else None
            ml_score = ml_binding_scores[i] if ml_binding_scores is not None else None
            admet = admet_properties_list[i] if admet_properties_list else None
            fp = fingerprints[i] if fingerprints is not None else None

            candidate = self.score(
                molecule=molecules[i],
                target_id=target_id,
                docking_result=docking,
                ml_binding_score=ml_score,
                admet_properties=admet,
                fingerprint=fp,
            )
            results.append(candidate)

        return results


class CandidateRanker:
    """
    Rank and filter scored candidates.

    Provides various ranking strategies and filtering options.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        """Initialize ranker."""
        self.config = config or ScoringConfig()

    def rank(
        self,
        candidates: list[ScoredCandidate],
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[ScoredCandidate]:
        """
        Rank candidates by final score.

        Args:
            candidates: List of scored candidates.
            top_n: Return only top N candidates.
            min_score: Minimum score threshold.

        Returns:
            Sorted list of candidates with rank assigned.
        """
        top_n = top_n or self.config.top_n
        min_score = min_score or self.config.min_score

        # Filter by minimum score
        filtered = [c for c in candidates if c.final_score >= min_score]

        # Sort by final score (descending)
        sorted_candidates = sorted(
            filtered,
            key=lambda c: c.final_score,
            reverse=True,
        )

        # Take top N
        top_candidates = sorted_candidates[:top_n]

        # Assign ranks
        for i, candidate in enumerate(top_candidates, 1):
            candidate.rank = i

        return top_candidates

    def rank_pareto(
        self,
        candidates: list[ScoredCandidate],
        objectives: list[str] = None,
        top_n: Optional[int] = None,
    ) -> list[ScoredCandidate]:
        """
        Rank using Pareto optimization for multiple objectives.

        Identifies non-dominated solutions that represent trade-offs
        between different scoring objectives.

        Args:
            candidates: List of scored candidates.
            objectives: Score attributes to optimize (default: all).
            top_n: Maximum candidates to return.

        Returns:
            Pareto-optimal candidates.
        """
        if objectives is None:
            objectives = ["docking_score", "ml_binding_score", "admet_score"]

        top_n = top_n or self.config.top_n

        # Extract objective values
        n = len(candidates)
        scores = np.zeros((n, len(objectives)))

        for i, candidate in enumerate(candidates):
            for j, obj in enumerate(objectives):
                scores[i, j] = getattr(candidate, obj, 0.0)

        # Find Pareto front
        pareto_mask = self._pareto_front(scores)
        pareto_candidates = [c for c, is_pareto in zip(candidates, pareto_mask) if is_pareto]

        # Sort by final score within Pareto front
        pareto_candidates.sort(key=lambda c: c.final_score, reverse=True)

        # Assign ranks
        for i, candidate in enumerate(pareto_candidates[:top_n], 1):
            candidate.rank = i

        return pareto_candidates[:top_n]

    def _pareto_front(self, scores: np.ndarray) -> np.ndarray:
        """
        Find Pareto-optimal points.

        Args:
            scores: (n_candidates, n_objectives) array, higher is better.

        Returns:
            Boolean mask of Pareto-optimal candidates.
        """
        n = scores.shape[0]
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_pareto[i]:
                continue

            # Check if any other point dominates this one
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue

                # j dominates i if j is >= in all objectives and > in at least one
                if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                    is_pareto[i] = False
                    break

        return is_pareto

    def select_diverse(
        self,
        candidates: list[ScoredCandidate],
        fingerprints: np.ndarray,
        n_select: int = 100,
        similarity_threshold: float = 0.7,
    ) -> list[ScoredCandidate]:
        """
        Select diverse candidates using MaxMin algorithm.

        Ensures chemical diversity in the final selection.

        Args:
            candidates: Ranked candidates.
            fingerprints: Molecular fingerprints.
            n_select: Number of candidates to select.
            similarity_threshold: Maximum similarity between selected compounds.

        Returns:
            Diverse subset of candidates.
        """
        if len(candidates) <= n_select:
            return candidates

        # Start with highest scoring candidate
        selected_indices = [0]
        selected_fps = [fingerprints[0]]

        while len(selected_indices) < n_select:
            best_idx = -1
            best_min_dist = -1

            for i in range(len(candidates)):
                if i in selected_indices:
                    continue

                # Calculate minimum distance to selected set
                min_dist = float("inf")
                for sel_fp in selected_fps:
                    # Tanimoto distance = 1 - Tanimoto similarity
                    intersection = np.sum(fingerprints[i] * sel_fp)
                    union = np.sum(fingerprints[i]) + np.sum(sel_fp) - intersection
                    similarity = intersection / union if union > 0 else 0
                    distance = 1 - similarity
                    min_dist = min(min_dist, distance)

                # Keep track of best candidate (max min-distance)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            if best_idx == -1 or best_min_dist < (1 - similarity_threshold):
                break

            selected_indices.append(best_idx)
            selected_fps.append(fingerprints[best_idx])

        # Return selected candidates
        selected = [candidates[i] for i in selected_indices]

        # Re-assign ranks
        for i, candidate in enumerate(selected, 1):
            candidate.rank = i

        return selected
