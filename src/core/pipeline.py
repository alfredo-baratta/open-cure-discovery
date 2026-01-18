"""
Virtual Screening Pipeline.

This module provides the main pipeline that orchestrates all
components for a complete drug discovery screening campaign.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional
import json
import csv

import numpy as np
from loguru import logger


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from src.core.models import (
    Molecule,
    ProteinTarget,
    DockingResult,
    ADMETProperties,
    ScoredCandidate,
)
from src.core.config import CampaignConfig, ScoringConfig as ConfigScoringConfig
from src.core.docking import (
    DockingEngine,
    DockingConfig,
    MoleculePreparator,
    ProteinPreparator,
    PreparationConfig,
)
from src.core.ml import (
    BindingPredictor,
    FingerprintGenerator,
    FingerprintConfig,
    FingerprintType,
)
from src.core.admet import (
    ADMETCalculator,
    DrugLikenessFilter,
)
from src.core.scoring import (
    CompositeScorer,
    CandidateRanker,
    ScoringConfig,
    ScoringWeights,
)


@dataclass
class PipelineConfig:
    """Configuration for the screening pipeline."""
    # Processing
    batch_size: int = 100
    use_gpu: bool = True
    num_workers: int = 4

    # Screening parameters
    run_docking: bool = True
    run_ml_prediction: bool = True
    run_admet: bool = True
    run_filters: bool = True

    # Scoring weights
    weight_docking: float = 0.35
    weight_ml_binding: float = 0.35
    weight_admet: float = 0.20
    weight_novelty: float = 0.10

    # Output
    top_n: int = 1000
    min_score: float = 0.3
    output_dir: Optional[Path] = None
    save_poses: bool = True

    # Checkpointing
    checkpoint_interval: int = 100
    resume_from_checkpoint: bool = False


@dataclass
class PipelineProgress:
    """Track pipeline progress."""
    total_molecules: int = 0
    processed_molecules: int = 0
    successful_docking: int = 0
    passed_filters: int = 0
    current_stage: str = "idle"
    start_time: Optional[datetime] = None
    last_checkpoint: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        if self.total_molecules == 0:
            return 0.0
        return (self.processed_molecules / self.total_molecules) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def molecules_per_second(self) -> float:
        """Get processing rate."""
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.processed_molecules / elapsed

    @property
    def eta_seconds(self) -> float:
        """Estimate remaining time in seconds."""
        rate = self.molecules_per_second
        if rate == 0:
            return 0.0
        remaining = self.total_molecules - self.processed_molecules
        return remaining / rate


@dataclass
class PipelineResults:
    """Results from a screening campaign."""
    campaign_id: str
    target_id: str
    total_screened: int = 0
    total_passed: int = 0
    candidates: list[ScoredCandidate] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    config: Optional[PipelineConfig] = None

    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "campaign_id": self.campaign_id,
            "target_id": self.target_id,
            "total_screened": self.total_screened,
            "total_passed": self.total_passed,
            "duration_seconds": self.duration_seconds,
            "top_candidates": [c.to_dict() for c in self.candidates[:10]],
        }


class ScreeningPipeline:
    """
    Main virtual screening pipeline.

    Orchestrates all components to run a complete drug discovery
    screening campaign from molecules to ranked candidates.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the screening pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or PipelineConfig()
        self.progress = PipelineProgress()

        # Initialize components
        self._init_components()

        # Callbacks
        self._progress_callback: Optional[Callable[[PipelineProgress], None]] = None

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        # Molecule preparation
        self.mol_preparator = MoleculePreparator(PreparationConfig())
        self.protein_preparator = ProteinPreparator()

        # Docking
        self.docking_engine = DockingEngine(prefer_gpu=self.config.use_gpu)
        self.docking_config = DockingConfig(
            exhaustiveness=8,
            num_poses=9,
            gpu_batch_size=self.config.batch_size,
        )

        # ML prediction
        self.binding_predictor = BindingPredictor(use_gpu=self.config.use_gpu)
        self.fingerprint_gen = FingerprintGenerator(FingerprintConfig(
            fp_type=FingerprintType.MORGAN,
            n_bits=2048,
            radius=2,
        ))

        # ADMET
        self.admet_calculator = ADMETCalculator()
        self.drug_filter = DrugLikenessFilter(
            apply_pains=True,
            apply_lipinski=True,
            apply_toxicophore=True,
        )

        # Scoring
        self.scorer = CompositeScorer(ScoringConfig(
            weights=ScoringWeights(
                docking=self.config.weight_docking,
                ml_binding=self.config.weight_ml_binding,
                admet=self.config.weight_admet,
                novelty=self.config.weight_novelty,
            ),
        ))
        self.ranker = CandidateRanker()

    def set_progress_callback(
        self,
        callback: Callable[[PipelineProgress], None],
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _update_progress(self, stage: str, increment: int = 0) -> None:
        """Update progress and notify callback."""
        self.progress.current_stage = stage
        self.progress.processed_molecules += increment

        if self._progress_callback:
            self._progress_callback(self.progress)

    def run(
        self,
        molecules: list[Molecule],
        target: ProteinTarget,
        campaign_id: Optional[str] = None,
    ) -> PipelineResults:
        """
        Run the complete screening pipeline.

        Args:
            molecules: List of molecules to screen.
            target: Protein target for docking.
            campaign_id: Optional campaign identifier.

        Returns:
            PipelineResults with ranked candidates.
        """
        campaign_id = campaign_id or f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize results
        results = PipelineResults(
            campaign_id=campaign_id,
            target_id=target.id,
            start_time=datetime.now(),
            config=self.config,
        )

        # Initialize progress
        self.progress = PipelineProgress(
            total_molecules=len(molecules),
            start_time=datetime.now(),
        )

        logger.info(f"Starting screening campaign: {campaign_id}")
        logger.info(f"Target: {target.name} ({target.id})")
        logger.info(f"Molecules to screen: {len(molecules)}")

        try:
            # Stage 1: Prepare target
            self._update_progress("preparing_target")
            prepared_target = self._prepare_target(target)

            # Stage 2: Process molecules in batches
            all_candidates = []
            batch_size = self.config.batch_size

            for i in range(0, len(molecules), batch_size):
                batch = molecules[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(molecules) + batch_size - 1) // batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches}")
                self._update_progress(f"batch_{batch_num}")

                # Process batch
                batch_candidates = self._process_batch(
                    batch,
                    prepared_target,
                )
                all_candidates.extend(batch_candidates)

                # Update progress
                self.progress.processed_molecules = min(i + batch_size, len(molecules))
                self._update_progress(f"batch_{batch_num}_complete")

                # Checkpoint
                if (i + batch_size) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(all_candidates, campaign_id)

            # Stage 3: Final ranking
            self._update_progress("ranking")
            ranked_candidates = self.ranker.rank(
                all_candidates,
                top_n=self.config.top_n,
                min_score=self.config.min_score,
            )

            # Finalize results
            results.total_screened = len(molecules)
            results.total_passed = len(ranked_candidates)
            results.candidates = ranked_candidates
            results.end_time = datetime.now()

            logger.info(f"Screening complete!")
            logger.info(f"Screened: {results.total_screened}")
            logger.info(f"Passed: {results.total_passed}")
            logger.info(f"Duration: {results.duration_seconds:.1f}s")

            # Save results
            if self.config.output_dir:
                self._save_results(results)

            self._update_progress("complete")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self._update_progress("error")
            raise

        return results

    def _prepare_target(self, target: ProteinTarget) -> ProteinTarget:
        """Prepare target for docking."""
        if target.is_prepared:
            logger.info("Target already prepared")
            return target

        logger.info("Preparing target structure...")

        if target.pdb_id:
            # Download and prepare from PDB
            prepared = self.protein_preparator.prepare_from_pdb_id(
                target.pdb_id,
                binding_site=target.primary_site,
            )
            # Copy over any existing binding sites
            if target.binding_sites and not prepared.binding_sites:
                prepared.binding_sites = target.binding_sites
            return prepared

        elif target.structure_path:
            # Prepare from local file
            return self.protein_preparator.prepare_from_file(
                target.structure_path,
                target.id,
                binding_site=target.primary_site,
            )

        else:
            raise ValueError("Target must have either pdb_id or structure_path")

    def _process_batch(
        self,
        molecules: list[Molecule],
        target: ProteinTarget,
    ) -> list[ScoredCandidate]:
        """Process a batch of molecules through the pipeline."""
        candidates = []

        # Step 1: Pre-filter with drug-likeness
        if self.config.run_filters:
            filtered_molecules = []
            filter_results = []

            for mol in molecules:
                passed, results = self.drug_filter.filter(mol.smiles)
                if passed:
                    filtered_molecules.append(mol)
                    filter_results.append(results)
                else:
                    self.progress.processed_molecules += 1

            molecules = filtered_molecules
            self.progress.passed_filters += len(filtered_molecules)

        if not molecules:
            return candidates

        # Step 2: Generate fingerprints
        smiles_list = [m.smiles for m in molecules]
        fingerprints = self.fingerprint_gen.generate_batch(smiles_list)

        # Step 3: ML binding prediction
        ml_scores = None
        if self.config.run_ml_prediction:
            ml_scores = self.binding_predictor.predict(smiles_list, target.id)

        # Step 4: ADMET calculation
        admet_list = None
        if self.config.run_admet:
            admet_list = self.admet_calculator.calculate_batch(smiles_list)

        # Step 5: Prepare molecules for docking
        docking_results = None
        if self.config.run_docking and self.docking_engine.is_available:
            # Prepare molecules
            prepared_molecules = []
            for mol in molecules:
                try:
                    prepared = self.mol_preparator.prepare(mol)
                    prepared_molecules.append(prepared)
                except Exception as e:
                    logger.warning(f"Failed to prepare {mol.id}: {e}")
                    prepared_molecules.append(mol)  # Keep unprepared

            # Run docking
            docking_results = list(self.docking_engine.dock_batch(
                prepared_molecules,
                target,
                self.docking_config,
            ))

            self.progress.successful_docking += sum(
                1 for r in docking_results if r.success
            )

        # Step 6: Score candidates
        for i, mol in enumerate(molecules):
            docking = docking_results[i] if docking_results else None
            ml_score = ml_scores[i] if ml_scores is not None else None
            admet = admet_list[i] if admet_list else None
            fp = fingerprints[i] if fingerprints is not None else None

            candidate = self.scorer.score(
                molecule=mol,
                target_id=target.id,
                docking_result=docking,
                ml_binding_score=ml_score,
                admet_properties=admet,
                fingerprint=fp,
            )

            candidates.append(candidate)

        return candidates

    def _save_checkpoint(
        self,
        candidates: list[ScoredCandidate],
        campaign_id: str,
    ) -> None:
        """Save checkpoint for resume capability."""
        if not self.config.output_dir:
            return

        checkpoint_dir = self.config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"{campaign_id}_checkpoint.json"

        checkpoint_data = {
            "campaign_id": campaign_id,
            "processed": self.progress.processed_molecules,
            "total": self.progress.total_molecules,
            "timestamp": datetime.now().isoformat(),
            "candidates_count": len(candidates),
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, cls=NumpyEncoder)

        self.progress.last_checkpoint = datetime.now()
        logger.debug(f"Checkpoint saved: {checkpoint_file}")

    def _save_results(self, results: PipelineResults) -> None:
        """Save results to output directory."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary JSON
        summary_file = output_dir / f"{results.campaign_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2, cls=NumpyEncoder)
        logger.info(f"Summary saved: {summary_file}")

        # Save full results CSV
        csv_file = output_dir / f"{results.campaign_id}_results.csv"
        self._save_csv(results.candidates, csv_file)
        logger.info(f"Results saved: {csv_file}")

        # Save top candidates SMILES
        smiles_file = output_dir / f"{results.campaign_id}_top.smi"
        with open(smiles_file, "w") as f:
            for c in results.candidates[:self.config.top_n]:
                f.write(f"{c.molecule.smiles}\t{c.molecule.id}\t{c.final_score:.4f}\n")
        logger.info(f"Top SMILES saved: {smiles_file}")

    def _save_csv(
        self,
        candidates: list[ScoredCandidate],
        path: Path,
    ) -> None:
        """Save candidates to CSV file."""
        if not candidates:
            return

        fieldnames = [
            "rank", "molecule_id", "smiles", "name",
            "final_score", "docking_score", "ml_binding_score",
            "admet_score", "novelty_score", "binding_energy",
            "passes_filters",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for c in candidates:
                row = {
                    "rank": c.rank,
                    "molecule_id": c.molecule.id,
                    "smiles": c.molecule.smiles,
                    "name": c.molecule.name or "",
                    "final_score": f"{c.final_score:.4f}",
                    "docking_score": f"{c.docking_score:.4f}",
                    "ml_binding_score": f"{c.ml_binding_score:.4f}",
                    "admet_score": f"{c.admet_score:.4f}",
                    "novelty_score": f"{c.novelty_score:.4f}",
                    "binding_energy": f"{c.best_binding_energy:.2f}" if c.best_binding_energy else "",
                    "passes_filters": c.passes_filters,
                }
                writer.writerow(row)


def run_screening(
    molecules: list[Molecule],
    target: ProteinTarget,
    config: Optional[PipelineConfig] = None,
    output_dir: Optional[Path] = None,
) -> PipelineResults:
    """
    Convenience function to run a screening campaign.

    Args:
        molecules: Molecules to screen.
        target: Protein target.
        config: Pipeline configuration.
        output_dir: Where to save results.

    Returns:
        PipelineResults with ranked candidates.
    """
    if config is None:
        config = PipelineConfig()

    if output_dir:
        config.output_dir = output_dir

    pipeline = ScreeningPipeline(config)
    return pipeline.run(molecules, target)
