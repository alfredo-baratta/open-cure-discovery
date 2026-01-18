"""
Integration tests for Open Cure Discovery.

These tests verify that all components work together correctly.
"""

import tempfile
from pathlib import Path

import pytest
import numpy as np

# Skip all tests if RDKit is not available
pytest.importorskip("rdkit")


class TestMoleculePreparation:
    """Test molecule preparation pipeline."""

    def test_smiles_to_fingerprint(self):
        """Test SMILES to fingerprint conversion."""
        from src.core.ml.fingerprints import FingerprintGenerator

        gen = FingerprintGenerator()
        fp = gen.generate("CCO")  # Ethanol

        assert fp is not None
        assert len(fp) == 2048
        assert fp.dtype == np.float32
        assert fp.sum() > 0  # Has some bits set

    def test_batch_fingerprints(self):
        """Test batch fingerprint generation."""
        from src.core.ml.fingerprints import FingerprintGenerator

        smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
        gen = FingerprintGenerator()
        fps = gen.generate_batch(smiles_list)

        assert fps.shape == (3, 2048)
        assert np.all(fps.sum(axis=1) > 0)

    def test_molecular_descriptors(self):
        """Test molecular descriptor calculation."""
        from src.core.ml.fingerprints import MolecularDescriptorCalculator

        calc = MolecularDescriptorCalculator()
        desc = calc.calculate("CCO")  # Ethanol

        assert "mol_weight" in desc
        assert desc["mol_weight"] == pytest.approx(46.07, rel=0.01)
        assert "logp" in desc
        assert "qed" in desc


class TestADMETPrediction:
    """Test ADMET prediction pipeline."""

    def test_admet_calculation(self):
        """Test full ADMET property calculation."""
        from src.core.admet import ADMETCalculator

        calc = ADMETCalculator()
        props = calc.calculate("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

        assert props.qed_score is not None
        assert 0 <= props.qed_score <= 1
        assert props.lipinski_violations >= 0
        assert props.intestinal_absorption is not None
        assert props.herg_inhibition is not None

    def test_admet_batch(self):
        """Test batch ADMET calculation."""
        from src.core.admet import ADMETCalculator

        calc = ADMETCalculator()
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        results = calc.calculate_batch(smiles_list)

        assert len(results) == 3
        assert all(r.qed_score is not None for r in results)


class TestFilters:
    """Test molecular filtering."""

    def test_pains_filter_pass(self):
        """Test PAINS filter on clean compound."""
        from src.core.admet import PAINSFilter

        filt = PAINSFilter()
        result = filt.filter("CCO")  # Ethanol - should pass

        assert result.passed is True
        assert len(result.alerts) == 0

    def test_lipinski_filter(self):
        """Test Lipinski filter."""
        from src.core.admet import LipinskiFilter

        filt = LipinskiFilter(max_violations=1)

        # Small molecule - should pass
        result = filt.filter("CCO")
        assert result.passed is True
        assert result.details["num_violations"] == 0

    def test_combined_filter(self):
        """Test combined drug-likeness filter."""
        from src.core.admet import DrugLikenessFilter

        filt = DrugLikenessFilter()
        passed, results = filt.filter("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

        assert isinstance(passed, bool)
        assert len(results) == 3  # PAINS, Lipinski, Toxicophore


class TestMLPrediction:
    """Test ML binding prediction."""

    def test_binding_prediction(self):
        """Test binding affinity prediction."""
        from src.core.ml import BindingPredictor

        predictor = BindingPredictor(use_gpu=False)
        score = predictor.predict("CCO")

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_binding_batch(self):
        """Test batch binding prediction."""
        from src.core.ml import BindingPredictor

        predictor = BindingPredictor(use_gpu=False)
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        scores = predictor.predict(smiles_list)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)


class TestScoring:
    """Test scoring and ranking."""

    def test_composite_scoring(self):
        """Test composite scoring function."""
        from src.core.models import Molecule
        from src.core.scoring import CompositeScorer

        scorer = CompositeScorer()
        mol = Molecule(id="test", smiles="CCO")

        candidate = scorer.score(
            molecule=mol,
            target_id="target1",
            ml_binding_score=0.8,
        )

        assert candidate.final_score > 0
        assert candidate.ml_binding_score == 0.8

    def test_candidate_ranking(self):
        """Test candidate ranking."""
        from src.core.models import Molecule, ScoredCandidate
        from src.core.scoring import CandidateRanker

        ranker = CandidateRanker()

        candidates = [
            ScoredCandidate(
                molecule=Molecule(id=f"mol{i}", smiles="CCO"),
                target_id="target1",
                final_score=i / 10,
            )
            for i in range(10)
        ]

        ranked = ranker.rank(candidates, top_n=5, min_score=0.0)

        assert len(ranked) == 5
        assert ranked[0].final_score == 0.9
        assert ranked[0].rank == 1


class TestPipeline:
    """Test complete screening pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline instantiation."""
        from src.core.pipeline import ScreeningPipeline, PipelineConfig

        config = PipelineConfig(
            run_docking=False,
            use_gpu=False,
        )
        pipeline = ScreeningPipeline(config)

        assert pipeline is not None
        assert pipeline.config.run_docking is False

    def test_mini_screening(self):
        """Test mini screening run without docking."""
        from src.core.models import Molecule, ProteinTarget, BindingSite
        from src.core.pipeline import ScreeningPipeline, PipelineConfig

        # Create test molecules
        molecules = [
            Molecule(id="mol1", smiles="CCO", name="Ethanol"),
            Molecule(id="mol2", smiles="CC(=O)O", name="Acetic acid"),
            Molecule(id="mol3", smiles="c1ccccc1", name="Benzene"),
        ]

        # Create test target (using real COX-2 structure)
        target = ProteinTarget(
            id="test_target",
            name="Test Target",
            pdb_id="1CX2",  # COX-2 structure
            binding_sites=[BindingSite(center=(0, 0, 0))],
        )

        # Configure pipeline (no docking)
        config = PipelineConfig(
            run_docking=False,
            run_ml_prediction=True,
            run_admet=True,
            run_filters=True,
            use_gpu=False,
            batch_size=10,
            top_n=10,
            min_score=0.0,
        )

        # Run pipeline
        pipeline = ScreeningPipeline(config)
        results = pipeline.run(molecules, target, campaign_id="test")

        assert results.total_screened == 3
        assert len(results.candidates) > 0
        assert results.duration_seconds >= 0

    def test_pipeline_with_output(self):
        """Test pipeline with file output."""
        from src.core.models import Molecule, ProteinTarget, BindingSite
        from src.core.pipeline import ScreeningPipeline, PipelineConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            molecules = [
                Molecule(id="mol1", smiles="CCO", name="Ethanol"),
                Molecule(id="mol2", smiles="CC(=O)O", name="Acetic acid"),
            ]

            target = ProteinTarget(
                id="test_target",
                name="Test Target",
                pdb_id="1CX2",  # COX-2 structure
                binding_sites=[BindingSite(center=(0, 0, 0))],
            )

            config = PipelineConfig(
                run_docking=False,
                use_gpu=False,
                output_dir=output_dir,
                min_score=0.0,
            )

            pipeline = ScreeningPipeline(config)
            results = pipeline.run(molecules, target, campaign_id="output_test")

            # Check output files
            assert (output_dir / "output_test_summary.json").exists()
            assert (output_dir / "output_test_results.csv").exists()
            assert (output_dir / "output_test_top.smi").exists()


class TestDataLoaders:
    """Test data loader functionality."""

    def test_chembl_loader_init(self):
        """Test ChEMBL loader initialization."""
        from src.data.loaders import ChEMBLLoader

        loader = ChEMBLLoader()
        assert loader is not None

    def test_pdb_loader_init(self):
        """Test PDB loader initialization."""
        from src.data.loaders import PDBLoader

        loader = PDBLoader()
        assert loader is not None

    def test_zinc_loader_init(self):
        """Test ZINC loader initialization."""
        from src.data.loaders import ZINCLoader

        loader = ZINCLoader()
        assert loader is not None


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_workflow(self):
        """Test complete workflow from SMILES to ranked results."""
        from src.core.models import Molecule, ProteinTarget, BindingSite
        from src.core.ml import FingerprintGenerator, BindingPredictor
        from src.core.admet import ADMETCalculator, DrugLikenessFilter
        from src.core.scoring import CompositeScorer, CandidateRanker

        # Input
        smiles_list = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ]

        molecules = [
            Molecule(id=f"mol{i}", smiles=s)
            for i, s in enumerate(smiles_list)
        ]

        target = ProteinTarget(
            id="COX2",
            name="Cyclooxygenase-2",
            binding_sites=[BindingSite(center=(0, 0, 0))],
        )

        # Step 1: Filter
        drug_filter = DrugLikenessFilter()
        filtered = []
        for mol in molecules:
            passed, _ = drug_filter.filter(mol.smiles)
            if passed:
                filtered.append(mol)

        # Step 2: Fingerprints
        fp_gen = FingerprintGenerator()
        fingerprints = fp_gen.generate_batch([m.smiles for m in filtered])

        # Step 3: ML prediction
        predictor = BindingPredictor(use_gpu=False)
        ml_scores = predictor.predict([m.smiles for m in filtered])

        # Step 4: ADMET
        admet_calc = ADMETCalculator()
        admet_props = admet_calc.calculate_batch([m.smiles for m in filtered])

        # Step 5: Score
        scorer = CompositeScorer()
        candidates = scorer.score_batch(
            filtered,
            target.id,
            ml_binding_scores=ml_scores,
            admet_properties_list=admet_props,
            fingerprints=fingerprints,
        )

        # Step 6: Rank
        ranker = CandidateRanker()
        ranked = ranker.rank(candidates, top_n=10, min_score=0.0)

        # Verify
        assert len(ranked) > 0
        assert ranked[0].rank == 1
        assert ranked[0].final_score >= ranked[-1].final_score
        assert all(c.ml_binding_score is not None for c in ranked)
        assert all(c.admet_score is not None for c in ranked)
