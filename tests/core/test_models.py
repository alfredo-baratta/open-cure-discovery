"""Tests for core data models."""

import pytest

from src.core.models import (
    Molecule,
    MoleculeState,
    ProteinTarget,
    BindingSite,
    DockingPose,
    DockingResult,
    ADMETProperties,
    ScoredCandidate,
)


class TestMolecule:
    """Tests for Molecule class."""

    def test_molecule_creation(self):
        """Test basic molecule creation."""
        mol = Molecule(
            id="mol1",
            smiles="CCO",
            name="Ethanol",
        )
        assert mol.id == "mol1"
        assert mol.smiles == "CCO"
        assert mol.name == "Ethanol"
        assert mol.state == MoleculeState.RAW

    def test_molecule_auto_id(self):
        """Test automatic ID generation from SMILES."""
        mol = Molecule(id="", smiles="CCO")
        assert mol.id != ""
        assert len(mol.id) == 12  # MD5 hash truncated

    def test_molecule_is_prepared(self):
        """Test is_prepared property."""
        mol = Molecule(id="mol1", smiles="CCO")
        assert not mol.is_prepared

        mol.pdbqt_content = "ATOM..."
        assert mol.is_prepared


class TestBindingSite:
    """Tests for BindingSite class."""

    def test_binding_site_creation(self):
        """Test binding site creation."""
        site = BindingSite(
            center=(10.0, 20.0, 30.0),
            size=(22.0, 22.0, 22.0),
            name="active_site",
        )
        assert site.center == (10.0, 20.0, 30.0)
        assert site.size == (22.0, 22.0, 22.0)
        assert site.name == "active_site"

    def test_to_autodock_config(self):
        """Test AutoDock config conversion."""
        site = BindingSite(
            center=(10.0, 20.0, 30.0),
            size=(22.0, 24.0, 26.0),
        )
        config = site.to_autodock_config()

        assert config["center_x"] == 10.0
        assert config["center_y"] == 20.0
        assert config["center_z"] == 30.0
        assert config["size_x"] == 22.0
        assert config["size_y"] == 24.0
        assert config["size_z"] == 26.0


class TestProteinTarget:
    """Tests for ProteinTarget class."""

    def test_target_creation(self):
        """Test target creation."""
        target = ProteinTarget(
            id="1M17",
            name="EGFR",
            pdb_id="1M17",
        )
        assert target.id == "1M17"
        assert target.name == "EGFR"
        assert target.organism == "Homo sapiens"

    def test_target_with_binding_site(self):
        """Test target with binding site."""
        site = BindingSite(center=(0, 0, 0))
        target = ProteinTarget(
            id="1M17",
            name="EGFR",
            binding_sites=[site],
        )
        assert len(target.binding_sites) == 1
        assert target.primary_site == site

    def test_target_is_prepared(self):
        """Test is_prepared property."""
        target = ProteinTarget(id="1M17", name="EGFR")
        assert not target.is_prepared

        target.pdbqt_content = "ATOM..."
        assert target.is_prepared


class TestDockingResult:
    """Tests for DockingResult class."""

    def test_docking_result_creation(self):
        """Test docking result creation."""
        result = DockingResult(
            molecule_id="mol1",
            target_id="1M17",
            poses=[
                DockingPose(rank=1, energy=-8.5),
                DockingPose(rank=2, energy=-7.2),
            ],
        )
        assert result.molecule_id == "mol1"
        assert result.num_poses == 2
        assert result.success is True

    def test_best_energy(self):
        """Test best_energy property."""
        result = DockingResult(
            molecule_id="mol1",
            target_id="1M17",
            poses=[
                DockingPose(rank=1, energy=-7.2),
                DockingPose(rank=2, energy=-8.5),
                DockingPose(rank=3, energy=-6.1),
            ],
        )
        assert result.best_energy == -8.5

    def test_best_pose(self):
        """Test best_pose property."""
        result = DockingResult(
            molecule_id="mol1",
            target_id="1M17",
            poses=[
                DockingPose(rank=1, energy=-7.2),
                DockingPose(rank=2, energy=-8.5),
            ],
        )
        assert result.best_pose.energy == -8.5

    def test_failed_docking(self):
        """Test failed docking result."""
        result = DockingResult(
            molecule_id="mol1",
            target_id="1M17",
            success=False,
            error_message="Docking failed",
        )
        assert not result.success
        assert result.best_energy is None


class TestADMETProperties:
    """Tests for ADMETProperties class."""

    def test_admet_creation(self):
        """Test ADMET properties creation."""
        props = ADMETProperties(
            qed_score=0.8,
            lipinski_violations=0,
            herg_inhibition=0.1,
        )
        assert props.qed_score == 0.8
        assert props.lipinski_violations == 0

    def test_toxicity_score(self):
        """Test toxicity score calculation."""
        props = ADMETProperties(
            herg_inhibition=0.2,
            ames_mutagenicity=0.1,
            hepatotoxicity=0.3,
        )
        # Average of toxicity scores
        expected = (0.2 + 0.1 + 0.3) / 3
        assert props.toxicity_score == pytest.approx(expected)

    def test_admet_score(self):
        """Test composite ADMET score."""
        props = ADMETProperties(
            qed_score=0.8,
            lipinski_violations=1,
            herg_inhibition=0.1,
            ames_mutagenicity=0.1,
            hepatotoxicity=0.1,
        )
        # Should be positive but less than QED due to penalties
        assert 0 < props.admet_score < 0.8


class TestScoredCandidate:
    """Tests for ScoredCandidate class."""

    def test_scored_candidate_creation(self):
        """Test scored candidate creation."""
        mol = Molecule(id="mol1", smiles="CCO")
        candidate = ScoredCandidate(
            molecule=mol,
            target_id="1M17",
            docking_score=0.8,
            ml_binding_score=0.7,
            admet_score=0.6,
            novelty_score=0.5,
            final_score=0.65,
        )
        assert candidate.molecule.id == "mol1"
        assert candidate.final_score == 0.65

    def test_to_dict(self):
        """Test conversion to dictionary."""
        mol = Molecule(id="mol1", smiles="CCO", name="Ethanol")
        candidate = ScoredCandidate(
            molecule=mol,
            target_id="1M17",
            docking_score=0.8,
            final_score=0.7,
            rank=1,
        )
        d = candidate.to_dict()

        assert d["molecule_id"] == "mol1"
        assert d["smiles"] == "CCO"
        assert d["name"] == "Ethanol"
        assert d["docking_score"] == 0.8
        assert d["rank"] == 1
