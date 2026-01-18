"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest

from src.core.config import (
    CampaignConfig,
    DockingConfig,
    HardwareConfig,
    OutputConfig,
    ProjectConfig,
    ScoringConfig,
    TargetConfig,
    MoleculeLibraryConfig,
    get_preset,
)


class TestHardwareConfig:
    """Tests for HardwareConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HardwareConfig()
        assert config.gpu_memory_limit == 5000
        assert config.batch_size == "auto"
        assert config.num_cpu_workers == 4
        assert config.use_gpu is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HardwareConfig(
            gpu_memory_limit=8000,
            batch_size=256,
            num_cpu_workers=8,
            use_gpu=False,
        )
        assert config.gpu_memory_limit == 8000
        assert config.batch_size == 256

    def test_invalid_gpu_memory(self):
        """Test validation of GPU memory limits."""
        with pytest.raises(ValueError):
            HardwareConfig(gpu_memory_limit=500)  # Below minimum

        with pytest.raises(ValueError):
            HardwareConfig(gpu_memory_limit=100000)  # Above maximum


class TestDockingConfig:
    """Tests for DockingConfig."""

    def test_default_values(self):
        """Test default docking configuration."""
        config = DockingConfig()
        assert config.exhaustiveness == 8
        assert config.num_poses == 9
        assert config.energy_range == 3.0

    def test_exhaustiveness_validation(self):
        """Test exhaustiveness parameter validation."""
        config = DockingConfig(exhaustiveness=1)
        assert config.exhaustiveness == 1

        config = DockingConfig(exhaustiveness=32)
        assert config.exhaustiveness == 32

        with pytest.raises(ValueError):
            DockingConfig(exhaustiveness=0)

        with pytest.raises(ValueError):
            DockingConfig(exhaustiveness=64)


class TestScoringConfig:
    """Tests for ScoringConfig."""

    def test_default_weights(self):
        """Test default scoring weights."""
        config = ScoringConfig()
        total = (
            config.weight_docking
            + config.weight_ml_binding
            + config.weight_admet
            + config.weight_novelty
        )
        assert total == pytest.approx(1.0)

    def test_weight_validation(self):
        """Test weight parameter validation."""
        with pytest.raises(ValueError):
            ScoringConfig(weight_docking=-0.1)

        with pytest.raises(ValueError):
            ScoringConfig(weight_docking=1.5)


class TestProjectConfig:
    """Tests for ProjectConfig."""

    def test_required_fields(self):
        """Test that required fields are enforced."""
        config = ProjectConfig(name="test-project", disease="oncology")
        assert config.name == "test-project"
        assert config.disease == "oncology"

    def test_name_validation(self):
        """Test project name validation."""
        with pytest.raises(ValueError):
            ProjectConfig(name="", disease="oncology")


class TestCampaignConfig:
    """Tests for complete campaign configuration."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = CampaignConfig(
            project=ProjectConfig(name="test", disease="oncology"),
            target=TargetConfig(pdb_id="1M17"),
            molecules=MoleculeLibraryConfig(),
        )
        assert config.project.name == "test"
        assert config.target.pdb_id == "1M17"

    def test_yaml_roundtrip(self):
        """Test saving and loading from YAML."""
        config = CampaignConfig(
            project=ProjectConfig(name="test", disease="oncology"),
            target=TargetConfig(pdb_id="1M17"),
            molecules=MoleculeLibraryConfig(name="test-lib"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)

            loaded = CampaignConfig.from_yaml(path)
            assert loaded.project.name == config.project.name
            assert loaded.target.pdb_id == config.target.pdb_id


class TestPresets:
    """Tests for disease presets."""

    def test_lung_cancer_preset(self):
        """Test lung cancer preset exists and has expected fields."""
        preset = get_preset("lung-cancer")
        assert preset is not None
        assert "targets" in preset
        assert "EGFR" in preset["targets"]

    def test_nonexistent_preset(self):
        """Test that nonexistent preset returns None."""
        preset = get_preset("nonexistent-disease")
        assert preset is None

    def test_case_insensitive(self):
        """Test that preset lookup is case-insensitive."""
        preset1 = get_preset("lung-cancer")
        preset2 = get_preset("LUNG-CANCER")
        preset3 = get_preset("Lung-Cancer")

        assert preset1 == preset2 == preset3
