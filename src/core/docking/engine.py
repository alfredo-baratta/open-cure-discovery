"""
Molecular docking engine with GPU acceleration.

This module provides the main docking engine that orchestrates
molecular docking using AutoDock-GPU or fallback to Vina.
"""

import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import shutil

from loguru import logger

from src.core.models import (
    DockingPose,
    DockingResult,
    Molecule,
    ProteinTarget,
)


@dataclass
class DockingConfig:
    """Configuration for docking engine."""
    exhaustiveness: int = 8
    num_poses: int = 9
    energy_range: float = 3.0
    cpu_threads: int = 4
    gpu_batch_size: int = 128
    timeout: int = 300  # seconds per molecule


class DockingEngineBase(ABC):
    """Abstract base class for docking engines."""

    @abstractmethod
    def dock(
        self,
        molecule: Molecule,
        target: ProteinTarget,
        config: Optional[DockingConfig] = None,
    ) -> DockingResult:
        """Dock a single molecule to a target."""
        pass

    @abstractmethod
    def dock_batch(
        self,
        molecules: list[Molecule],
        target: ProteinTarget,
        config: Optional[DockingConfig] = None,
    ) -> Iterator[DockingResult]:
        """Dock a batch of molecules to a target."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this docking engine is available."""
        pass


class AutoDockGPUEngine(DockingEngineBase):
    """
    AutoDock-GPU docking engine.

    This engine provides GPU-accelerated molecular docking using
    the AutoDock-GPU implementation.
    """

    def __init__(self, executable_path: Optional[Path] = None):
        """
        Initialize AutoDock-GPU engine.

        Args:
            executable_path: Path to autodock_gpu executable.
                           If None, searches PATH.
        """
        self.executable = executable_path or self._find_executable()
        self._temp_dir: Optional[Path] = None

    def _find_executable(self) -> Optional[Path]:
        """Find AutoDock-GPU executable in PATH."""
        # Try common names
        for name in ["autodock_gpu", "autodock_gpu_128wi", "AutoDock-GPU"]:
            path = shutil.which(name)
            if path:
                return Path(path)
        return None

    def is_available(self) -> bool:
        """Check if AutoDock-GPU is available."""
        if self.executable is None:
            return False
        try:
            result = subprocess.run(
                [str(self.executable), "--version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def dock(
        self,
        molecule: Molecule,
        target: ProteinTarget,
        config: Optional[DockingConfig] = None,
    ) -> DockingResult:
        """
        Dock a single molecule to a target.

        Args:
            molecule: Prepared molecule with PDBQT content.
            target: Prepared protein target.
            config: Docking configuration.

        Returns:
            DockingResult with poses and energies.
        """
        config = config or DockingConfig()
        start_time = time.time()

        # Validate inputs
        if not molecule.is_prepared:
            return DockingResult(
                molecule_id=molecule.id,
                target_id=target.id,
                success=False,
                error_message="Molecule not prepared (no PDBQT content)",
            )

        if not target.is_prepared:
            return DockingResult(
                molecule_id=molecule.id,
                target_id=target.id,
                success=False,
                error_message="Target not prepared (no PDBQT content)",
            )

        if not target.primary_site:
            return DockingResult(
                molecule_id=molecule.id,
                target_id=target.id,
                success=False,
                error_message="No binding site defined for target",
            )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Write ligand PDBQT
                ligand_file = tmpdir / "ligand.pdbqt"
                ligand_file.write_text(molecule.pdbqt_content)

                # Write or link receptor PDBQT
                if target.pdbqt_path and target.pdbqt_path.exists():
                    receptor_file = target.pdbqt_path
                else:
                    receptor_file = tmpdir / "receptor.pdbqt"
                    receptor_file.write_text(target.pdbqt_content)

                # Create grid parameter file
                grid_config = target.primary_site.to_autodock_config()
                grid_file = tmpdir / "grid.gpf"
                self._write_grid_file(grid_file, receptor_file, grid_config)

                # Create docking parameter file
                dpf_file = tmpdir / "dock.dpf"
                self._write_dpf_file(dpf_file, ligand_file, config)

                # Run AutoDock-GPU
                output_file = tmpdir / "output.dlg"
                result = self._run_autodock(
                    receptor_file,
                    ligand_file,
                    grid_config,
                    output_file,
                    config,
                )

                if result.returncode != 0:
                    return DockingResult(
                        molecule_id=molecule.id,
                        target_id=target.id,
                        success=False,
                        error_message=f"AutoDock-GPU failed: {result.stderr}",
                        computation_time=time.time() - start_time,
                    )

                # Parse results
                poses = self._parse_dlg_file(output_file)

                return DockingResult(
                    molecule_id=molecule.id,
                    target_id=target.id,
                    binding_site_name=target.primary_site.name,
                    poses=poses,
                    computation_time=time.time() - start_time,
                    engine="autodock-gpu",
                    success=True,
                )

        except Exception as e:
            logger.error(f"Docking failed for {molecule.id}: {e}")
            return DockingResult(
                molecule_id=molecule.id,
                target_id=target.id,
                success=False,
                error_message=str(e),
                computation_time=time.time() - start_time,
            )

    def dock_batch(
        self,
        molecules: list[Molecule],
        target: ProteinTarget,
        config: Optional[DockingConfig] = None,
    ) -> Iterator[DockingResult]:
        """
        Dock a batch of molecules to a target.

        AutoDock-GPU can process multiple ligands in parallel on GPU.

        Args:
            molecules: List of prepared molecules.
            target: Prepared protein target.
            config: Docking configuration.

        Yields:
            DockingResult for each molecule.
        """
        config = config or DockingConfig()

        # Process in batches based on GPU memory
        batch_size = config.gpu_batch_size

        for i in range(0, len(molecules), batch_size):
            batch = molecules[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}, "
                       f"molecules {i + 1}-{min(i + batch_size, len(molecules))}")

            # For now, process sequentially (batch mode requires file setup)
            for mol in batch:
                yield self.dock(mol, target, config)

    def _run_autodock(
        self,
        receptor: Path,
        ligand: Path,
        grid_config: dict,
        output: Path,
        config: DockingConfig,
    ) -> subprocess.CompletedProcess:
        """Run AutoDock-GPU subprocess."""
        cmd = [
            str(self.executable),
            "--ffile", str(receptor),
            "--lfile", str(ligand),
            "--nrun", str(config.num_poses),
            "--nev", str(config.exhaustiveness * 1000000),
            "--resnam", str(output.stem),
        ]

        # Add grid center and size
        cmd.extend([
            "--xcenter", str(grid_config["center_x"]),
            "--ycenter", str(grid_config["center_y"]),
            "--zcenter", str(grid_config["center_z"]),
            "--xsize", str(grid_config["size_x"]),
            "--ysize", str(grid_config["size_y"]),
            "--zsize", str(grid_config["size_z"]),
        ])

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
            cwd=output.parent,
        )

    def _write_grid_file(self, path: Path, receptor: Path, config: dict) -> None:
        """Write AutoDock grid parameter file."""
        content = f"""npts {int(config['size_x'])} {int(config['size_y'])} {int(config['size_z'])}
gridfld receptor.maps.fld
spacing 0.375
receptor_types A C HD N NA OA SA
ligand_types A C HD N NA OA SA
receptor {receptor.name}
gridcenter {config['center_x']} {config['center_y']} {config['center_z']}
smooth 0.5
map receptor.A.map
map receptor.C.map
map receptor.HD.map
map receptor.N.map
map receptor.NA.map
map receptor.OA.map
map receptor.SA.map
elecmap receptor.e.map
dsolvmap receptor.d.map
dielectric -0.1465
"""
        path.write_text(content)

    def _write_dpf_file(self, path: Path, ligand: Path, config: DockingConfig) -> None:
        """Write AutoDock docking parameter file."""
        content = f"""autodock_parameter_version 4.2
outlev 1
intelec
seed pid time
ligand_types A C HD N NA OA SA
fld receptor.maps.fld
map receptor.A.map
map receptor.C.map
map receptor.HD.map
map receptor.N.map
map receptor.NA.map
map receptor.OA.map
map receptor.SA.map
elecmap receptor.e.map
desolvmap receptor.d.map
move {ligand.name}
about 0.0 0.0 0.0
tran0 random
quat0 random
dihe0 random
tstep 2.0
qstep 50.0
dstep 50.0
torsdof 0
rmstol 2.0
extnrg 1000.0
e0max 0.0 10000
ga_pop_size 150
ga_num_evals {config.exhaustiveness * 1000000}
ga_num_generations 27000
ga_elitism 1
ga_mutation_rate 0.02
ga_crossover_rate 0.8
ga_window_size 10
ga_cauchy_alpha 0.0
ga_cauchy_beta 1.0
set_ga
sw_max_its 300
sw_max_succ 4
sw_max_fail 4
sw_rho 1.0
sw_lb_rho 0.01
ls_search_freq 0.06
set_psw1
unbound_model bound
ga_run {config.num_poses}
analysis
"""
        path.write_text(content)

    def _parse_dlg_file(self, path: Path) -> list[DockingPose]:
        """Parse AutoDock DLG output file for poses."""
        poses = []

        if not path.exists():
            return poses

        content = path.read_text()
        current_pose = None
        pose_pdbqt_lines = []
        in_pose = False

        for line in content.split("\n"):
            # Look for DOCKED entries
            if line.startswith("DOCKED: MODEL"):
                in_pose = True
                pose_pdbqt_lines = []

            elif line.startswith("DOCKED: ENDMDL"):
                in_pose = False
                if current_pose:
                    current_pose.pdbqt_content = "\n".join(pose_pdbqt_lines)
                    poses.append(current_pose)
                    current_pose = None

            elif in_pose:
                # Remove DOCKED: prefix
                clean_line = line.replace("DOCKED: ", "")
                pose_pdbqt_lines.append(clean_line)

                # Parse energy from USER line
                if "Estimated Free Energy of Binding" in line:
                    try:
                        energy = float(line.split("=")[1].split()[0])
                        rank = len(poses) + 1
                        current_pose = DockingPose(rank=rank, energy=energy)
                    except (IndexError, ValueError):
                        pass

            # Parse RMSD from clustering
            if "RANKING" in line and current_pose:
                try:
                    parts = line.split()
                    rmsd_idx = parts.index("RMSD") if "RMSD" in parts else -1
                    if rmsd_idx > 0:
                        current_pose.rmsd_lb = float(parts[rmsd_idx + 1])
                except (IndexError, ValueError):
                    pass

        return poses


class VinaEngine(DockingEngineBase):
    """
    AutoDock Vina docking engine (CPU fallback).

    This engine provides CPU-based molecular docking using
    AutoDock Vina when GPU is not available.
    """

    def __init__(self, executable_path: Optional[Path] = None):
        """Initialize Vina engine."""
        self.executable = executable_path or self._find_executable()

    def _find_executable(self) -> Optional[Path]:
        """Find Vina executable in PATH."""
        for name in ["vina", "vina_1.2.5", "autodock_vina"]:
            path = shutil.which(name)
            if path:
                return Path(path)
        return None

    def is_available(self) -> bool:
        """Check if Vina is available."""
        if self.executable is None:
            return False
        try:
            result = subprocess.run(
                [str(self.executable), "--version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def dock(
        self,
        molecule: Molecule,
        target: ProteinTarget,
        config: Optional[DockingConfig] = None,
    ) -> DockingResult:
        """Dock a single molecule using Vina."""
        config = config or DockingConfig()
        start_time = time.time()

        if not molecule.is_prepared or not target.is_prepared:
            return DockingResult(
                molecule_id=molecule.id,
                target_id=target.id,
                success=False,
                error_message="Molecule or target not prepared",
            )

        if not target.primary_site:
            return DockingResult(
                molecule_id=molecule.id,
                target_id=target.id,
                success=False,
                error_message="No binding site defined",
            )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Write files
                ligand_file = tmpdir / "ligand.pdbqt"
                ligand_file.write_text(molecule.pdbqt_content)

                receptor_file = tmpdir / "receptor.pdbqt"
                if target.pdbqt_path and target.pdbqt_path.exists():
                    receptor_file = target.pdbqt_path
                else:
                    receptor_file.write_text(target.pdbqt_content)

                output_file = tmpdir / "output.pdbqt"
                site = target.primary_site

                # Run Vina
                cmd = [
                    str(self.executable),
                    "--receptor", str(receptor_file),
                    "--ligand", str(ligand_file),
                    "--out", str(output_file),
                    "--center_x", str(site.center[0]),
                    "--center_y", str(site.center[1]),
                    "--center_z", str(site.center[2]),
                    "--size_x", str(site.size[0]),
                    "--size_y", str(site.size[1]),
                    "--size_z", str(site.size[2]),
                    "--exhaustiveness", str(config.exhaustiveness),
                    "--num_modes", str(config.num_poses),
                    "--energy_range", str(config.energy_range),
                    "--cpu", str(config.cpu_threads),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )

                if result.returncode != 0:
                    return DockingResult(
                        molecule_id=molecule.id,
                        target_id=target.id,
                        success=False,
                        error_message=f"Vina failed: {result.stderr}",
                        computation_time=time.time() - start_time,
                    )

                # Parse output
                poses = self._parse_vina_output(output_file, result.stdout)

                return DockingResult(
                    molecule_id=molecule.id,
                    target_id=target.id,
                    binding_site_name=site.name,
                    poses=poses,
                    computation_time=time.time() - start_time,
                    engine="vina",
                    success=True,
                )

        except Exception as e:
            return DockingResult(
                molecule_id=molecule.id,
                target_id=target.id,
                success=False,
                error_message=str(e),
                computation_time=time.time() - start_time,
            )

    def dock_batch(
        self,
        molecules: list[Molecule],
        target: ProteinTarget,
        config: Optional[DockingConfig] = None,
    ) -> Iterator[DockingResult]:
        """Dock batch of molecules (sequential for Vina)."""
        for mol in molecules:
            yield self.dock(mol, target, config)

    def _parse_vina_output(self, output_file: Path, stdout: str) -> list[DockingPose]:
        """Parse Vina PDBQT output and stdout for poses."""
        poses = []

        # Parse energies from stdout
        energies = []
        for line in stdout.split("\n"):
            if line.strip() and line[0].isdigit():
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        energies.append(float(parts[1]))
                except ValueError:
                    continue

        # Parse PDBQT output for poses
        if output_file.exists():
            content = output_file.read_text()
            models = content.split("MODEL")

            for i, model in enumerate(models[1:], 1):  # Skip first empty split
                pose_content = "MODEL" + model.split("ENDMDL")[0] + "ENDMDL\n"
                energy = energies[i - 1] if i <= len(energies) else 0.0

                poses.append(DockingPose(
                    rank=i,
                    energy=energy,
                    pdbqt_content=pose_content,
                ))

        return poses


class DockingEngine:
    """
    Main docking engine that selects the best available backend.

    This class automatically selects between GPU and CPU docking
    based on availability.
    """

    def __init__(self, prefer_gpu: bool = True):
        """
        Initialize docking engine.

        Args:
            prefer_gpu: Whether to prefer GPU acceleration.
        """
        self.prefer_gpu = prefer_gpu
        self._gpu_engine = AutoDockGPUEngine()
        self._cpu_engine = VinaEngine()
        self._active_engine: Optional[DockingEngineBase] = None

        self._select_engine()

    def _select_engine(self) -> None:
        """Select the best available docking engine."""
        if self.prefer_gpu and self._gpu_engine.is_available():
            self._active_engine = self._gpu_engine
            logger.info("Using AutoDock-GPU for docking")
        elif self._cpu_engine.is_available():
            self._active_engine = self._cpu_engine
            logger.info("Using AutoDock Vina (CPU) for docking")
        else:
            logger.warning("No docking engine available!")
            self._active_engine = None

    @property
    def is_available(self) -> bool:
        """Check if any docking engine is available."""
        return self._active_engine is not None

    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is active."""
        return isinstance(self._active_engine, AutoDockGPUEngine)

    def dock(
        self,
        molecule: Molecule,
        target: ProteinTarget,
        config: Optional[DockingConfig] = None,
    ) -> DockingResult:
        """Dock a single molecule."""
        if not self._active_engine:
            return DockingResult(
                molecule_id=molecule.id,
                target_id=target.id,
                success=False,
                error_message="No docking engine available",
            )
        return self._active_engine.dock(molecule, target, config)

    def dock_batch(
        self,
        molecules: list[Molecule],
        target: ProteinTarget,
        config: Optional[DockingConfig] = None,
    ) -> Iterator[DockingResult]:
        """Dock a batch of molecules."""
        if not self._active_engine:
            for mol in molecules:
                yield DockingResult(
                    molecule_id=mol.id,
                    target_id=target.id,
                    success=False,
                    error_message="No docking engine available",
                )
            return

        yield from self._active_engine.dock_batch(molecules, target, config)
