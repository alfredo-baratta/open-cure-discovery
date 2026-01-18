"""
Open Cure Discovery CLI - Main entry point.

This module provides the command-line interface for managing
drug discovery screening campaigns.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="ocd",
    help="Open Cure Discovery - GPU-accelerated drug discovery",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Display version information."""
    if value:
        from src import __version__

        console.print(f"Open Cure Discovery v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """
    Open Cure Discovery - Accelerating drug discovery through open-source virtual screening.

    Run 'ocd COMMAND --help' for more information on a specific command.
    """
    pass


@app.command()
def init(
    disease: str = typer.Option(
        ...,
        "--disease",
        "-d",
        help="Disease focus area (e.g., 'oncology', 'lung-cancer', 'alzheimer')",
    ),
    path: Path = typer.Option(
        Path("."),
        "--path",
        "-p",
        help="Project directory path",
    ),
) -> None:
    """Initialize a new screening project for a specific disease."""
    console.print(
        Panel(
            f"[bold green]Initializing project for: {disease}[/bold green]",
            title="Open Cure Discovery",
        )
    )

    # Create project structure
    project_path = path / f"ocd-{disease}"
    directories = ["campaigns", "results", "logs", "cache"]

    try:
        project_path.mkdir(parents=True, exist_ok=True)
        for dir_name in directories:
            (project_path / dir_name).mkdir(exist_ok=True)

        # Create default config
        config_content = f"""# Open Cure Discovery Configuration
# Disease: {disease}

project:
  name: "{disease}-screening"
  disease: "{disease}"

hardware:
  gpu_memory_limit: 5000  # MB, leave buffer for system
  batch_size: auto

screening:
  top_candidates: 1000
  score_threshold: -7.0  # kcal/mol

output:
  format: ["csv", "sdf"]
  save_poses: true
"""
        config_file = project_path / "config.yaml"
        config_file.write_text(config_content)

        console.print(f"[green]✓[/green] Created project at: {project_path}")
        console.print(f"[green]✓[/green] Configuration file: {config_file}")
        console.print()
        console.print("Next steps:")
        console.print(f"  1. cd {project_path}")
        console.print(f"  2. ocd download --target <TARGET_NAME>")
        console.print(f"  3. ocd screen")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def check_gpu() -> None:
    """Check GPU availability and estimate performance."""
    console.print(
        Panel("[bold]GPU Detection & Performance Estimation[/bold]", title="Open Cure Discovery")
    )

    try:
        from src.utils.gpu.detector import GPUDetector

        detector = GPUDetector()
        info = detector.detect()

        if info.available:
            table = Table(title="GPU Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("GPU Model", info.name)
            table.add_row("VRAM", f"{info.memory_total} MB")
            table.add_row("VRAM Free", f"{info.memory_free} MB")
            table.add_row("CUDA Version", info.cuda_version)
            table.add_row("Compute Capability", info.compute_capability)
            table.add_row("Est. Molecules/Day", f"~{info.estimated_throughput:,}")

            console.print(table)
            console.print()
            console.print(
                f"[green]✓[/green] Recommended batch size: {info.recommended_batch_size}"
            )
        else:
            console.print("[yellow]No CUDA-capable GPU detected.[/yellow]")
            console.print("The system will run in CPU-only mode (slower).")

    except ImportError:
        console.print("[yellow]GPU detection requires PyTorch with CUDA.[/yellow]")
        console.print("Install with: pip install torch --index-url https://download.pytorch.org/whl/cu118")


@app.command()
def download(
    target: str = typer.Option(
        None,
        "--target",
        "-t",
        help="Target protein to download (e.g., 'EGFR', 'BRAF')",
    ),
    molecules: str = typer.Option(
        None,
        "--molecules",
        "-m",
        help="Molecule library to download (e.g., 'zinc-druglike', 'chembl-approved')",
    ),
    list_available: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List available targets and molecule libraries",
    ),
) -> None:
    """Download target proteins and molecule libraries."""
    if list_available:
        # Show available targets
        target_table = Table(title="Available Targets")
        target_table.add_column("Name", style="cyan")
        target_table.add_column("Disease Area", style="green")
        target_table.add_column("PDB ID", style="yellow")

        targets = [
            ("EGFR", "Oncology (NSCLC)", "1M17"),
            ("HER2", "Oncology (Breast)", "3PP0"),
            ("BRAF", "Oncology (Melanoma)", "4RZV"),
            ("KRAS-G12C", "Oncology (Multiple)", "6OIM"),
            ("BCL2", "Oncology (Leukemia)", "6O0K"),
            ("Main-Protease", "COVID-19", "6LU7"),
            ("ACE2", "Cardiovascular", "1R42"),
            ("Beta-Secretase", "Alzheimer's", "4B05"),
        ]

        for name, disease, pdb in targets:
            target_table.add_row(name, disease, pdb)

        console.print(target_table)
        console.print()

        # Show available molecule libraries
        mol_table = Table(title="Available Molecule Libraries")
        mol_table.add_column("Name", style="cyan")
        mol_table.add_column("Size", style="green")
        mol_table.add_column("Description", style="white")

        libraries = [
            ("zinc-druglike", "~1.5M", "Drug-like subset from ZINC"),
            ("zinc-leads", "~5M", "Lead-like compounds from ZINC"),
            ("chembl-approved", "~2.5K", "FDA-approved drugs from ChEMBL"),
            ("chembl-bioactive", "~500K", "Bioactive compounds from ChEMBL"),
            ("natural-products", "~200K", "Natural product derivatives"),
        ]

        for name, size, desc in libraries:
            mol_table.add_row(name, size, desc)

        console.print(mol_table)
        return

    if not target and not molecules:
        console.print("[yellow]Specify --target, --molecules, or --list[/yellow]")
        raise typer.Exit(1)

    if target:
        console.print(f"[cyan]Downloading target:[/cyan] {target}")
        console.print("[yellow]Note: Download functionality not yet implemented[/yellow]")

    if molecules:
        console.print(f"[cyan]Downloading molecules:[/cyan] {molecules}")
        console.print("[yellow]Note: Download functionality not yet implemented[/yellow]")


@app.command()
def screen(
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        "-c",
        help="Path to campaign configuration file",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from last checkpoint",
    ),
) -> None:
    """Run a virtual screening campaign."""
    console.print(
        Panel(
            "[bold]Starting Virtual Screening Campaign[/bold]",
            title="Open Cure Discovery",
        )
    )

    if not config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        console.print("Run 'ocd init' first to create a project.")
        raise typer.Exit(1)

    console.print(f"[cyan]Configuration:[/cyan] {config}")
    console.print(f"[cyan]Resume:[/cyan] {resume}")
    console.print()
    console.print("[yellow]Note: Screening engine not yet implemented[/yellow]")
    console.print("This is where the docking and ML prediction pipeline would run.")


@app.command()
def results(
    top: int = typer.Option(
        100,
        "--top",
        "-n",
        help="Number of top results to show",
    ),
    export: str = typer.Option(
        None,
        "--export",
        "-e",
        help="Export format (csv, sdf, json)",
    ),
    campaign: Path = typer.Option(
        Path("."),
        "--campaign",
        "-c",
        help="Campaign directory",
    ),
) -> None:
    """View and export screening results."""
    console.print(
        Panel(
            f"[bold]Screening Results (Top {top})[/bold]",
            title="Open Cure Discovery",
        )
    )

    results_dir = campaign / "results"
    if not results_dir.exists():
        console.print("[yellow]No results found.[/yellow]")
        console.print("Run 'ocd screen' first to generate results.")
        return

    console.print("[yellow]Note: Results viewer not yet implemented[/yellow]")

    if export:
        console.print(f"[cyan]Would export to:[/cyan] {export} format")


@app.command()
def status() -> None:
    """Show current campaign status."""
    console.print(
        Panel(
            "[bold]Campaign Status[/bold]",
            title="Open Cure Discovery",
        )
    )

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", "Not Running")
    table.add_row("Molecules Processed", "0")
    table.add_row("Current Target", "N/A")
    table.add_row("Top Score", "N/A")
    table.add_row("ETA", "N/A")

    console.print(table)
    console.print()
    console.print("[dim]Run 'ocd screen' to start a campaign[/dim]")


if __name__ == "__main__":
    app()
