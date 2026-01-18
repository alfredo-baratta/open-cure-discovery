"""
GPU detection and performance estimation utilities.

This module provides functionality to detect NVIDIA GPUs,
check CUDA availability, and estimate screening performance.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Information about a detected GPU."""

    available: bool
    name: str = "N/A"
    memory_total: int = 0  # MB
    memory_free: int = 0  # MB
    cuda_version: str = "N/A"
    compute_capability: str = "N/A"
    estimated_throughput: int = 0  # molecules per day
    recommended_batch_size: int = 32

    @property
    def memory_used(self) -> int:
        """Calculate used memory in MB."""
        return self.memory_total - self.memory_free


class GPUDetector:
    """
    Detect GPU hardware and estimate performance.

    This class checks for CUDA-capable GPUs and provides
    performance estimates for molecular docking workloads.
    """

    # Performance estimates based on GPU generation (molecules/day)
    GPU_PERFORMANCE_MAP = {
        # Pascal (10xx series)
        "1060": 100_000,
        "1070": 150_000,
        "1080": 200_000,
        # Turing (20xx series)
        "2060": 180_000,
        "2070": 250_000,
        "2080": 350_000,
        # Ampere (30xx series)
        "3060": 300_000,
        "3070": 450_000,
        "3080": 600_000,
        "3090": 800_000,
        # Ada Lovelace (40xx series)
        "4060": 400_000,
        "4070": 600_000,
        "4080": 900_000,
        "4090": 1_200_000,
    }

    # Recommended batch sizes based on VRAM (MB)
    BATCH_SIZE_MAP = [
        (4000, 64),  # 4GB
        (6000, 128),  # 6GB
        (8000, 192),  # 8GB
        (12000, 256),  # 12GB
        (16000, 384),  # 16GB
        (24000, 512),  # 24GB
    ]

    def detect(self) -> GPUInfo:
        """
        Detect GPU and return information.

        Returns:
            GPUInfo object with GPU details and performance estimates.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return GPUInfo(available=False)

            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)

            # Get memory info
            memory_total = props.total_memory // (1024 * 1024)  # Convert to MB
            memory_free = (
                torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            ) // (1024 * 1024)

            # For initial detection, estimate free as ~90% of total
            if memory_free <= 0:
                memory_free = int(memory_total * 0.9)

            # Get CUDA version
            cuda_version = torch.version.cuda or "Unknown"

            # Get compute capability
            compute_capability = f"{props.major}.{props.minor}"

            # Estimate performance
            estimated_throughput = self._estimate_performance(props.name, memory_total)

            # Recommend batch size
            recommended_batch_size = self._recommend_batch_size(memory_total)

            return GPUInfo(
                available=True,
                name=props.name,
                memory_total=memory_total,
                memory_free=memory_free,
                cuda_version=cuda_version,
                compute_capability=compute_capability,
                estimated_throughput=estimated_throughput,
                recommended_batch_size=recommended_batch_size,
            )

        except ImportError:
            return GPUInfo(available=False)
        except Exception:
            return GPUInfo(available=False)

    def _estimate_performance(self, gpu_name: str, vram_mb: int) -> int:
        """
        Estimate molecules per day based on GPU model.

        Args:
            gpu_name: Full GPU name string.
            vram_mb: Total VRAM in MB.

        Returns:
            Estimated molecules processed per day.
        """
        gpu_name_upper = gpu_name.upper()

        # Try to match known GPU models
        for model, performance in self.GPU_PERFORMANCE_MAP.items():
            if model in gpu_name_upper:
                return performance

        # Fallback: estimate based on VRAM
        # Rough estimate: ~15,000 molecules/day per GB of VRAM
        vram_gb = vram_mb / 1024
        return int(vram_gb * 15_000)

    def _recommend_batch_size(self, vram_mb: int) -> int:
        """
        Recommend batch size based on available VRAM.

        Args:
            vram_mb: Total VRAM in MB.

        Returns:
            Recommended batch size.
        """
        for threshold, batch_size in self.BATCH_SIZE_MAP:
            if vram_mb <= threshold:
                return batch_size

        # For very large GPUs
        return 512

    def get_memory_usage(self) -> Optional[tuple[int, int]]:
        """
        Get current GPU memory usage.

        Returns:
            Tuple of (used_mb, total_mb) or None if unavailable.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return None

            device = torch.cuda.current_device()
            used = torch.cuda.memory_allocated(device) // (1024 * 1024)
            total = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)

            return (used, total)

        except Exception:
            return None


def check_cuda_compatibility() -> dict:
    """
    Check CUDA compatibility and return diagnostic info.

    Returns:
        Dictionary with compatibility information.
    """
    result = {
        "torch_installed": False,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_available": False,
        "cudnn_version": None,
        "gpu_count": 0,
        "errors": [],
    }

    try:
        import torch

        result["torch_installed"] = True
        result["cuda_available"] = torch.cuda.is_available()

        if result["cuda_available"]:
            result["cuda_version"] = torch.version.cuda
            result["gpu_count"] = torch.cuda.device_count()

            if torch.backends.cudnn.is_available():
                result["cudnn_available"] = True
                result["cudnn_version"] = torch.backends.cudnn.version()

    except ImportError as e:
        result["errors"].append(f"PyTorch not installed: {e}")
    except Exception as e:
        result["errors"].append(f"Error checking CUDA: {e}")

    return result
