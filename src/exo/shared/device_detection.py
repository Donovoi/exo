"""Device detection utilities for exo.

This module provides backend-agnostic device detection to support
Metal (macOS), CUDA (Linux), and CPU-only backends.
"""

import sys
from typing import Literal, final

import mlx.core as mx

from exo.shared.logging import get_logger

BackendType = Literal["metal", "cuda", "cpu"]

logger = get_logger(__name__)


@final
class DeviceInfo:
    """Information about the detected compute device."""

    def __init__(
        self,
        backend: BackendType,
        is_gpu: bool,
        device_name: str | None = None,
        memory_bytes: int | None = None,
    ) -> None:
        self.backend = backend
        self.is_gpu = is_gpu
        self.device_name = device_name
        self.memory_bytes = memory_bytes

    def __repr__(self) -> str:
        return (
            f"DeviceInfo(backend={self.backend!r}, is_gpu={self.is_gpu}, "
            f"device_name={self.device_name!r}, memory_bytes={self.memory_bytes})"
        )


def detect_available_backend() -> BackendType:
    """Detect the best available compute backend.

    Returns:
        "metal" on macOS with Metal support
        "cuda" on Linux with NVIDIA GPU and CUDA support
        "cpu" on systems without GPU support
    """
    if sys.platform == "darwin":
        if mx.metal.is_available():
            logger.debug("Metal backend detected")
            return "metal"
        logger.debug("macOS without Metal support, falling back to CPU")
        return "cpu"

    # Linux (native or WSL2)
    if sys.platform == "linux":
        if mx.cuda.is_available():
            logger.debug("CUDA backend detected")
            return "cuda"
        logger.debug("Linux without CUDA support, using CPU")
        return "cpu"

    # Fallback for other platforms
    logger.debug(f"Unsupported platform {sys.platform}, using CPU")
    return "cpu"


def is_gpu_available() -> bool:
    """Check if any GPU backend is available.

    Returns:
        True if Metal (macOS) or CUDA (Linux) is available, False otherwise.
    """
    return detect_available_backend() in ("metal", "cuda")


def get_device_info() -> DeviceInfo:
    """Get detailed information about the detected compute device.

    Returns:
        DeviceInfo object with backend type and device details.
    """
    backend = detect_available_backend()

    if backend == "metal":
        try:
            # mx.metal.device_info() returns a dict with device details
            metal_info = mx.metal.device_info()
            device_name = str(metal_info.get("device_name", "Unknown Metal Device"))
            max_rec_size = metal_info.get("max_recommended_working_set_size")
            memory_bytes = int(max_rec_size) if max_rec_size is not None else None
            return DeviceInfo(
                backend="metal",
                is_gpu=True,
                device_name=device_name,
                memory_bytes=memory_bytes,
            )
        except Exception as e:
            logger.warning(f"Failed to get Metal device info: {e}")
            return DeviceInfo(backend="metal", is_gpu=True)

    if backend == "cuda":
        # CUDA device info - basic detection
        # Note: Detailed CUDA info requires nvidia-smi or CUDA API
        return DeviceInfo(
            backend="cuda",
            is_gpu=True,
            device_name="NVIDIA CUDA Device",
            memory_bytes=None,  # Can be extended to query nvidia-smi
        )

    # CPU backend
    return DeviceInfo(
        backend="cpu",
        is_gpu=False,
        device_name="CPU",
        memory_bytes=None,
    )


def is_metal_available() -> bool:
    """Check if Metal backend is available (macOS only).

    This is a convenience wrapper around mx.metal.is_available() that
    also checks the platform.

    Returns:
        True if on macOS and Metal is available.
    """
    return sys.platform == "darwin" and mx.metal.is_available()


def is_cuda_available() -> bool:
    """Check if CUDA backend is available (Linux only).

    This is a convenience wrapper around mx.cuda.is_available() that
    also checks the platform.

    Returns:
        True if on Linux and CUDA is available.
    """
    return sys.platform == "linux" and mx.cuda.is_available()
