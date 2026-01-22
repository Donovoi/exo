"""Tests for device detection utilities."""

import sys
from unittest.mock import patch


class TestDetectAvailableBackend:
    """Tests for detect_available_backend function."""

    def test_darwin_with_metal(self):
        """Test Metal detection on macOS."""
        with (
            patch.object(sys, "platform", "darwin"),
            patch("mlx.core.metal.is_available", return_value=True),
        ):
            # Need to reimport to pick up the mocked platform
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.metal, "is_available", return_value=True):
                result = dd.detect_available_backend()
                assert result == "metal"

    def test_darwin_without_metal(self):
        """Test CPU fallback on macOS without Metal."""
        with patch.object(sys, "platform", "darwin"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.metal, "is_available", return_value=False):
                result = dd.detect_available_backend()
                assert result == "cpu"

    def test_linux_with_cuda(self):
        """Test CUDA detection on Linux."""
        with patch.object(sys, "platform", "linux"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.cuda, "is_available", return_value=True):
                result = dd.detect_available_backend()
                assert result == "cuda"

    def test_linux_without_cuda(self):
        """Test CPU fallback on Linux without CUDA."""
        with patch.object(sys, "platform", "linux"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.cuda, "is_available", return_value=False):
                result = dd.detect_available_backend()
                assert result == "cpu"


class TestIsGpuAvailable:
    """Tests for is_gpu_available function."""

    def test_gpu_available_on_metal(self):
        """Test GPU available returns True for Metal."""
        with patch.object(sys, "platform", "darwin"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.metal, "is_available", return_value=True):
                result = dd.is_gpu_available()
                assert result is True

    def test_gpu_available_on_cuda(self):
        """Test GPU available returns True for CUDA."""
        with patch.object(sys, "platform", "linux"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.cuda, "is_available", return_value=True):
                result = dd.is_gpu_available()
                assert result is True

    def test_gpu_not_available_on_cpu(self):
        """Test GPU available returns False for CPU-only."""
        with patch.object(sys, "platform", "linux"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.cuda, "is_available", return_value=False):
                result = dd.is_gpu_available()
                assert result is False


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_metal_device_info(self):
        """Test DeviceInfo for Metal backend."""
        with patch.object(sys, "platform", "darwin"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            mock_metal_info = {
                "device_name": "Apple M3 Ultra",
                "max_recommended_working_set_size": 512 * 1024 * 1024 * 1024,
            }

            with (
                patch.object(dd.mx.metal, "is_available", return_value=True),
                patch.object(dd.mx.metal, "device_info", return_value=mock_metal_info),
            ):
                info = dd.get_device_info()
                assert info.backend == "metal"
                assert info.is_gpu is True
                assert info.device_name == "Apple M3 Ultra"
                assert info.memory_bytes == 512 * 1024 * 1024 * 1024

    def test_cuda_device_info(self):
        """Test DeviceInfo for CUDA backend."""
        with patch.object(sys, "platform", "linux"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.cuda, "is_available", return_value=True):
                info = dd.get_device_info()
                assert info.backend == "cuda"
                assert info.is_gpu is True
                assert info.device_name == "NVIDIA CUDA Device"

    def test_cpu_device_info(self):
        """Test DeviceInfo for CPU backend."""
        with patch.object(sys, "platform", "linux"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.cuda, "is_available", return_value=False):
                info = dd.get_device_info()
                assert info.backend == "cpu"
                assert info.is_gpu is False
                assert info.device_name == "CPU"


class TestConvenienceFunctions:
    """Tests for is_metal_available and is_cuda_available."""

    def test_is_metal_available_on_darwin(self):
        """Test is_metal_available on macOS."""
        with patch.object(sys, "platform", "darwin"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.metal, "is_available", return_value=True):
                assert dd.is_metal_available() is True

            with patch.object(dd.mx.metal, "is_available", return_value=False):
                assert dd.is_metal_available() is False

    def test_is_metal_available_on_linux(self):
        """Test is_metal_available returns False on Linux."""
        with patch.object(sys, "platform", "linux"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            # Should always return False on Linux regardless of metal availability
            assert dd.is_metal_available() is False

    def test_is_cuda_available_on_linux(self):
        """Test is_cuda_available on Linux."""
        with patch.object(sys, "platform", "linux"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            with patch.object(dd.mx.cuda, "is_available", return_value=True):
                assert dd.is_cuda_available() is True

            with patch.object(dd.mx.cuda, "is_available", return_value=False):
                assert dd.is_cuda_available() is False

    def test_is_cuda_available_on_darwin(self):
        """Test is_cuda_available returns False on macOS."""
        with patch.object(sys, "platform", "darwin"):
            import importlib

            import exo.shared.device_detection as dd

            importlib.reload(dd)

            # Should always return False on macOS regardless of cuda availability
            assert dd.is_cuda_available() is False
