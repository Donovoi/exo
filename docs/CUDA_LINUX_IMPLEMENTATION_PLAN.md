# CUDA on Linux Support Implementation Plan

## Executive Summary

This document outlines the comprehensive implementation plan to add CUDA support for Linux (including Windows Subsystem for Linux 2) in exo. The MLX framework now officially supports CUDA on Linux via `mlx[cuda]`, enabling exo to leverage NVIDIA GPUs for inference.

## Current State Analysis

### MLX CUDA Support Status

As of 2025, MLX officially supports CUDA on Linux:

- **Installation**: `pip install "mlx[cuda]"` or `mlx[cuda-XX]` for specific CUDA versions
- **Requirements**:
  - CUDA 12.0+ 
  - NVIDIA Driver 550.54.14+
  - GPU with compute capability 7.0+ (Volta or later)
- **Supported Operations**: Most common operations including text generation and LLM inference work
- **Known Limitations**:
  - Some quantized operations not fully supported
  - FFTs and advanced linear algebra may not be available
  - No automatic CPU fallback for unsupported operations

### Current exo Architecture

1. **MLX Backend** (`src/exo/worker/engines/mlx/`):
   - Uses `mlx==0.30.3` on macOS (Metal)
   - Uses `mlx[cpu]==0.30.3` on Linux (CPU-only)
   - Distributed backends: `ring` (standard), `jaccl` (RDMA)

2. **Platform Detection**:
   - `pyproject.toml` uses `sys_platform` markers
   - Code uses `mx.metal.is_available()` for Metal-specific features
   - No CUDA availability detection exists

3. **Current Linux Limitation**:
   - README states: "exo currently runs on CPU on Linux"
   - PLATFORMS.md lists "Linux CUDA Support" as planned Tier 1

## Implementation Steps

### Step 1: Dependency Management Updates

**PR #1: Add CUDA optional dependency support**

**Changes Required:**

1. Update `pyproject.toml`:
   ```toml
   # Project dependencies - default to CPU on Linux, GPU on macOS
   dependencies = [
       ...
       "mlx==0.30.3; sys_platform == 'darwin'",
       "mlx[cpu]==0.30.3; sys_platform == 'linux'",  # Default for Linux
       ...
   ]
   
   # CUDA optional dependency
   [project.optional-dependencies]
   cuda = [
       "mlx[cuda]>=0.30.0; sys_platform == 'linux'",
   ]
   cuda12 = [
       "mlx[cuda-12]>=0.30.0; sys_platform == 'linux'",
   ]
   cuda13 = [
       "mlx[cuda-13]>=0.30.0; sys_platform == 'linux'",
   ]
   ```

2. Update `tool.uv` environments:
   ```toml
   [tool.uv]
   environments = [
       "sys_platform == 'darwin'",
       "sys_platform == 'linux'",
   ]
   ```

3. Update lockfile: `uv lock --all-features`

**Testing Strategy:**
- Verify installation works on CPU-only Linux
- Verify CUDA installation path on Linux with NVIDIA GPU
- Test that macOS installation is unaffected

---

### Step 2: Runtime Device Detection

**PR #2: Add CUDA/Metal/CPU device detection utilities**

**Changes Required:**

1. Create new utility module `src/exo/shared/device_detection.py`:
   ```python
   """Device detection utilities for exo."""
   
   import sys
   from typing import Literal
   
   import mlx.core as mx
   
   BackendType = Literal["metal", "cuda", "cpu"]
   
   def detect_available_backend() -> BackendType:
       """Detect the best available compute backend."""
       if sys.platform == "darwin":
           if mx.metal.is_available():
               return "metal"
           return "cpu"
       
       # Linux
       if mx.cuda.is_available():
           return "cuda"
       return "cpu"
   
   def get_device_info() -> dict[str, object]:
       """Get detailed device information."""
       backend = detect_available_backend()
       info: dict[str, object] = {"backend": backend}
       
       if backend == "metal":
           info["metal_info"] = mx.metal.device_info()
       elif backend == "cuda":
           # CUDA device info when available
           info["cuda_available"] = True
       else:
           info["cpu_only"] = True
       
       return info
   
   def is_gpu_available() -> bool:
       """Check if any GPU backend is available."""
       return detect_available_backend() in ("metal", "cuda")
   ```

2. Update `src/exo/worker/engines/mlx/utils_mlx.py`:
   - Replace `mx.metal.is_available()` calls with backend-agnostic checks
   - Handle CUDA-specific memory management

**Testing Strategy:**
- Unit tests with mocked `mx.cuda.is_available()` / `mx.metal.is_available()`
- Integration tests on CUDA-enabled CI runners (if available)
- Manual testing on Linux with NVIDIA GPU

---

### Step 3: Update MLX Engine for CUDA Compatibility

**PR #3: Make MLX engine CUDA-compatible**

**Changes Required:**

1. Update `src/exo/worker/engines/mlx/utils_mlx.py`:
   ```python
   from exo.shared.device_detection import detect_available_backend, is_gpu_available
   
   def set_wired_limit_for_model(model_size: Memory):
       """Set memory limits - only applicable for Metal backend."""
       backend = detect_available_backend()
       
       if backend == "metal":
           if not mx.metal.is_available():
               return
           # existing Metal code...
       elif backend == "cuda":
           # CUDA memory management (different approach)
           # CUDA uses explicit memory allocation, no wired limit concept
           pass
       # CPU backend - no memory limits needed
   ```

2. Update `src/exo/worker/engines/mlx/generator/generate.py`:
   - Ensure stream creation works with CUDA backend
   - Handle CUDA-specific synchronization if needed

3. Update distributed initialization:
   - Verify `ring` backend works over TCP on Linux
   - Add NCCL backend support for multi-GPU CUDA (future enhancement)

**Testing Strategy:**
- Run existing MLX tests on Linux
- Verify model loading works on CUDA
- Test single-node inference on CUDA
- Test basic distributed (ring) communication on Linux

---

### Step 4: CI/CD Pipeline for CUDA Testing

**PR #4: Add CUDA testing infrastructure**

**Changes Required:**

1. Create new workflow `.github/workflows/cuda-linux.yml`:
   ```yaml
   name: CUDA Linux Tests
   
   on:
     push:
       branches: [main, staging]
     pull_request:
       branches: [main, staging]
   
   jobs:
     cuda-tests:
       runs-on: [self-hosted, cuda]  # Requires self-hosted runner with NVIDIA GPU
       # Alternative: use GitHub-hosted GPU runners when available
       steps:
         - uses: actions/checkout@v4
         
         - name: Setup Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.13'
         
         - name: Install CUDA dependencies
           run: |
             uv sync --extra cuda
         
         - name: Run CUDA-specific tests
           run: |
             uv run pytest -m cuda
   ```

2. Add pytest markers for CUDA tests:
   ```toml
   # pyproject.toml
   [tool.pytest.ini_options]
   markers = [
       "slow: marks tests as slow (deselected by default)",
       "cuda: marks tests that require CUDA (deselected by default unless on CUDA runner)"
   ]
   ```

3. Create CUDA-specific test file `tests/test_cuda_backend.py`:
   ```python
   import pytest
   import sys
   
   # Skip entire module if not on Linux
   pytestmark = [
       pytest.mark.skipif(sys.platform != "linux", reason="CUDA only on Linux"),
       pytest.mark.cuda,
   ]
   
   def test_cuda_detection():
       """Test CUDA backend detection."""
       from exo.shared.device_detection import detect_available_backend
       
       backend = detect_available_backend()
       assert backend in ("cuda", "cpu")
   
   @pytest.mark.skipif(not cuda_available(), reason="No CUDA GPU available")
   def test_cuda_inference():
       """Test basic inference on CUDA."""
       # Test implementation
       pass
   ```

**Testing Strategy:**
- Manual testing on CUDA-enabled machines initially
- Self-hosted runner setup documentation
- Matrix testing: CPU-only Linux, CUDA Linux, macOS

---

### Step 5: Documentation Updates

**PR #5: Update documentation for CUDA Linux support**

**Changes Required:**

1. Update `README.md`:
   - Add Linux CUDA installation instructions
   - Add WSL2 setup instructions
   - Update hardware accelerator section

2. Update `PLATFORMS.md`:
   - Move Linux CUDA to Tier 1 (tested)
   - Add specific CUDA hardware tested
   - Document WSL2 support status

3. Create `docs/cuda-setup.md`:
   - Prerequisites (NVIDIA driver, CUDA toolkit)
   - Installation steps for native Linux
   - Installation steps for WSL2
   - Troubleshooting common issues
   - Performance considerations

4. Update `AGENTS.md`:
   - Add CUDA build commands
   - Document testing with `--extra cuda`

**Documentation Structure:**

```markdown
# CUDA Setup Guide for exo on Linux

## Prerequisites

### Native Linux
- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- NVIDIA Driver 550.54.14 or newer
- CUDA 12.0+ (included via MLX)
- Ubuntu 22.04/24.04, Fedora 40+, or similar

### Windows Subsystem for Linux 2 (WSL2)
- Windows 11 or Windows 10 21H2+
- WSL2 with Ubuntu 22.04+
- NVIDIA GPU Driver for Windows (with WSL support)
- DO NOT install NVIDIA drivers inside WSL

## Installation

### Standard (CPU-only)
```bash
uv sync
uv run exo
```

### With CUDA Support
```bash
uv sync --extra cuda
uv run exo
```

## Verification
```bash
# Check CUDA availability
uv run python -c "import mlx.core as mx; print('CUDA:', mx.cuda.is_available())"
```
```

---

### Step 6: Memory Management for CUDA

**PR #6: Implement CUDA-specific memory management**

**Changes Required:**

1. Update memory detection in `src/exo/shared/types/memory.py` or create CUDA-specific utilities

2. Add CUDA memory reporting:
   ```python
   def get_cuda_memory_info() -> dict[str, int]:
       """Get CUDA GPU memory information."""
       import subprocess
       import json
       
       try:
           result = subprocess.run(
               ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used", 
                "--format=csv,nounits,noheader"],
               capture_output=True, text=True
           )
           total, free, used = map(int, result.stdout.strip().split(", "))
           return {"total": total * 1024 * 1024, "free": free * 1024 * 1024, "used": used * 1024 * 1024}
       except Exception:
           return {"total": 0, "free": 0, "used": 0}
   ```

3. Update worker resource reporting for CUDA memory

**Testing Strategy:**
- Mock nvidia-smi output for unit tests
- Integration test on CUDA hardware
- Verify memory limits prevent OOM

---

### Step 7: Distributed CUDA Support (Future)

**PR #7: Add NCCL backend for multi-GPU CUDA**

This is a future enhancement once single-GPU support is stable.

**Changes Required:**
- Add NCCL initialization for multi-GPU setups
- Update instance types for CUDA distributed
- Test multi-node CUDA communication

---

## Testing Strategy Overview

### Unit Tests
| Test Area | Location | CUDA Specific |
|-----------|----------|---------------|
| Device detection | `tests/test_device_detection.py` | Yes |
| Backend selection | `tests/test_backend.py` | Yes |
| Memory utilities | `tests/test_memory.py` | Yes |

### Integration Tests
| Test Area | Location | Requirements |
|-----------|----------|--------------|
| Model loading | `tests/test_cuda_model_load.py` | CUDA GPU |
| Inference | `tests/test_cuda_inference.py` | CUDA GPU |
| Distributed (ring) | `tests/test_cuda_distributed.py` | Multiple nodes |

### Manual Testing Checklist
- [ ] Install on Ubuntu 24.04 with NVIDIA RTX GPU
- [ ] Install on WSL2 with NVIDIA GPU passthrough
- [ ] Run single-node inference with small model (Llama-3.2-1B)
- [ ] Run multi-node inference over TCP
- [ ] Verify dashboard shows CUDA backend
- [ ] Test memory reporting accuracy

### CI/CD Matrix
| Platform | Runner | Backend |
|----------|--------|---------|
| macOS (aarch64) | macos-26 | Metal |
| Linux (x86_64) | ubuntu-latest | CPU |
| Linux CUDA | self-hosted-cuda | CUDA |
| Linux (aarch64) | ubuntu-24.04-arm | CPU |

## WSL2 Specific Considerations

### Requirements
1. Windows 11 or Windows 10 21H2+
2. WSL2 (not WSL1)
3. NVIDIA GPU Driver for Windows with WSL support
4. Ubuntu 22.04 or later inside WSL2

### Setup Steps
1. Enable WSL2 and install Ubuntu
2. Install NVIDIA driver on Windows (NOT inside WSL)
3. Verify GPU access: `nvidia-smi` inside WSL
4. Install exo with CUDA: `uv sync --extra cuda`

### Known Limitations
- Slightly higher latency than native Linux
- Some CUDA features may not be available in WSL
- Performance ~90-95% of native Linux

## Rollout Plan

### Phase 1: Foundation (PRs #1-2)
- Dependency management
- Device detection utilities
- Timeline: 1-2 weeks

### Phase 2: Core Functionality (PRs #3-4)
- MLX engine CUDA compatibility
- Basic CI/CD
- Timeline: 2-3 weeks

### Phase 3: Documentation & Polish (PRs #5-6)
- Complete documentation
- Memory management
- Timeline: 1-2 weeks

### Phase 4: Advanced Features (PR #7)
- Multi-GPU NCCL support
- Performance optimization
- Timeline: Future

## Issue Tracking

Each PR should be linked to a corresponding GitHub issue for tracking:

1. **Issue: Add CUDA optional dependency support**
   - PR #1: Dependency management updates

2. **Issue: Implement runtime device detection**
   - PR #2: Device detection utilities

3. **Issue: Make MLX engine CUDA-compatible**
   - PR #3: MLX engine updates

4. **Issue: Add CUDA testing infrastructure**
   - PR #4: CI/CD pipeline

5. **Issue: Update documentation for CUDA Linux support**
   - PR #5: Documentation updates

6. **Issue: Implement CUDA memory management**
   - PR #6: Memory utilities

7. **Issue: Add NCCL backend for multi-GPU CUDA**
   - PR #7: Distributed CUDA (future)

## Success Criteria

1. Users can install exo with CUDA support using `uv sync --extra cuda`
2. exo detects and uses CUDA backend automatically on Linux with NVIDIA GPU
3. Single-node inference works correctly on CUDA
4. Distributed inference works over TCP (ring backend) on CUDA
5. Documentation covers installation, setup, and troubleshooting
6. WSL2 users can run exo with GPU acceleration
7. CI/CD validates CUDA functionality (when runners available)

## References

- [MLX CUDA Documentation](https://ml-explore.github.io/mlx/build/html/install.html)
- [MLX CUDA GitHub Discussion](https://github.com/ml-explore/mlx/discussions/2422)
- [NVIDIA CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Ubuntu WSL2 CUDA Guide](https://documentation.ubuntu.com/wsl/stable/howto/gpu-cuda/)
