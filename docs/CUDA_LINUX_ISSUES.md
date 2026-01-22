# GitHub Issues for CUDA Linux Support

This document contains the issue templates for each step of the CUDA Linux implementation. Each issue should be created separately and linked to its corresponding PR.

---

## Issue 1: Add CUDA Optional Dependency Support

**Title:** `[CUDA] Add CUDA optional dependency support for Linux`

**Labels:** `enhancement`, `linux`, `cuda`

**Description:**

### Summary
Add optional CUDA dependency support to enable GPU acceleration on Linux systems with NVIDIA GPUs.

### Background
MLX now officially supports CUDA on Linux via `mlx[cuda]`. This issue tracks adding the optional dependency configuration to exo's build system.

### Requirements
- Add `cuda`, `cuda12`, and `cuda13` optional dependencies to `pyproject.toml`
- Default Linux installation should remain CPU-only (`mlx[cpu]`)
- Users can opt-in to CUDA with `uv sync --extra cuda`
- macOS installation should be unaffected

### Acceptance Criteria
- [ ] `uv sync` on Linux installs `mlx[cpu]` (no CUDA)
- [ ] `uv sync --extra cuda` on Linux installs `mlx[cuda]`
- [ ] macOS installation works unchanged
- [ ] Lock file regenerated and tested

### Implementation Notes
See [CUDA_LINUX_IMPLEMENTATION_PLAN.md](./CUDA_LINUX_IMPLEMENTATION_PLAN.md#step-1-dependency-management-updates) for detailed implementation.

---

## Issue 2: Implement Runtime Device Detection

**Title:** `[CUDA] Implement runtime device detection utilities`

**Labels:** `enhancement`, `linux`, `cuda`

**Description:**

### Summary
Create device detection utilities that can identify whether Metal (macOS), CUDA (Linux), or CPU-only backends are available at runtime.

### Background
exo needs to know which compute backend to use at runtime. Currently, the code only checks for Metal availability on macOS. We need backend-agnostic detection.

### Requirements
- Create `src/exo/shared/device_detection.py` module
- Implement `detect_available_backend()` function
- Implement `get_device_info()` function
- Implement `is_gpu_available()` function
- Support Metal, CUDA, and CPU backends

### Acceptance Criteria
- [ ] Detection returns "metal" on macOS with Metal support
- [ ] Detection returns "cuda" on Linux with NVIDIA GPU
- [ ] Detection returns "cpu" on systems without GPU support
- [ ] Unit tests cover all backends (with mocking)
- [ ] Type checking passes

### Dependencies
- Issue #1 (CUDA dependency support)

### Implementation Notes
See [CUDA_LINUX_IMPLEMENTATION_PLAN.md](./CUDA_LINUX_IMPLEMENTATION_PLAN.md#step-2-runtime-device-detection) for detailed implementation.

---

## Issue 3: Make MLX Engine CUDA-Compatible

**Title:** `[CUDA] Update MLX engine for CUDA compatibility`

**Labels:** `enhancement`, `linux`, `cuda`

**Description:**

### Summary
Update the MLX inference engine to work with the CUDA backend on Linux while maintaining Metal support on macOS.

### Background
The current MLX engine code contains macOS-specific calls like `mx.metal.is_available()` and Metal-specific memory management. These need to be made backend-agnostic.

### Requirements
- Update `src/exo/worker/engines/mlx/utils_mlx.py` for backend-agnostic operation
- Replace `mx.metal.is_available()` calls with device detection utilities
- Handle CUDA-specific memory management (no wired limit concept)
- Ensure stream creation works with CUDA backend
- Verify distributed `ring` backend works over TCP on Linux

### Acceptance Criteria
- [ ] Model loading works on CUDA backend
- [ ] Single-node inference produces correct output on CUDA
- [ ] Distributed inference (ring over TCP) works on Linux
- [ ] No regressions on macOS Metal
- [ ] Type checking passes

### Dependencies
- Issue #1 (CUDA dependency support)
- Issue #2 (Device detection utilities)

### Implementation Notes
See [CUDA_LINUX_IMPLEMENTATION_PLAN.md](./CUDA_LINUX_IMPLEMENTATION_PLAN.md#step-3-update-mlx-engine-for-cuda-compatibility) for detailed implementation.

---

## Issue 4: Add CUDA Testing Infrastructure

**Title:** `[CUDA] Add CUDA testing infrastructure and CI pipeline`

**Labels:** `enhancement`, `linux`, `cuda`, `ci/cd`

**Description:**

### Summary
Create testing infrastructure for CUDA-specific functionality, including pytest markers and optional CI/CD pipeline support.

### Background
To ensure CUDA support works correctly, we need dedicated tests that can verify CUDA functionality. These tests should be skipped automatically when CUDA is not available.

### Requirements
- Add `cuda` pytest marker to `pyproject.toml`
- Create `tests/test_cuda_backend.py` with CUDA-specific tests
- Create `.github/workflows/cuda-linux.yml` (optional, for self-hosted runners)
- Document manual testing procedures

### Acceptance Criteria
- [ ] `pytest -m cuda` runs only CUDA tests
- [ ] CUDA tests skip gracefully on non-CUDA systems
- [ ] CI workflow file exists (even if not active)
- [ ] Manual testing checklist documented
- [ ] Tests cover: device detection, model loading, basic inference

### Dependencies
- Issue #1 (CUDA dependency support)
- Issue #2 (Device detection utilities)
- Issue #3 (CUDA-compatible MLX engine)

### Implementation Notes
See [CUDA_LINUX_IMPLEMENTATION_PLAN.md](./CUDA_LINUX_IMPLEMENTATION_PLAN.md#step-4-cicd-pipeline-for-cuda-testing) for detailed implementation.

---

## Issue 5: Update Documentation for CUDA Linux Support

**Title:** `[CUDA] Update documentation for CUDA Linux and WSL2 support`

**Labels:** `documentation`, `linux`, `cuda`

**Description:**

### Summary
Update all relevant documentation to reflect CUDA support on Linux, including native Linux and WSL2 (Windows Subsystem for Linux).

### Background
Users need clear instructions on how to install and run exo with CUDA support on Linux systems, including the increasingly popular WSL2 environment.

### Requirements
- Update `README.md` with Linux CUDA installation instructions
- Update `PLATFORMS.md` to reflect CUDA support status
- Create `docs/cuda-setup.md` with detailed setup guide
- Update `AGENTS.md` with CUDA build/test commands
- Include WSL2-specific instructions

### Documentation Contents
1. **Prerequisites**: NVIDIA driver, CUDA requirements
2. **Installation**: `uv sync --extra cuda` instructions
3. **WSL2 Setup**: Windows driver installation, WSL2 configuration
4. **Verification**: How to verify CUDA is working
5. **Troubleshooting**: Common issues and solutions
6. **Performance**: Tips for optimal performance

### Acceptance Criteria
- [ ] README has clear Linux CUDA installation section
- [ ] PLATFORMS.md updated with CUDA tier status
- [ ] docs/cuda-setup.md covers all setup scenarios
- [ ] WSL2-specific instructions included
- [ ] AGENTS.md updated with CUDA commands

### Dependencies
- Issue #1 (CUDA dependency support)
- Issue #3 (CUDA-compatible MLX engine)

### Implementation Notes
See [CUDA_LINUX_IMPLEMENTATION_PLAN.md](./CUDA_LINUX_IMPLEMENTATION_PLAN.md#step-5-documentation-updates) for detailed implementation.

---

## Issue 6: Implement CUDA Memory Management

**Title:** `[CUDA] Implement CUDA-specific memory management`

**Labels:** `enhancement`, `linux`, `cuda`

**Description:**

### Summary
Implement CUDA-specific memory management utilities to properly report and manage GPU memory on NVIDIA hardware.

### Background
The current memory management uses macOS Metal-specific APIs (`mx.metal.device_info()`, `mx.set_wired_limit()`). CUDA requires different memory management approaches.

### Requirements
- Create CUDA memory reporting utilities (via nvidia-smi or CUDA API)
- Update worker resource reporting for CUDA memory
- Implement memory limit enforcement for CUDA (if applicable)
- Add memory info to dashboard/API

### Acceptance Criteria
- [ ] CUDA GPU memory accurately reported
- [ ] Memory usage visible in dashboard
- [ ] Memory limits prevent OOM crashes
- [ ] Works with multi-GPU systems
- [ ] Unit tests with mocked nvidia-smi

### Dependencies
- Issue #2 (Device detection utilities)
- Issue #3 (CUDA-compatible MLX engine)

### Implementation Notes
See [CUDA_LINUX_IMPLEMENTATION_PLAN.md](./CUDA_LINUX_IMPLEMENTATION_PLAN.md#step-6-memory-management-for-cuda) for detailed implementation.

---

## Issue 7: Add NCCL Backend for Multi-GPU CUDA (Future)

**Title:** `[CUDA] Add NCCL backend for multi-GPU distributed inference`

**Labels:** `enhancement`, `linux`, `cuda`, `future`

**Description:**

### Summary
Add support for NVIDIA NCCL (NVIDIA Collective Communications Library) as a distributed backend for efficient multi-GPU communication on Linux.

### Background
While the `ring` backend works over TCP, NCCL provides optimized collective operations for NVIDIA GPUs, including NVLink support for high-bandwidth communication.

### Requirements
- Add NCCL backend initialization in distributed code
- Create new instance type for NCCL-based instances
- Handle multi-GPU topology detection
- Support NVLink when available

### Acceptance Criteria
- [ ] NCCL backend initializes on multi-GPU systems
- [ ] Distributed inference works with NCCL
- [ ] NVLink utilized when available
- [ ] Fallback to TCP ring when NCCL unavailable
- [ ] Performance benchmarks show improvement over ring

### Dependencies
- Issue #1-6 (All previous CUDA issues)
- Stable single-GPU CUDA support

### Notes
This is a **future enhancement** and should be prioritized after single-GPU support is stable and tested.

### Implementation Notes
See [CUDA_LINUX_IMPLEMENTATION_PLAN.md](./CUDA_LINUX_IMPLEMENTATION_PLAN.md#step-7-distributed-cuda-support-future) for detailed implementation.

---

## Summary Table

| Issue | Title | Priority | Dependencies |
|-------|-------|----------|--------------|
| #1 | Add CUDA optional dependency support | P0 | None |
| #2 | Implement runtime device detection | P0 | #1 |
| #3 | Make MLX engine CUDA-compatible | P0 | #1, #2 |
| #4 | Add CUDA testing infrastructure | P1 | #1, #2, #3 |
| #5 | Update documentation | P1 | #1, #3 |
| #6 | Implement CUDA memory management | P1 | #2, #3 |
| #7 | Add NCCL backend (Future) | P2 | All |

## Timeline

- **Phase 1 (Weeks 1-2):** Issues #1, #2
- **Phase 2 (Weeks 3-4):** Issues #3, #4
- **Phase 3 (Weeks 5-6):** Issues #5, #6
- **Future:** Issue #7
