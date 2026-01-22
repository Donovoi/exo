# EXO Platform Support

## Tier 1 support - tested and maintained

### Apple Silicon macOS
- Mac Studio: M3 Ultra
- Mac Mini: M4 Pro
- Macbook Pro: M5, M4 Max

### Linux CUDA (NVIDIA GPU)
- Requires: NVIDIA Driver 550.54.14+, GPU compute capability 7.0+ (Volta or newer)
- Install with: `uv sync --extra cuda`
- Tested hardware:
  - NVIDIA RTX 40xx series
  - NVIDIA DGX Spark

### Linux CPU
- Default installation: `uv sync`
- Works on any x86_64 or aarch64 Linux

### Windows Subsystem for Linux 2 (WSL2)
- Requires: Windows NVIDIA driver with WSL support
- Install with: `uv sync --extra cuda` inside WSL2
- See [docs/cuda-setup.md](docs/cuda-setup.md) for setup instructions

## Tier 2 support - checked occasionally, should run without crashing

Linux CUDA on various hardware:
- Framework Desktop with NVIDIA GPU

## Tier 3 support - minimal support and testing, but no theoretical reason it shouldn't work

(None currently)

# Planned

## Tier 2

Linux Vulkan Support -- depends heavily on ecosystem
- Framework Desktop with AMD/Intel GPU

## Longer term

Windows Native CUDA Support (without WSL2)

Windows CPU Support

