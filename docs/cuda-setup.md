# CUDA Setup Guide for exo on Linux

This guide covers how to set up exo with CUDA support on native Linux and Windows Subsystem for Linux 2 (WSL2).

## Prerequisites

### Hardware Requirements

- **NVIDIA GPU** with compute capability 7.0 or higher:
  - Volta (V100, Titan V)
  - Turing (RTX 20xx, GTX 16xx, Quadro RTX)
  - Ampere (RTX 30xx, A100, A6000)
  - Ada Lovelace (RTX 40xx, L40, L4)
  - Hopper (H100, H200)

### Software Requirements

#### Native Linux

- **Operating System**: Ubuntu 22.04+, Fedora 40+, or similar modern Linux distribution
- **NVIDIA Driver**: Version 550.54.14 or newer
- **Python**: 3.13+
- **uv**: Python package manager
- **Node.js**: 18+ (for building the dashboard)
- **Rust**: Nightly toolchain (for building bindings)

#### Windows Subsystem for Linux 2 (WSL2)

- **Windows**: Windows 11 or Windows 10 21H2+
- **WSL2**: With Ubuntu 22.04 or later
- **NVIDIA Driver**: Windows NVIDIA driver with WSL support (do NOT install Linux drivers inside WSL)

## Installation

### Step 1: Verify NVIDIA Driver (Native Linux)

```bash
# Check driver version
nvidia-smi

# Expected output shows driver version 550+ and CUDA version
```

### Step 1 (WSL2): Install Windows NVIDIA Driver

1. Download the latest NVIDIA driver for Windows from [NVIDIA's website](https://www.nvidia.com/drivers)
2. Install the driver on Windows (not inside WSL)
3. Open WSL2 and verify GPU access:

```bash
# Inside WSL2
nvidia-smi

# Should show your GPU - no additional driver installation needed in WSL
```

### Step 2: Install Dependencies

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust (nightly)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly

# Install Node.js (Ubuntu/Debian)
sudo apt update
sudo apt install nodejs npm

# Or using nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
nvm install 18
```

### Step 3: Clone and Build exo

```bash
# Clone the repository
git clone https://github.com/exo-explore/exo
cd exo

# Build the dashboard
cd dashboard && npm install && npm run build && cd ..

# Install with CUDA support
uv sync --extra cuda
```

### Step 4: Verify CUDA Support

```bash
# Check if CUDA is detected
uv run python -c "import mlx.core as mx; print('CUDA available:', mx.cuda.is_available())"

# Expected output: CUDA available: True
```

### Step 5: Run exo

```bash
# Start exo (will automatically use CUDA if available)
uv run exo
```

## Verification

### Check Device Detection

exo automatically detects the best available backend. You can verify with:

```bash
uv run python -c "
from exo.shared.device_detection import get_device_info
info = get_device_info()
print(f'Backend: {info.backend}')
print(f'GPU Available: {info.is_gpu}')
print(f'Device: {info.device_name}')
"
```

### Expected Output (CUDA)

```
Backend: cuda
GPU Available: True
Device: NVIDIA CUDA Device
```

### Expected Output (CPU Fallback)

```
Backend: cpu
GPU Available: False
Device: CPU
```

## Troubleshooting

### CUDA Not Detected

1. **Verify NVIDIA driver is installed:**
   ```bash
   nvidia-smi
   ```
   If this fails, install or update your NVIDIA driver.

2. **Check if mlx[cuda] is installed:**
   ```bash
   uv pip list | grep mlx
   ```
   You should see `mlx-cuda` in the list.

3. **Reinstall with CUDA:**
   ```bash
   uv sync --extra cuda --reinstall
   ```

### WSL2 Specific Issues

1. **"No GPU detected in WSL":**
   - Ensure you installed the Windows NVIDIA driver (not Linux driver inside WSL)
   - Update WSL: `wsl --update`
   - Restart WSL: `wsl --shutdown` then reopen

2. **Permission errors:**
   - Add your user to the `video` group: `sudo usermod -aG video $USER`
   - Log out and back in

### Performance Issues

1. **Slow inference:**
   - Check GPU utilization: `nvidia-smi`
   - Ensure model fits in GPU memory
   - Consider using quantized models

2. **Out of memory:**
   - Use smaller models or quantized versions
   - Free GPU memory: `nvidia-smi --gpu-reset` (use with caution)

## CUDA Version Selection

exo supports multiple CUDA versions through optional dependencies:

```bash
# Auto-detect CUDA version (recommended)
uv sync --extra cuda

# Specific CUDA 12 version
uv sync --extra cuda12

# Specific CUDA 13 version
uv sync --extra cuda13
```

## Performance Tips

1. **Use quantized models** for better memory efficiency
2. **Monitor GPU memory** with `nvidia-smi` or `watch -n 1 nvidia-smi`
3. **Close other GPU applications** before running large models
4. **WSL2 overhead**: Expect ~5-10% performance overhead compared to native Linux

## Known Limitations

1. **Some MLX operations** may not be fully implemented for CUDA
2. **Multi-GPU support** (NCCL) is planned but not yet available
3. **WSL2 performance** is slightly lower than native Linux

## Getting Help

- [GitHub Issues](https://github.com/exo-explore/exo/issues) - Report bugs or request features
- [Discord](https://discord.gg/TJ4P57arEm) - Community support
- [MLX CUDA Discussion](https://github.com/ml-explore/mlx/discussions/2422) - MLX-specific CUDA issues
