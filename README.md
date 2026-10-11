# Python Experiments

Collection of Python experiments, benchmarks, and ML tools.

## 📂 Projects

### 🎬 [video_ml/](src/video_ml/)
AI-powered video and image enhancement toolkit with flexible pipeline support:

**Core Modules** (`core/`):
- `enhancer.py` - RealESRGAN for 4x super-resolution (images & videos)
- `pipeline.py` - VideoPipeline for video processing
- `video_utils.py` - Audio extraction/merging utilities

**Examples** (`examples/`):
- `main.py` - Single entry point for all image and video processing
- `config.toml` - Configuration file for customizing processing

**Key Features**:
- 4x super-resolution enhancement for images and videos
- Automatic audio preservation for videos
- Configuration-driven workflow
- GPU/CPU auto-detection with configurable device selection
- ffmpeg-based video encoding for maximum compatibility

**Model**: Pre-trained PyTorch model (64 MB)
- RealESRGAN_x4plus.pth - 4x super-resolution enhancement

### 🎨 [icon_gen/](src/icon_gen/)
An icon for every ability in a [bevy-rpg](https://github.com/adacovsk/bevy-rpg) checkout, rendered locally on the CPU with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and composed from each record's own data:
- **Glyph** - an AI render of a light emblem on black, prompted from the ability's name and description
- **Compose** - frame shape by ability type, glyph tinted by damage type / harm-heal-help / tradition, vector badges for action cost and area shape (`palette.toml`)
- **Uniqueness** - icons too alike in shape *and* colour get the later key's glyph re-rolled
- **Coverage** - reports abilities without an icon and icons whose key no longer exists

```bash
pixi run icons-setup                 # pinned sd binaries + weights (~15 GB, gitignored)
pixi run icons --sample 20           # a spread across ability types, for review
pixi run icons --only fireball,heal  # specific keys
pixi run icons --all                 # everything; resumable, glyphs are cached
pixi run icons --report              # coverage and collisions only
```
Options: `--model turbo|flux`, `--icon-size` (default 128: sharp up to 250% UI scale), `--export <dir>`.

**Models**: SD-Turbo (Stability Community License) · FLUX.1 [schnell] Q4_0 (Apache 2.0) · TAESD/TAEF1 decoders

### 📊 [benchmarks/](src/benchmarks/)
Performance benchmarks and comparisons:
- Pandas vs Polars on 10M rows
- Vectorized vs non-vectorized operations

### 🔧 [utils/](src/utils/)
Utility scripts:
- **examples/** - HEIC conversion, batch renaming tools

### 🧪 [simulation/](src/simulation/)
Statistical simulations:
- **examples/** - Ensemble smoothing experiments

## 🚀 Quick Start

```bash
# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Install package with dependencies
pip install -e .

# Download model (optional - only needed for enhancement)
bash src/video_ml/download_models.sh

# Run processing (configure in config.toml)
python -m video_ml.examples.main

# Or import core classes directly
python -c "from video_ml.core import ImageEnhancer, VideoPipeline; print('Loaded successfully')"
```

## ⚙️ Configuration

Edit `src/video_ml/examples/config.toml` to customize processing:

```toml
# Device configuration
[device]
device = "auto"  # Options: "cuda", "cpu", or "auto" (auto-detect GPU)

# Video enhancement configuration
[video_processing]
input_video = "src/video_ml/examples/test_inputs/test_video.mp4"
output_video = "src/video_ml/examples/test_outputs/enhanced_video.mp4"
weights_path = "model_weights/RealESRGAN_x4plus.pth"
preserve_audio = true

# Image enhancement configuration
[image_enhancer]
input_dir = "src/video_ml/examples/test_inputs"
output_dir = "src/video_ml/examples/test_outputs"
weights_path = "model_weights/RealESRGAN_x4plus.pth"
batch_size = 1
```

## 📦 Dependencies

Core packages:
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision
- **Pandas/Polars** - Data analysis
- **Pillow** - Image processing (including HEIF support)
- **ffmpeg** - Video encoding (system dependency)

See `pyproject.toml` for complete list.

### Development dependencies
```bash
pip install -e .[dev]  # Includes pytest, black, ruff, mypy
```

## 🎯 GPU Support

For NVIDIA GPU acceleration, install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
