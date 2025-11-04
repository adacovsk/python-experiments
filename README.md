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
