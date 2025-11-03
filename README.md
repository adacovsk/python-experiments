# Python Experiments 🐍

Collection of Python experiments, benchmarks, and ML tools.

## 📂 Projects

### 🎬 [video_ml/](src/video_ml/)
AI-powered video and image enhancement toolkit with flexible pipeline support:

**Core Modules** (`core/`):
- `enhancer.py` - RealESRGAN for 4x super-resolution (images & videos)
- `denoiser.py` - NAFNet for noise removal
- `sharpener.py` - Convolution-based sharpening (no model required)
- `video_utils.py` - Audio extraction/merging utilities

**Example Scripts** (`examples/`):
- `enhance_images.py` - Batch image enhancement
- `process_video.py` - Unified video processing pipeline

**Key Features**:
- Combine operations: denoise + enhance + sharpen in one pass
- Automatic audio preservation
- Configuration-driven workflow via `config.toml`
- DRY architecture with shared utilities

**Models**: 2 pre-trained PyTorch models (129 MB total)
- RealESRGAN_x4plus.pth (64 MB) - Enhancement
- NAFNet-GoPro-width32.pth (65 MB) - Denoising

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
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package with dependencies
pip install -e .

# Download models (optional - only needed for enhance/denoise)
bash src/video_ml/download_models.sh

# Run video processing (configure operations in config.toml)
python -m video_ml.examples.process_video

# Run image enhancement
python -m video_ml.examples.enhance_images

# Or import core classes directly
python -c "from video_ml.core import ImageEnhancer, Sharpener; print('Loaded successfully')"
```

## ⚙️ Configuration

Edit `src/video_ml/examples/config.toml` to customize processing:

```toml
[video_processing]
input_video = "test_inputs/test_video.mp4"
output_video = "test_outputs/processed_output.mp4"

# Enable any combination of operations
enable_denoise = false
enable_enhance = true
enable_sharpen = false

# Operation-specific settings
enhance_weights_path = "model_weights/RealESRGAN_x4plus.pth"
denoise_model_path = "model_weights/NAFNet-GoPro-width32.pth"
sharpen_strength = 1.0
```

## 📦 Dependencies

Core packages:
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision
- **Pandas/Polars** - Data analysis
- **MoviePy** - Video editing
- **Pillow** - Image processing

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
