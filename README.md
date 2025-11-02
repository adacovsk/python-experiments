# Python Experiments 🐍

Collection of Python experiments, benchmarks, and ML tools.

## 📂 Projects

### 🎬 [video_ml/](src/video_ml/)
AI-powered video and image enhancement tools:
- **core/** - Reusable backend classes (ImageEnhancer, VideoProcessor, etc.)
- **examples/** - Example scripts with config.toml
  - Video upscaling with RealESRGAN (4x resolution)
  - Image denoising with NAFNet
  - Frame interpolation for smooth slow-motion
  - Batch processing with ffmpeg integration

**Models**: 3 pre-trained PyTorch models (137 MB total)

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

# Run benchmarks
python src/benchmarks/pandas_vs_polars.py

# Run video enhancement examples
python src/video_ml/examples/enhance_images.py

# Or import core classes directly
python -c "from video_ml.core import ImageEnhancer; print('Loaded successfully')"
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
