# Video & Image ML Processing Tools

AI-powered video and image enhancement using PyTorch and pre-trained models.

## 🎬 Scripts

### Video Processing
- **`video-processor.py`** - Main video processing pipeline
- **`video-enhancer-with-fps.py`** - Video enhancement with FPS control
- **`frame-interpolator.py`** - Frame interpolation for smooth slow-mo

### Image Enhancement
- **`pretrained-sharpener.py`** - Image sharpening using RealESRGAN
- **`image-sharpener.py`** - Custom image sharpening
- **`denoise.py`** - Image denoising using NAFNet
- **`ensemble.py`** - Ensemble multiple enhancement models

### Utilities
- **`heic2png.py`** - Convert HEIC images to PNG
- **`Rename.py`** - Batch file renaming
- **`pad.py`** - Image padding utilities

## 🤖 Pre-trained Models

- **RealESRGAN_x4plus.pth** (64 MB) - 4x upscaling model
- **realesr-general-wdn-x4v3.pth** (4.7 MB) - General purpose denoising
- **NAFNet-GoPro-width32.pth** (65 MB) - GoPro video deblurring

## 📁 Directories

- `enhanced_images/` - Output for enhanced images
- `input_images/` - Input images for processing
- `temp_frames/` - Temporary video frame extraction

## 🚀 Usage

```bash
# Sharpen image with RealESRGAN
python pretrained-sharpener.py input.jpg

# Denoise video
python denoise.py video.mp4

# Frame interpolation (2x slow-mo)
python frame-interpolator.py video.mp4 --factor 2
```

## 🔧 Requirements

See main `pyproject.toml` for dependencies. Key packages:
- PyTorch (with CUDA for GPU acceleration)
- OpenCV
- Pillow
- MoviePy

Install from project root:
```bash
pip install -e .
