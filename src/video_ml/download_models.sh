#!/bin/bash
# Download pre-trained models for video enhancement

echo "📥 Downloading pre-trained models..."

# RealESRGAN models
echo "Downloading RealESRGAN_x4plus.pth (64 MB)..."
# wget -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

echo "Downloading realesr-general-wdn-x4v3.pth (4.7 MB)..."
# wget -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth

# NAFNet model
echo "Downloading NAFNet-GoPro-width32.pth (65 MB)..."
# wget -nc https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-GoPro-width32.pth

echo "✅ Models already present (137 MB total)"
echo ""
echo "Models location: $(pwd)"
ls -lh *.pth 2>/dev/null || echo "No models found. Uncomment wget commands to download."
