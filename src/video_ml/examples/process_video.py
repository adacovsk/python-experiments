"""Unified video processing script - supports combining denoise, enhance, and sharpen."""

import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import get_device, load_config
from video_ml.core.denoiser import VideoProcessor
from video_ml.core.enhancer import ImageEnhancer
from video_ml.core.sharpener import Sharpener
from video_ml.core.video_utils import extract_audio, merge_audio_video


def process_frame(frame, processors):
    """
    Apply multiple processors to a frame in sequence.

    Args:
        frame: Input frame (BGR format)
        processors: List of (name, processor) tuples

    Returns:
        Processed frame (BGR format)
    """
    processed = frame

    for name, processor in processors:
        if name == "denoise":
            processed = processor.process_frame(processed)
        elif name == "enhance":
            # Enhancement: BGR->RGB->PIL->process->RGB->BGR
            frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            enhanced_pil, _ = processor._process_pil_image(frame_pil)
            enhanced_rgb = np.array(enhanced_pil)
            processed = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        elif name == "sharpen":
            processed = processor.sharpen_frame(processed)

    return processed


def process_video(input_video, output_video, processors, preserve_audio=True):
    """
    Process video with multiple operations in sequence.

    Args:
        input_video: Path to input video
        output_video: Path to output video
        processors: List of (name, processor) tuples to apply in order
        preserve_audio: Whether to preserve audio
    """
    # Extract audio if needed
    audio_file = None
    if preserve_audio:
        audio_file = "temp_audio.aac"
        has_audio = extract_audio(input_video, audio_file)
        if not has_audio:
            audio_file = None

    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    operations = [name for name, _ in processors]
    print(f"\nProcessing video with: {' + '.join(operations)}")
    print(f"Input resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Process first frame to get output dimensions
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    processed_frame = process_frame(first_frame, processors)
    out_height, out_width = processed_frame.shape[:2]
    print(f"Output resolution: {out_width}x{out_height}")

    # Create temporary output video (without audio)
    temp_output = output_video.replace('.mp4', '_temp.mp4') if audio_file else output_video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (out_width, out_height))
    out.write(processed_frame)

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    # Process remaining frames
    with tqdm(total=total_frames-1, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, processors)
            out.write(processed_frame)
            pbar.update(1)

    cap.release()
    out.release()

    # Merge audio back if needed
    if audio_file and os.path.exists(audio_file):
        print("Merging audio...")
        merge_audio_video(temp_output, audio_file, output_video)
        os.remove(temp_output)
        os.remove(audio_file)

    print(f"Video saved to {output_video}")


def main():
    config = load_config()
    video_config = config.get("video_processing", {})
    device = get_device(config)

    # Check which operations are enabled
    enable_denoise = video_config.get("enable_denoise", False)
    enable_enhance = video_config.get("enable_enhance", False)
    enable_sharpen = video_config.get("enable_sharpen", False)

    if not any([enable_denoise, enable_enhance, enable_sharpen]):
        print("Error: No operations enabled in config.toml")
        print("Set at least one of: enable_denoise, enable_enhance, enable_sharpen to true")
        return

    # Build processing pipeline
    processors = []

    if enable_denoise:
        print("Loading denoising model...")
        denoiser = VideoProcessor(
            model_path=video_config["denoise_model_path"],
            device=device
        )
        processors.append(("denoise", denoiser))

    if enable_enhance:
        print("Loading enhancement model...")
        enhancer = ImageEnhancer(
            weights_path=video_config["enhance_weights_path"],
            device=device
        )
        processors.append(("enhance", enhancer))

    if enable_sharpen:
        print("Initializing sharpener...")
        strength = video_config.get("sharpen_strength", 1.0)
        sharpener = Sharpener(strength=strength)
        processors.append(("sharpen", sharpener))

    process_video(
        input_video=video_config["input_video"],
        output_video=video_config["output_video"],
        processors=processors,
        preserve_audio=video_config.get("preserve_audio", True)
    )


if __name__ == "__main__":
    main()
