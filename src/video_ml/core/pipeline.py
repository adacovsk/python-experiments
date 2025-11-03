"""Video processing pipeline that can apply multiple operations in sequence."""

import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from video_ml.core.video_utils import extract_audio, merge_audio_video


class VideoPipeline:
    """Process videos with any combination of denoise, enhance, and sharpen operations."""

    def __init__(self):
        """Initialize the video pipeline."""
        self.processors = []

    def add_enhancer(self, weights_path, device="cpu"):
        """Add enhancement operation to the pipeline."""
        from video_ml.core.enhancer import ImageEnhancer

        enhancer = ImageEnhancer(weights_path=weights_path, device=device)
        self.processors.append(("enhance", enhancer))
        return self

    def process_frame(self, frame):
        """
        Apply all processors to a frame in sequence.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Processed frame (BGR format)
        """
        processed = frame

        for name, processor in self.processors:
            if name == "enhance":
                # Enhancement: BGR->RGB->PIL->process->RGB->BGR
                frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                enhanced_pil, _ = processor._process_pil_image(frame_pil)
                enhanced_rgb = np.array(enhanced_pil)
                processed = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

        return processed

    def process_video(self, input_video, output_video, preserve_audio=True):
        """
        Process video with configured operations.

        Args:
            input_video: Path to input video
            output_video: Path to output video
            preserve_audio: Whether to preserve audio
        """
        if not self.processors:
            raise ValueError("No processors configured. Add at least one operation.")

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

        operations = [name for name, _ in self.processors]
        print(f"\nProcessing video with: {' + '.join(operations)}")
        print(f"Input resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")

        # Process first frame to get output dimensions
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame")

        processed_frame = self.process_frame(first_frame)
        out_height, out_width = processed_frame.shape[:2]
        print(f"Output resolution: {out_width}x{out_height}")

        # Use temporary directory for frame storage
        temp_dir = tempfile.mkdtemp(prefix="video_ml_")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        try:
            # Save all processed frames as images
            print("Processing and saving frames...")
            cv2.imwrite(os.path.join(frames_dir, "frame_000000.jpg"), processed_frame)

            # Reset video to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

            # Process remaining frames
            frame_idx = 1
            with tqdm(total=total_frames - 1, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame = self.process_frame(frame)
                    cv2.imwrite(
                        os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg"),
                        processed_frame
                    )
                    frame_idx += 1
                    pbar.update(1)

            cap.release()

            # Use ffmpeg to create video from frames
            print("Encoding video with ffmpeg...")
            temp_video = output_video.replace(".mp4", "_temp.mp4") if audio_file else output_video

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, "frame_%06d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # High quality
                "-preset", "medium",
                temp_video
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            # Merge audio back if needed
            if audio_file and os.path.exists(audio_file):
                print("Merging audio...")
                merge_audio_video(temp_video, audio_file, output_video)
                os.remove(temp_video)
                os.remove(audio_file)

            print(f"Video saved to {output_video}")

        finally:
            # Clean up temporary frames
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @classmethod
    def from_config(cls, config, device="cpu"):
        """
        Create a pipeline from configuration dictionary.

        Args:
            config: Video processing configuration dict
            device: Device to use ('cpu' or 'cuda')

        Returns:
            Configured VideoPipeline instance
        """
        pipeline = cls()
        print("Loading enhancement model...")
        pipeline.add_enhancer(
            weights_path=config["weights_path"],
            device=device
        )
        return pipeline
