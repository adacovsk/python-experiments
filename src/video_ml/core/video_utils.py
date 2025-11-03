"""Utility functions for video processing."""

import os
import subprocess

import cv2
from tqdm import tqdm


def pad_frame(frame, target_width, target_height, color=(0, 0, 0)):
    """Pad a frame to target dimensions, centering the original content.

    Args:
        frame: Input frame (numpy array)
        target_width: Target width in pixels
        target_height: Target height in pixels
        color: Padding color (B, G, R) tuple, default is black

    Returns:
        Padded frame
    """
    h, w = frame.shape[:2]

    if w == target_width and h == target_height:
        return frame  # No padding needed

    # Calculate padding needed
    top = (target_height - h) // 2 if h < target_height else 0
    bottom = target_height - h - top if h < target_height else 0
    left = (target_width - w) // 2 if w < target_width else 0
    right = target_width - w - left if w < target_width else 0

    # Add padding
    padded_frame = cv2.copyMakeBorder(
        frame, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded_frame


def frames_to_video(frames_folder, output_video, fps=30.0, frame_pattern=('*.jpg', '*.png'), pad_to=None):
    """Convert a folder of frames to a video file.

    Args:
        frames_folder: Path to folder containing frames
        output_video: Path to output video file
        fps: Frames per second for output video
        frame_pattern: Tuple of file patterns to match (e.g., ('*.jpg', '*.png'))
        pad_to: Optional tuple (width, height) to pad frames to

    Returns:
        bool: True if successful, False otherwise
    """
    # Get all frames matching the patterns
    import glob
    frames = []
    for pattern in frame_pattern:
        frames.extend(glob.glob(os.path.join(frames_folder, pattern)))
    frames = sorted(frames)

    if not frames:
        print(f"No frames found in {frames_folder}")
        return False

    # Read first frame to get dimensions
    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        print(f"Could not read first frame: {frames[0]}")
        return False

    # Determine output dimensions
    if pad_to:
        width, height = pad_to
    else:
        height, width = first_frame.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Converting {len(frames)} frames to video at {fps} FPS...")
    for frame_path in tqdm(frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read {frame_path}, skipping")
            continue

        # Pad if needed
        if pad_to:
            frame = pad_frame(frame, width, height)

        out.write(frame)

    out.release()
    print(f"Video saved to {output_video}")
    return True


def pad_frames_in_folder(input_folder, target_width, target_height, inplace=True, output_folder=None):
    """Pad all frames in a folder to target dimensions.

    Args:
        input_folder: Path to folder containing frames
        target_width: Target width in pixels
        target_height: Target height in pixels
        inplace: If True, overwrite original frames. If False, save to output_folder
        output_folder: Path to output folder (required if inplace=False)

    Returns:
        int: Number of frames processed
    """
    if not inplace and output_folder is None:
        raise ValueError("output_folder must be specified when inplace=False")

    if not inplace:
        os.makedirs(output_folder, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Padding {len(frame_files)} frames to {target_width}x{target_height}...")
    processed = 0

    for filename in tqdm(frame_files):
        input_path = os.path.join(input_folder, filename)
        frame = cv2.imread(input_path)

        if frame is None:
            print(f"Warning: Could not read {input_path}, skipping")
            continue

        # Pad the frame
        padded_frame = pad_frame(frame, target_width, target_height)

        # Save frame
        output_path = input_path if inplace else os.path.join(output_folder, filename)
        cv2.imwrite(output_path, padded_frame)
        processed += 1

    print(f"Processed {processed} frames")
    return processed


def extract_audio(video_path, audio_path):
    """Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to input video file
        audio_path: Path to save extracted audio (e.g., 'audio.aac' or 'audio.mp3')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file if exists
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'copy',  # Copy audio codec without re-encoding
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Audio extracted to {audio_path}")
            return True
        else:
            print("Warning: Could not extract audio (video may not have audio track)")
            return False
    except Exception as e:
        print(f"Warning: Audio extraction failed: {e}")
        return False


def merge_audio_video(video_path, audio_path, output_path):
    """Merge video with audio track using ffmpeg.

    Args:
        video_path: Path to video file (without audio or with audio to be replaced)
        audio_path: Path to audio file
        output_path: Path to save output video with audio

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file if exists
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video codec without re-encoding
            '-c:a', 'aac',  # Encode audio as AAC
            '-shortest',  # Finish encoding when shortest stream ends
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Audio merged successfully: {output_path}")
            return True
        else:
            print(f"Error merging audio: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error merging audio: {e}")
        return False
