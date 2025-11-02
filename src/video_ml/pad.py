import cv2
import os
import sys
import numpy as np

from utils.config_loader import load_config

# Load configuration
config = load_config()
pad_config = config["pad"]
input_folder = pad_config["input_folder"]
output_video = pad_config["output_video"]
frame_rate = pad_config["frame_rate"]
target_width = pad_config["target_width"]
target_height = pad_config["target_height"]

if not os.path.exists(input_folder):
    print(f"Error: Input folder does not exist: {input_folder}")
    print("Please update the path in config.toml")
    sys.exit(1)

def pad_to_720p(frame):
    """Pads a frame to 1280x720 if needed, centering the original content."""
    h, w = frame.shape[:2]
    if w == target_width and h == target_height:
        return frame  # No padding needed
    
    # Calculate padding needed
    top = (target_height - h) // 2 if h < target_height else 0
    bottom = target_height - h - top if h < target_height else 0
    left = (target_width - w) // 2 if w < target_width else 0
    right = target_width - w - left if w < target_width else 0

    # Add padding with black color
    color = [0, 0, 0]  # Black padding
    padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return padded_frame

# Pad each frame and overwrite in the same folder
frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

for filename in frame_files:
    file_path = os.path.join(input_folder, filename)
    frame = cv2.imread(file_path)
    
    if frame is None:
        print(f"Warning: Could not read {file_path}. Skipping this frame.")
        continue  # Skip this frame if it couldn't be loaded
    
    # Pad the frame to 720p if necessary
    padded_frame = pad_to_720p(frame)
    
    # Overwrite the original frame with the padded frame
    cv2.imwrite(file_path, padded_frame)

# Now create a video from the padded frames
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 output
out = None

for filename in frame_files:
    frame_path = os.path.join(input_folder, filename)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Warning: Could not read {frame_path} during video creation. Skipping this frame.")
        continue  # Skip this frame if it couldn't be loaded
    
    # Initialize VideoWriter with frame dimensions if not done
    if out is None:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    
    # Write the frame to the video
    out.write(frame)

# Release the VideoWriter to save the file
if out is not None:
    out.release()
    print("Video created successfully:", output_video)
else:
    print("No valid frames found. Video was not created.")
