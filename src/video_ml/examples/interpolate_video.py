"""Example script demonstrating frame interpolation."""

import sys
import torch
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from video_ml.core.interpolator import VideoFrameInterpolator


def load_config():
    """Load configuration from config.toml in the examples directory."""
    config_path = Path(__file__).parent / "config.toml"

    if not config_path.exists():
        print(f"Error: config.toml not found at {config_path}")
        sys.exit(1)

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main():
    config = load_config()
    interp_config = config["frame_interpolator"]

    try:
        interpolator = VideoFrameInterpolator(
            weights_path=interp_config["weights_path"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        interpolator.interpolate_video(
            input_path=interp_config["input_video"],
            output_path=interp_config["output_video"],
            target_fps=interp_config.get("target_fps", 60)
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
