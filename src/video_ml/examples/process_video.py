"""Example script demonstrating full video processing pipeline."""

import sys
import torch
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from video_ml.core.video_processor import EnhancedVideoProcessor


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
    processor_config = config["video_processor"]

    try:
        processor = EnhancedVideoProcessor(
            sr_weights_path=processor_config["sr_weights_path"],
            interpolation_weights_path=processor_config["interpolation_weights_path"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        processor.enhance_video(
            input_path=processor_config["input_video"],
            output_path=processor_config["output_video"],
            target_height=processor_config.get("target_height", 720),
            target_fps=processor_config.get("target_fps", 60)
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
