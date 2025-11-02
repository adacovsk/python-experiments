"""Example script demonstrating video denoising with NAFNet."""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from video_ml.core.denoiser import process_video


def load_config():
    """Load configuration from config.toml in the examples directory."""
    config_path = Path(__file__).parent / "config.toml"

    if not config_path.exists():
        print(f"Error: config.toml not found at {config_path}")
        print("Please create config.toml in the examples directory.")
        sys.exit(1)

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main():
    config = load_config()
    denoiser_config = config["denoiser"]

    process_video(
        denoiser_config["input_video"],
        denoiser_config["output_video"],
        denoiser_config["model_path"],
        denoiser_config.get("temp_folder", "temp_frames")
    )


if __name__ == "__main__":
    main()
