"""Example script demonstrating image enhancement with RealESRGAN."""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from video_ml.core.enhancer import ImageEnhancer


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
    enhancer_config = config["image_enhancer"]

    # Initialize the enhancer
    enhancer = ImageEnhancer(weights_path=enhancer_config["weights_path"])

    # Process a batch of images
    enhancer.batch_process(
        input_dir=enhancer_config["input_dir"],
        output_dir=enhancer_config["output_dir"],
        batch_size=enhancer_config.get("batch_size", 1)
    )


if __name__ == "__main__":
    main()
