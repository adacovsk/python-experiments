"""General utility scripts and tools."""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

__version__ = "0.1.0"


def load_config(config_path=None):
    """Load configuration from config.toml.

    Args:
        config_path: Optional path to config file. If not provided, looks for
                    config.toml in the video_ml/examples directory.

    Returns:
        dict: Configuration dictionary from config.toml
    """
    if config_path is None:
        # Default to video_ml/examples/config.toml
        config_path = Path(__file__).parent.parent / "video_ml" / "examples" / "config.toml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"Error: config.toml not found at {config_path}")
        print("Please create config.toml in the examples directory.")
        sys.exit(1)

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_device(config):
    """Get device based on config setting.

    Args:
        config: Configuration dictionary

    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    import torch

    device_setting = config.get("device", {}).get("device", "auto")

    if device_setting == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_setting

    return device
