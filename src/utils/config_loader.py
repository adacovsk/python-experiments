"""Configuration loader utility for python-experiments."""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config():
    """
    Load configuration from config.toml in the project root.

    Returns:
        dict: Parsed configuration dictionary

    Raises:
        SystemExit: If config.toml is not found
    """
    # Find project root (3 levels up from this file: src/utils/config_loader.py -> src/utils/ -> src/ -> root/)
    config_path = Path(__file__).parent.parent.parent / "config.toml"

    if not config_path.exists():
        print(f"Error: config.toml not found at {config_path}")
        print("Please create config.toml in the project root with your settings.")
        sys.exit(1)

    with open(config_path, "rb") as f:
        return tomllib.load(f)
