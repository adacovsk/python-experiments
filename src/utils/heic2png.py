import os
import sys
from PIL import Image
import pillow_heif

from utils.config_loader import load_config


def heic_to_png(heic_file, png_file):
    """Convert a single HEIC file to PNG."""
    heif_image = pillow_heif.open_heif(heic_file)
    image = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
    image.save(png_file, "PNG")


def convert_folder_heic_to_png(folder_path):
    """Convert all HEIC files in a folder to PNG."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".heic"):
            heic_file = os.path.join(folder_path, filename)
            png_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.png")
            heic_to_png(heic_file, png_file)
            print(f"Converted {filename} to PNG.")


if __name__ == "__main__":
    config = load_config()
    folder_path = config["heic2png"]["input_folder"]

    if not os.path.exists(folder_path):
        print(f"Error: Input folder does not exist: {folder_path}")
        print("Please update the path in config.toml")
        sys.exit(1)

    convert_folder_heic_to_png(folder_path)
