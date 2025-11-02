import os
from PIL import Image
import pillow_heif

def heic_to_png(heic_file, png_file):
    # Open the HEIC file using pillow_heif
    heif_image = pillow_heif.open_heif(heic_file)
    
    # Convert to a PIL Image
    image = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)

    # Save as PNG
    image.save(png_file, "PNG")

def convert_folder_heic_to_png(folder_path):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".heic"):  # Case-insensitive check
            heic_file = os.path.join(folder_path, filename)
            png_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.png")
            heic_to_png(heic_file, png_file)
            print(f"Converted {filename} to PNG.")

# Example usage: change the path according to your system
folder_path = r"C:\Users\Adam\Desktop\Print pictures for Dahlia"  # or use forward slashes
convert_folder_heic_to_png(folder_path)
