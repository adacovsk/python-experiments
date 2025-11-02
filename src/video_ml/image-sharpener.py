import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageSharpener:
    def __init__(self, strength=1.0):
        """
        Initialize the image sharpener
        Args:
            strength (float): Sharpening strength (default: 1.0)
        """
        self.strength = strength
        
        # Define sharpening kernel
        self.kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Add padding to maintain image size
        self.pad = nn.ReplicationPad2d(1)

    def sharpen(self, image_path="1234.jpg"):
        """
        Sharpen an image using a convolution kernel
        Args:
            image_path (str): Path to the input image
        Returns:
            PIL.Image: Sharpened image
        """
        # Load and prepare image
        img = Image.open(image_path).convert('RGB')
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Split into channels
        r, g, b = img_tensor.split(1, dim=1)
        
        # Apply sharpening to each channel
        r_sharp = self._apply_sharpening(r)
        g_sharp = self._apply_sharpening(g)
        b_sharp = self._apply_sharpening(b)
        
        # Combine channels
        sharpened = torch.cat([r_sharp, g_sharp, b_sharp], dim=1)
        
        # Clip values to valid range
        sharpened = torch.clamp(sharpened, 0, 1)
        
        # Convert back to PIL Image
        to_pil = transforms.ToPILImage()
        return to_pil(sharpened.squeeze(0))
    
    def _apply_sharpening(self, channel):
        """
        Apply sharpening to a single channel
        """
        # Pad the input
        padded = self.pad(channel)
        
        # Apply sharpening kernel
        sharpened = F.conv2d(padded, self.kernel)
        
        # Combine with original image
        return channel + (sharpened * self.strength)

    def __call__(self, image_path):
        """
        Make the class callable
        """
        return self.sharpen(image_path)

def main():
    # Create an instance of the sharpener
    sharpener = ImageSharpener()
    
    # Sharpen the image
    sharpened_image = sharpener.sharpen()
    
    # Save the result
    sharpened_image.save("sharpened_1234.jpg")

if __name__ == "__main__":
    main()
