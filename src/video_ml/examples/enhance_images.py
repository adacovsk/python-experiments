"""Example script demonstrating image enhancement with RealESRGAN."""

from utils import get_device, load_config
from video_ml.core.enhancer import ImageEnhancer


def main():
    config = load_config()
    enhancer_config = config["image_enhancer"]
    device = get_device(config)

    # Initialize the enhancer
    enhancer = ImageEnhancer(
        weights_path=enhancer_config["weights_path"],
        device=device
    )

    # Process a batch of images
    enhancer.batch_process(
        input_dir=enhancer_config["input_dir"],
        output_dir=enhancer_config["output_dir"],
        batch_size=enhancer_config.get("batch_size", 1)
    )


if __name__ == "__main__":
    main()
