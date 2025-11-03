"""Main controller script for video_ml - handles both image and video processing."""

from utils import get_device, load_config
from video_ml.core.enhancer import ImageEnhancer
from video_ml.core.pipeline import VideoPipeline


def main():
    config = load_config()
    device = get_device(config)

    # Video processing mode
    if "video_processing" in config:
        video_config = config["video_processing"]
        pipeline = VideoPipeline.from_config(video_config, device)
        pipeline.process_video(
            input_video=video_config["input_video"],
            output_video=video_config["output_video"],
            preserve_audio=video_config.get("preserve_audio", True)
        )
        return

    # Image enhancement mode
    if "image_enhancer" in config:
        cfg = config["image_enhancer"]
        enhancer = ImageEnhancer(weights_path=cfg["weights_path"], device=device)
        enhancer.batch_process(
            input_dir=cfg["input_dir"],
            output_dir=cfg["output_dir"],
            batch_size=cfg.get("batch_size", 1)
        )
        return

    print("Error: No valid configuration found in config.toml")


if __name__ == "__main__":
    main()
