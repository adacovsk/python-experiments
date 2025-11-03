import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block from ESRGAN."""
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch))
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = feat.clone()
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

def pixel_unshuffle(x, scale):
    b, c, h, w = x.size()
    h //= scale
    w //= scale
    out_channel = c * (scale ** 2)
    out = x.view(b, c, h, scale, w, scale)
    out = out.permute(0, 1, 3, 5, 2, 4).contiguous()
    out = out.view(b, out_channel, h, w)
    return out

class ImageEnhancer:
    def __init__(self, weights_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        ).to(device)
        self.model.eval()

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            print("No weights provided or file not found. Using initialized weights.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load_weights(self, weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            self.model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")

    def enhance_image(self, image_path, output_path=None):
        try:
            img = Image.open(image_path).convert('RGB')
            output_image, success = self._process_pil_image(img)

            if success and output_path:
                output_image.save(output_path)

            return output_image, success
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, False

    def batch_process(self, input_dir, output_dir, batch_size=1):
        """
        Process multiple images in batches

        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save enhanced images
            batch_size (int): Number of images to process simultaneously
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get list of supported image files
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in supported_formats]

        if not image_files:
            print(f"No supported images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        # Process images in batches
        total_time = 0
        successful = 0
        failed = 0

        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_files = image_files[i:i + batch_size]
            batch_start_time = time.time()

            for img_path in batch_files:
                output_path = output_dir / f"enhanced_{img_path.name}"

                _, success = self.enhance_image(str(img_path), str(output_path))
                if success:
                    successful += 1
                else:
                    failed += 1

            batch_time = time.time() - batch_start_time
            total_time += batch_time

            # Free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_time = total_time / len(image_files)
        print("\nProcessing complete!")
        print(f"Total images processed: {len(image_files)}")
        print(f"Successful: {successful}, Failed: {failed}")
        print(f"Average processing time per image: {avg_time:.2f} seconds")
        print(f"Total processing time: {total_time:.2f} seconds")

    def enhance_video(self, input_video, output_video, preserve_audio=True):
        """
        Enhance video by processing each frame (DRY approach reusing enhance_image logic).

        Args:
            input_video (str): Path to input video file
            output_video (str): Path to output video file
            preserve_audio (bool): Whether to preserve audio from original video
        """
        import cv2

        from video_ml.core.video_utils import extract_audio, merge_audio_video

        # Extract audio if needed
        audio_file = None
        if preserve_audio:
            audio_file = "temp_audio.aac"
            has_audio = extract_audio(input_video, audio_file)
            if not has_audio:
                audio_file = None

        # Open video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("\nProcessing video:")
        print(f"Input resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")

        # Process first frame to get output dimensions
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame")

        # Convert BGR to RGB, process, and convert back
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        enhanced_pil, _ = self._process_pil_image(frame_pil)
        enhanced_rgb = np.array(enhanced_pil)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

        out_height, out_width = enhanced_bgr.shape[:2]
        print(f"Output resolution: {out_width}x{out_height} (4x upscale)")

        # Create temporary output video (without audio)
        temp_output = output_video.replace('.mp4', '_temp.mp4') if audio_file else output_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (out_width, out_height))
        out.write(enhanced_bgr)

        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

        # Process remaining frames
        with tqdm(total=total_frames-1, desc="Enhancing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR->RGB->PIL->process->RGB->BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                enhanced_pil, _ = self._process_pil_image(frame_pil)
                enhanced_rgb = np.array(enhanced_pil)
                enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

                out.write(enhanced_bgr)
                pbar.update(1)

        cap.release()
        out.release()

        # Merge audio back if needed
        if audio_file and os.path.exists(audio_file):
            print("Merging audio...")
            merge_audio_video(temp_output, audio_file, output_video)
            os.remove(temp_output)
            os.remove(audio_file)

        print(f"Video saved to {output_video}")

    def _process_pil_image(self, pil_image):
        """Helper method to process a PIL image (DRY helper for both image and video)."""
        try:
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_tensor = self.model(input_tensor)

            output_tensor = output_tensor.clamp(0, 1)
            output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())

            return output_image, True
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, False

