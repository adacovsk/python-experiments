import gc
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel//2, out_channels=dw_channel//2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = []
        if dec_blk_nums is None:
            dec_blk_nums = []

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.ending = nn.Conv2d(in_channels=chan, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs, strict=False):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1], strict=False):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (8 - h % 8) % 8
        mod_pad_w = (8 - w % 8) % 8
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

class VideoProcessor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu', tile_size=None, tile_overlap=32):
        self.device = device
        print(f"VideoProcessor using device: {self.device}")

        # Adaptive tile size based on device
        if tile_size is None:
            # Larger tiles for GPU, smaller for CPU to manage memory
            self.tile_size = 512 if device == 'cuda' else 256
        else:
            self.tile_size = tile_size

        self.tile_overlap = tile_overlap  # Overlap between tiles for smooth blending
        print(f"Using tile size: {self.tile_size}x{self.tile_size} with overlap: {self.tile_overlap}")

        # Initialize NAFNet model for GoPro-width32
        # Architecture matches official NAFNet-GoPro-width32.pth
        self.model = NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28],
            dec_blk_nums=[1, 1, 1, 1]
        ).to(self.device)

        # Load pretrained weights
        # Note: weights_only=False is used because model files may contain optimizer state
        # Only load models from trusted sources (official repositories)
        weights = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'params' in weights:
            weights = weights['params']
        self.model.load_state_dict(weights)
        self.model.eval()

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    def process_tile(self, tile):
        with torch.no_grad():
            return self.model(tile)

    def pre_process(self, img):
        img = torch.from_numpy(img).float()
        img = img / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def post_process(self, img):
        img = img.squeeze(0).permute(1, 2, 0)
        img = img * 255.0
        img = img.cpu().numpy().clip(0, 255).astype(np.uint8)
        return img

    def split_tiles(self, img, tile_size, tile_overlap):
        """Split image into overlapping tiles for processing."""
        _, _, h, w = img.shape
        tiles = []
        positions = []

        # Calculate stride (tile size minus overlap)
        stride = tile_size - tile_overlap

        # Generate tile positions with overlap
        y_positions = list(range(0, h - tile_size + 1, stride))
        if not y_positions or y_positions[-1] + tile_size < h:
            y_positions.append(h - tile_size)

        x_positions = list(range(0, w - tile_size + 1, stride))
        if not x_positions or x_positions[-1] + tile_size < w:
            x_positions.append(w - tile_size)

        for y in y_positions:
            for x in x_positions:
                # Extract tile
                tile = img[:, :, y:y+tile_size, x:x+tile_size]
                tiles.append(tile)

                # Store position info (y_start, x_start)
                positions.append((y, x))

        return tiles, positions

    def merge_tiles(self, tiles, positions, img_shape, tile_size, tile_overlap):
        """Merge overlapping tiles with weighted blending for smooth transitions."""
        output = torch.zeros(img_shape, device=self.device)
        weight_sum = torch.zeros(img_shape, device=self.device)

        # Create a weight mask for blending
        # Higher weights in the center, lower at edges
        weight_mask = self.create_weight_mask(tile_size, tile_overlap)

        for tile, (y, x) in zip(tiles, positions, strict=False):
            # Place tile with weights
            output[:, :, y:y+tile_size, x:x+tile_size] += tile * weight_mask
            weight_sum[:, :, y:y+tile_size, x:x+tile_size] += weight_mask

        # Normalize by weight sum
        output = torch.where(weight_sum > 0, output / weight_sum, output)
        return output

    def create_weight_mask(self, tile_size, overlap):
        """Create a weight mask for tile blending with smooth transitions at borders."""
        weight = torch.ones((1, 1, tile_size, tile_size), device=self.device)

        if overlap > 0:
            # Create linear ramp for blending at edges
            ramp = torch.linspace(0, 1, overlap, device=self.device)

            # Apply ramp to all four edges
            # Top edge
            for i in range(overlap):
                weight[:, :, i, :] *= ramp[i]
            # Bottom edge
            for i in range(overlap):
                weight[:, :, tile_size-overlap+i, :] *= ramp[overlap-1-i]
            # Left edge
            for i in range(overlap):
                weight[:, :, :, i] *= ramp[i]
            # Right edge
            for i in range(overlap):
                weight[:, :, :, tile_size-overlap+i] *= ramp[overlap-1-i]

        return weight

    def process_frame(self, frame):
        # Pre-process
        img = self.pre_process(frame)
        _, _, h, w = img.shape

        # For small images or if tile size is larger than image, process without tiling
        if h <= self.tile_size and w <= self.tile_size:
            with torch.no_grad():
                # Pad to multiple of 8 for NAFNet
                img_padded = self.model.check_image_size(img.to(self.device))
                processed = self.model(img_padded)
                # Crop back to original size
                processed = processed[:, :, :h, :w]
            output = self.post_process(processed)
        else:
            # Split into tiles for larger images
            tiles, positions = self.split_tiles(img, self.tile_size, self.tile_overlap)

            # Process tiles
            processed_tiles = []
            for tile in tiles:
                processed_tile = self.process_tile(tile)
                processed_tiles.append(processed_tile)
                del tile  # Free memory
                gc.collect()

            # Merge tiles with weighted blending
            merged = self.merge_tiles(processed_tiles, positions, img.shape, self.tile_size, self.tile_overlap)

            # Post-process
            output = self.post_process(merged)

            # Clear memory
            del processed_tiles, merged
            gc.collect()

        return output

def process_video(input_path, output_path, model_path, temp_folder='temp_frames', device='cuda' if torch.cuda.is_available() else 'cpu', save_frames=False, frames_folder=None):
    """
    Process video with denoising model.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        model_path: Path to model weights
        temp_folder: Folder for temporary files during processing
        device: Device to use ('cuda' or 'cpu')
        save_frames: If True, save processed frames for workflow chaining
        frames_folder: Folder to save processed frames (if save_frames=True)
    """
    os.makedirs(temp_folder, exist_ok=True)

    if save_frames:
        if frames_folder is None:
            frames_folder = temp_folder + '_processed'
        os.makedirs(frames_folder, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize processor
    processor = VideoProcessor(model_path, device=device)

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save original frame
            temp_path = os.path.join(temp_folder, f'frame_{frame_idx:06d}.jpg')
            cv2.imwrite(temp_path, frame)

            # Process frame
            try:
                processed = processor.process_frame(frame)
                out.write(processed)

                # Save processed frame if requested
                if save_frames:
                    processed_path = os.path.join(frames_folder, f'denoised_{frame_idx:06d}.jpg')
                    cv2.imwrite(processed_path, processed)

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                out.write(frame)  # Write original frame if processing fails

            # Clean up temp frame (not the processed one)
            if os.path.exists(temp_path):
                os.remove(temp_path)

            frame_idx += 1
            print(f'Processed frame {frame_idx}/{total_frames}')

            # Force garbage collection
            gc.collect()

    finally:
        cap.release()
        out.release()

        try:
            os.rmdir(temp_folder)
        except OSError:
            pass

        print(f"Video processing complete. Output saved to {output_path}")
        if save_frames:
            print(f"Processed frames saved to {frames_folder}")

