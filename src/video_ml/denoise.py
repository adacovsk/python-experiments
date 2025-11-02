import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from torch.nn.parallel import DataParallel

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel*2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel*2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        x = self.conv3(x)

        y = inp
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)

        return inp + x * self.beta + y * self.gamma

class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

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

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
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
    def __init__(self, model_path):
        self.device = 'cpu'
        self.tile_size = 256  # Process image in tiles to save memory
        self.tile_pad = 32    # Padding for each tile to avoid boundary artifacts
        
        # Initialize NAFNet model
        self.model = NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2]
        )
        
        # Load pretrained weights
        weights = torch.load(model_path, map_location='cpu')
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
        return img

    def post_process(self, img):
        img = img.squeeze(0).permute(1, 2, 0)
        img = img * 255.0
        img = img.cpu().numpy().clip(0, 255).astype(np.uint8)
        return img

    def split_tiles(self, img, tile_size, tile_pad):
        _, _, h, w = img.shape
        tiles = []
        positions = []
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Calculate actual tile size (handling edge cases)
                effective_y = min(y + tile_size, h)
                effective_x = min(x + tile_size, w)
                
                # Calculate padded coordinates
                pad_y1 = max(0, y - tile_pad)
                pad_y2 = min(h, effective_y + tile_pad)
                pad_x1 = max(0, x - tile_pad)
                pad_x2 = min(w, effective_x + tile_pad)
                
                # Extract and store tile with padding
                tile = img[:, :, pad_y1:pad_y2, pad_x1:pad_x2]
                tiles.append(tile)
                
                # Store position info for reconstruction
                pos = (y, effective_y, x, effective_x,  # Original coordinates
                      pad_y1, pad_y2, pad_x1, pad_x2,  # Padded coordinates
                      y - pad_y1, x - pad_x1)          # Offset within padded tile
                positions.append(pos)
        
        return tiles, positions

    def merge_tiles(self, tiles, positions, img_shape):
        output = torch.zeros(img_shape, device=self.device)
        count = torch.zeros(img_shape, device=self.device)
        
        for tile, pos in zip(tiles, positions):
            y1, y2, x1, x2, pad_y1, pad_y2, pad_x1, pad_x2, off_y, off_x = pos
            
            # Calculate the region in the tile that corresponds to the non-padded area
            tile_segment = tile[:, :, off_y:off_y + (y2-y1), off_x:off_x + (x2-x1)]
            
            # Add tile to output
            output[:, :, y1:y2, x1:x2] += tile_segment
            count[:, :, y1:y2, x1:x2] += 1
            
        # Average overlapping regions
        output = output / count
        return output

    def process_frame(self, frame):
        # Pre-process
        img = self.pre_process(frame)
        
        # Split into tiles
        tiles, positions = self.split_tiles(img, self.tile_size, self.tile_pad)
        
        # Process tiles
        processed_tiles = []
        for tile in tiles:
            processed_tile = self.process_tile(tile)
            processed_tiles.append(processed_tile)
            del tile  # Free memory
            gc.collect()
        
        # Merge tiles
        merged = self.merge_tiles(processed_tiles, positions, img.shape)
        
        # Post-process
        output = self.post_process(merged)
        
        # Clear memory
        del processed_tiles, merged
        gc.collect()
        
        return output

def process_video(input_path, output_path, model_path, temp_folder='temp_frames'):
    os.makedirs(temp_folder, exist_ok=True)
    
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
    processor = VideoProcessor(model_path)
    
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
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                out.write(frame)  # Write original frame if processing fails
            
            # Clean up
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
        except:
            pass
        
        print(f"Video processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    from utils.config_loader import load_config

    config = load_config()
    denoise_config = config["denoise"]

    process_video(
        denoise_config["input_video"],
        denoise_config["output_video"],
        denoise_config["model_path"],
        denoise_config.get("temp_folder", "temp_frames")
    )
