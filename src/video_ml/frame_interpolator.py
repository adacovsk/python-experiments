import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import shutil
import glob
import subprocess
from tqdm import tqdm

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Adjusting the first layer to accept 960x720 input
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = self.res1(out)
        out = self.res2(out)
        return out

class FrameInterpolator(nn.Module):
    def __init__(self):
        super(FrameInterpolator, self).__init__()
        self.feature_extractor = FeatureExtractor()
        
        # Flow estimation network
        self.flow_net = nn.Sequential(
            nn.Conv2d(256, 192, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(192),
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 4, 3, 1, 1)  # Output 2 flows (forward and backward)
        )
        
        # Synthesis network with skip connections
        self.synth_net = nn.Sequential(
            nn.Conv2d(14, 128, 3, 1, 1),  # Increased input channels for synthesis
            nn.LeakyReLU(0.2, True),
            ResBlock(128),
            ResBlock(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()  # Keep Sigmoid to ensure output is in [0, 1]
        )

    def forward(self, frame1, frame2):
        feat1 = self.feature_extractor(frame1)
        feat2 = self.feature_extractor(frame2)

        flow_input = torch.cat([feat1, feat2], dim=1)
        flows = self.flow_net(flow_input)
        flow_f, flow_b = flows.chunk(2, dim=1)

        # Upscale flows to match input frame size
        flow_f = F.interpolate(flow_f, size=(frame1.size(2), frame1.size(3)), mode='bilinear', align_corners=False)
        flow_b = F.interpolate(flow_b, size=(frame1.size(2), frame1.size(3)), mode='bilinear', align_corners=False)

        warped1 = self._warp(frame1, flow_f)
        warped2 = self._warp(frame2, flow_b)

        synth_input = torch.cat([warped1, warped2, flow_f, flow_b, frame1, frame2], dim=1)
        intermediate = self.synth_net(synth_input)

        return intermediate

    def _warp(self, x, flow):
        B, C, H, W = x.size()

        # Create normalized device coordinates grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        # Add flow to grid
        flow_grid = grid + flow.clone()

        # Normalize flow grid to [-1, 1]
        flow_grid = flow_grid.permute(0, 2, 3, 1)

        # Sample using grid
        warped = F.grid_sample(x, flow_grid, mode='bilinear', padding_mode='border', align_corners=False)
        return warped

class VideoFrameInterpolator:
    def __init__(self, weights_path=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.interpolator = FrameInterpolator().to(self.device)
        if weights_path and Path(weights_path).exists():
            self.load_weights(weights_path)
        self.interpolator.eval()

    def load_weights(self, weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.interpolator.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Failed to load weights: {str(e)}")
            
    def pad_to_target_size(self, frame, target_size):
        """Pad the frame to the target size while maintaining the aspect ratio."""
        h, w, _ = frame.shape
        target_h, target_w = target_size
        delta_w = target_w - w
        delta_h = target_h - h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]  # Padding color (black)
        return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    def interpolate_frame(self, frame1, frame2):
        """Generate intermediate frame between two frames at fixed 720p resolution."""
        target_H, target_W = 720, 1280
        
        # Convert frames to float32 for better precision
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)

        # Pad frames to target size while maintaining aspect ratio
        frame1_padded = self.pad_to_target_size(frame1, (target_H, target_W))
        frame2_padded = self.pad_to_target_size(frame2, (target_H, target_W))
        
        # Normalize and convert to RGB
        frame1_rgb = cv2.cvtColor(frame1_padded, cv2.COLOR_BGR2RGB) / 255.0
        frame2_rgb = cv2.cvtColor(frame2_padded, cv2.COLOR_BGR2RGB) / 255.0
        
        # Convert to tensors with proper dimensions
        frame1_tensor = torch.from_numpy(frame1_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        frame2_tensor = torch.from_numpy(frame2_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            try:
                intermediate = self.interpolator(frame1_tensor, frame2_tensor)
                
                # Convert back to image
                intermediate = intermediate.clamp(0, 1)
                intermediate_array = (intermediate.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                intermediate_bgr = cv2.cvtColor(intermediate_array, cv2.COLOR_RGB2BGR)
                
                return intermediate_bgr
            except RuntimeError as e:
                print(f"Error during frame interpolation: {str(e)}")
                # Return a simple blend as fallback
                return cv2.addWeighted(frame1_padded.astype(np.uint8), 0.5, 
                                     frame2_padded.astype(np.uint8), 0.5, 0)

    def interpolate_video(self, input_path, output_path, target_fps=None, interpolation_factor=2):
        """Interpolate video frames to increase frame rate"""
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {input_path}")
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if target_fps:
                interpolation_factor = max(1, int(target_fps / original_fps))
            final_fps = original_fps * interpolation_factor
            
            print(f"\nInterpolating video:")
            print(f"Original FPS: {original_fps:.2f}")
            print(f"Target FPS: {final_fps:.2f}")
            print(f"Interpolation factor: {interpolation_factor}x")
            
            frame_count = 0
            previous_frame = None
            
            with tqdm(total=total_frames * interpolation_factor, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if previous_frame is not None:
                        # Generate intermediate frames
                        for i in range(interpolation_factor - 1):
                            intermediate = self.interpolate_frame(previous_frame, frame)
                            cv2.imwrite(str(temp_dir / f"frame_{frame_count:06d}.png"), intermediate)
                            frame_count += 1
                            pbar.update(1)
                    
                    # Save original frame
                    cv2.imwrite(str(temp_dir / f"frame_{frame_count:06d}.png"), frame)
                    frame_count += 1
                    pbar.update(1)
                    
                    previous_frame = frame.copy()
                    
                    # Memory cleanup
                    if frame_count % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            cap.release()
            
            # Create final video
            print("\nCreating final video...")
            self._create_video_from_frames(temp_dir, output_path, final_fps)
            
        except Exception as e:
            print(f"Error during video interpolation: {str(e)}")
            raise
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _create_video_from_frames(self, temp_dir, output_path, fps):
        """Create video from frames using FFmpeg"""
        command = [
            'ffmpeg',
            '-framerate', f'{fps:.2f}',
            '-i', f'{temp_dir}/frame_%06d.png',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            result.check_returncode()
            print("Video creation successful")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            raise

def main():
    try:
        interpolator = VideoFrameInterpolator(
            weights_path="interpolator_weights.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        interpolator.interpolate_video(
            input_path="20130809_023600.mp4",
            output_path="interpolated_output.mp4",
            target_fps=30  # Or specify your desired target FPS
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
