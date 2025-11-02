import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import cv2
import subprocess
import math
from tqdm import tqdm
import time
import shutil
import glob

class ResidualDenseBlock(nn.Module):
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

class FrameInterpolator(nn.Module):
    def __init__(self):
        super(FrameInterpolator, self).__init__()
        # Flow estimation network
        self.flow_net = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 4, 3, 1, 1)  # 2 channels each for forward and backward flow
        )
        
        # Frame synthesis network
        self.synth_net = nn.Sequential(
            nn.Conv2d(10, 64, 3, 1, 1),  # 6 from warped frames + 4 from flow
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, frame1, frame2):
        # Concatenate frames
        input_frames = torch.cat([frame1, frame2], dim=1)
        
        # Estimate bidirectional flow
        flows = self.flow_net(input_frames)
        flow_f, flow_b = flows.chunk(2, dim=1)
        
        # Warp frames using estimated flow
        warped1 = self._warp(frame1, flow_f)
        warped2 = self._warp(frame2, flow_b)
        
        # Synthesize intermediate frame
        synth_input = torch.cat([warped1, warped2, flows], dim=1)
        intermediate = self.synth_net(synth_input)
        
        return intermediate
    
    def _warp(self, x, flow):
        B, C, H, W = x.size()
        # Create mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float().to(x.device)
        
        # Add flow to grid
        vgrid = grid + flow
        
        # Normalize grid values to [-1,1] for grid_sample
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
        
        # Reshape for grid_sample
        vgrid = vgrid.permute(0,2,3,1)
        
        # Perform warping
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border')
        return output

class VideoProcessor:
    def __init__(self, weights_path, device='cpu'):
        self.device = device
        print(f"Using device: {device}")
        
        # Initialize model
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        ).to(device)
        self.model.eval()
        
        # Load weights
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load_weights(self, weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            self.model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            raise Exception(f"Error loading weights: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Transform to tensor
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Convert back to image
        output_tensor = output_tensor.clamp(0, 1)
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
        
        # Convert back to BGR for OpenCV
        output_array = np.array(output_image)
        output_frame = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
        
        return output_frame

    def _convert_wmv_to_mp4(self, wmv_path, mp4_path):
        print("Converting WMV to MP4...")
        command = [
            'ffmpeg',
            '-i', wmv_path,
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'fast',
            '-y',
            mp4_path
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Conversion successful")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error during WMV conversion:\n{e.stderr}")
            raise

    def _verify_frames(self, temp_dir):
        """Verify frame sequence integrity"""
        frames = sorted(glob.glob(str(temp_dir / "frame_*.png")))
        if not frames:
            return False
            
        expected_count = int(os.path.basename(frames[-1]).split('_')[1].split('.')[0]) + 1
        actual_count = len(frames)
        
        if actual_count != expected_count:
            print(f"Warning: Frame count mismatch. Expected {expected_count}, found {actual_count}")
            return False
            
        return True

    def _create_video_from_frames(self, temp_dir, output_path, fps, target_height):
        """Create video from frames with enhanced error handling"""
        # Verify we have frames to process
        frame_pattern = f'{temp_dir}/frame_*.png'
        frames = sorted(glob.glob(frame_pattern))
        
        if not frames:
            raise ValueError(f"No frames found in {temp_dir}")
        
        print(f"Found {len(frames)} frames to process")
        
        # Build FFmpeg command
        command = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', f'{temp_dir}/frame_%06d.png',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-vf', f'scale=-1:{target_height}',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Video creation successful")
            return True
        except subprocess.CalledProcessError as e:
            print("\nFFmpeg Error Details:")
            print(f"Exit code: {e.returncode}")
            print("\nFFmpeg Output:")
            print(e.stdout)
            print("\nFFmpeg Error:")
            print(e.stderr)
            
            # Try alternative settings
            try:
                print("\nAttempting alternative encoding settings...")
                alternative_command = [
                    'ffmpeg',
                    '-framerate', str(fps),
                    '-i', f'{temp_dir}/frame_%06d.png',
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23',
                    '-vf', f'scale=-1:{target_height}',
                    '-pix_fmt', 'yuv420p',
                    '-max_muxing_queue_size', '1024',
                    '-y',
                    output_path
                ]
                
                result = subprocess.run(alternative_command, capture_output=True, text=True, check=True)
                print("Video creation successful with alternative settings")
                return True
                
            except subprocess.CalledProcessError as e2:
                print("\nAlternative encoding also failed:")
                print(e2.stderr)
                raise RuntimeError("Failed to create video with both standard and alternative settings")

    def enhance_video(self, input_path, output_path, target_height=720, resume=True):
        """Process video with disk-based frame storage and ability to resume"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Convert WMV if needed
        if input_path.lower().endswith('.wmv'):
            temp_mp4 = "temp_input.mp4"
            if not os.path.exists(temp_mp4):
                self._convert_wmv_to_mp4(input_path, temp_mp4)
            input_path = temp_mp4

        # Create or check temp directory
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)
        
        # Check for existing frames
        existing_frames = []
        if resume:
            existing_frames = sorted(glob.glob(str(temp_dir / "frame_*.png")))
            if existing_frames:
                last_frame_num = int(os.path.basename(existing_frames[-1]).split('_')[1].split('.')[0])
                print(f"Found {len(existing_frames)} existing frames. Resuming from frame {last_frame_num + 1}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nProcessing video:")
        print(f"Input resolution: {width}x{height}")
        print(f"Target height: {target_height}")
        print(f"Original FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Skip to last processed frame if resuming
        frame_count = 0
        if existing_frames and resume:
            frame_count = len(existing_frames)
            for _ in range(frame_count):
                cap.read()
        
        try:
            with tqdm(total=total_frames, initial=frame_count, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    enhanced_frame = self.process_frame(frame)
                    
                    # Save frame
                    frame_path = temp_dir / f"frame_{frame_count:06d}.png"
                    cv2.imwrite(str(frame_path), enhanced_frame)
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # Periodically clear memory
                    if frame_count % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            cap.release()
            
            # Create final video
            print("\nVerifying frames before video creation...")
            if self._verify_frames(temp_dir):
                print("Frame verification successful")
                print("\nCreating final video...")
                self._create_video_from_frames(
                    str(temp_dir),
                    output_path,
                    fps,
                    target_height
                )
                
                # Cleanup
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                
                if input_path.startswith('temp_'):
                    os.remove(input_path)
                
                print(f"\nVideo enhancement complete: {output_path}")
            else:
                raise RuntimeError("Frame verification failed")
            
        except KeyboardInterrupt:
            print

class FrameInterpolator(nn.Module):
    def __init__(self):
        super(FrameInterpolator, self).__init__()
        # Flow estimation network
        self.flow_net = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 4, 3, 1, 1)  # 2 channels each for forward and backward flow
        )
        
        # Frame synthesis network
        self.synth_net = nn.Sequential(
            nn.Conv2d(10, 64, 3, 1, 1),  # 6 from warped frames + 4 from flow
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, frame1, frame2):
        # Concatenate frames
        input_frames = torch.cat([frame1, frame2], dim=1)
        
        # Estimate bidirectional flow
        flows = self.flow_net(input_frames)
        flow_f, flow_b = flows.chunk(2, dim=1)
        
        # Warp frames using estimated flow
        warped1 = self._warp(frame1, flow_f)
        warped2 = self._warp(frame2, flow_b)
        
        # Synthesize intermediate frame
        synth_input = torch.cat([warped1, warped2, flows], dim=1)
        intermediate = self.synth_net(synth_input)
        
        return intermediate
    
    def _warp(self, x, flow):
        B, C, H, W = x.size()
        # Create mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float().to(x.device)
        
        # Add flow to grid
        vgrid = grid + flow
        
        # Normalize grid values to [-1,1] for grid_sample
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
        
        # Reshape for grid_sample
        vgrid = vgrid.permute(0,2,3,1)
        
        # Perform warping
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border')
        return output

class EnhancedVideoProcessor(VideoProcessor):
    def __init__(self, sr_weights_path, interpolation_weights_path=None, device='cuda'):
        super().__init__(sr_weights_path, device)
        self.interpolator = FrameInterpolator().to(device)
        self.interpolator.eval()
        
        if interpolation_weights_path and os.path.exists(interpolation_weights_path):
            self.load_interpolation_weights(interpolation_weights_path)
    
    def load_interpolation_weights(self, weights_path):
        try:
            # Load with weights_only=True for security
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            
            # Filter out unexpected keys
            model_state_dict = self.interpolator.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            
            # Check if we have all necessary keys
            missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
            if missing_keys:
                raise ValueError(f"Missing keys in state dict: {missing_keys}")
            
            self.interpolator.load_state_dict(filtered_state_dict, strict=False)
            print(f"Successfully loaded interpolation weights from {weights_path}")
        except Exception as e:
            raise Exception(f"Error loading interpolation weights: {str(e)}")

    def interpolate_frame(self, frame1, frame2):
        """Generate intermediate frame between two frames"""
        # Convert BGR to RGB and normalize
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) / 255.0
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) / 255.0
        
        # Convert to tensors
        frame1_tensor = torch.from_numpy(frame1_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        frame2_tensor = torch.from_numpy(frame2_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            intermediate = self.interpolator(frame1_tensor, frame2_tensor)
        
        # Convert back to image
        intermediate = intermediate.clamp(0, 1)
        intermediate_array = (intermediate.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        intermediate_frame = cv2.cvtColor(intermediate_array, cv2.COLOR_RGB2BGR)
        
        return intermediate_frame

    def enhance_video(self, input_path, output_path, target_height=720, target_fps=60):
        """Process video with both super-resolution and frame interpolation"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Convert WMV if needed
        if input_path.lower().endswith('.wmv'):
            temp_mp4 = "temp_input.mp4"
            self._convert_wmv_to_mp4(input_path, temp_mp4)
            input_path = temp_mp4

        # Create temp directory
        temp_dir = Path("temp_frames")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate required interpolation factor
        interpolation_factor = math.ceil(target_fps / original_fps)
        
        print(f"\nProcessing video:")
        print(f"Original FPS: {original_fps}")
        print(f"Target FPS: {target_fps}")
        print(f"Interpolation factor: {interpolation_factor}x")
        print(f"Total frames to generate: {total_frames * interpolation_factor}")
        
        # Process frames
        frame_count = 0
        previous_frame = None
        
        with tqdm(total=total_frames * interpolation_factor, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Enhance current frame
                enhanced_frame = self.process_frame(frame)
                
                # Handle interpolation
                if previous_frame is not None:
                    for i in range(interpolation_factor - 1):
                        # Generate intermediate frame
                        t = (i + 1) / interpolation_factor
                        intermediate = self.interpolate_frame(previous_frame, enhanced_frame)
                        
                        # Save intermediate frame
                        cv2.imwrite(str(temp_dir / f"frame_{frame_count:06d}.png"), intermediate)
                        frame_count += 1
                        pbar.update(1)
                
                # Save enhanced frame
                cv2.imwrite(str(temp_dir / f"frame_{frame_count:06d}.png"), enhanced_frame)
                frame_count += 1
                pbar.update(1)
                
                previous_frame = enhanced_frame.copy()
                
                # Periodically clear memory
                if frame_count % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        cap.release()
        
        # Create final video
        print("\nCreating final video...")
        self._create_video_from_frames(
            str(temp_dir),
            output_path,
            target_fps,
            target_height
        )
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        if input_path.startswith('temp_'):
            os.remove(input_path)
        
        print(f"\nVideo enhancement complete: {output_path}")

def main():
    import sys
    from utils.config_loader import load_config

    config = load_config()
    processor_config = config["video_processor"]

    try:
        # Initialize processor with both super-resolution and interpolation models
        processor = EnhancedVideoProcessor(
            sr_weights_path=processor_config["sr_weights_path"],
            interpolation_weights_path=processor_config["interpolation_weights_path"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Process video
        processor.enhance_video(
            input_path=processor_config["input_video"],
            output_path=processor_config["output_video"],
            target_height=processor_config.get("target_height", 720),
            target_fps=processor_config.get("target_fps", 60)
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()