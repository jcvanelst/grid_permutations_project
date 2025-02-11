import cv2
import numpy as np
from PIL import Image
import os
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import librosa
from tqdm import tqdm
from pathlib import Path

class AudioReactivePanelGenerator:
    def __init__(self, input_dir, output_file, audio_file, 
                 sync_mode='beat', fps=30, min_transition=0.2, max_transition=1.0):
        self.input_dir = input_dir
        self.output_file = output_file
        self.audio_file = audio_file
        self.sync_mode = sync_mode
        self.fps = fps
        self.min_transition = min_transition
        self.max_transition = max_transition
        
        # Image settings
        self.image_width = 900
        self.image_height = 900
        self.margin_percent = 0.01
        self.margin = int(self.image_width * self.margin_percent)
        self.panel_size = (self.image_width - (4 * self.margin)) // 3
        
        # Load and analyze audio file
        print("Analyzing audio file...")
        self.y, self.sr = librosa.load(audio_file)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Extract audio features
        self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.tempo = float(self.tempo)  # Convert to scalar float
        self.beat_times = librosa.frames_to_time(self.beats)
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
        # Load and prepare images
        self.images = self.load_images(input_dir)
        print(f"Loaded {len(self.images)} panel images")

    def load_images(self, input_dir):
        """Load and resize all panel images"""
        image_paths = sorted([
            str(f) for f in Path(input_dir).glob('*')
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ])[:9]
        
        if len(image_paths) != 9:
            raise ValueError(f"Need exactly 9 images, found {len(image_paths)}")
        
        images = []
        for path in image_paths:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((self.panel_size, self.panel_size), Image.Resampling.LANCZOS)
            
            # Center image in panel
            new_img = Image.new('RGB', (self.panel_size, self.panel_size), 'white')
            offset = ((self.panel_size - img.width) // 2,
                     (self.panel_size - img.height) // 2)
            new_img.paste(img, offset)
            images.append(np.array(new_img))
        
        return images

    def create_grid_frame(self, arrangement, effect_intensity=0):
        """Create a single frame with the current arrangement of panels"""
        frame = np.full((self.image_height, self.image_width, 3), 255, dtype=np.uint8)
        
        for idx, img_idx in enumerate(arrangement):
            row = idx // 3
            col = idx % 3
            
            x = self.margin + (col * (self.panel_size + self.margin))
            y = self.margin + (row * (self.panel_size + self.margin))
            
            # Get panel image
            panel = self.images[img_idx].copy()
            
            # Apply audio-reactive effects
            if effect_intensity > 0:
                # Brighten based on audio intensity
                panel = cv2.convertScaleAbs(panel, alpha=1 + effect_intensity * 0.3, beta=0)
            
            # Place panel in frame
            frame[y:y+self.panel_size, x:x+self.panel_size] = panel
        
        return frame

    def create_transition(self, frame1, frame2, progress, effect='crossfade'):
        """Create transition between two frames with proper type handling"""
        # Ensure progress is a float between 0 and 1
        progress = float(np.clip(progress, 0.0, 1.0))

        if effect == 'crossfade':
            return cv2.addWeighted(
                frame1.astype(np.float32), 
                1.0 - progress,
                frame2.astype(np.float32), 
                progress, 
                0.0
            ).astype(np.uint8)
        elif effect == 'slide':
            h, w = frame1.shape[:2]
            offset = int(w * progress)
            result = np.copy(frame1)
            result[:, :w-offset] = frame1[:, offset:]
            result[:, w-offset:] = frame2[:, :offset]
            return result
        elif effect == 'zoom':
            scale = 1.0 + progress * 0.2
            center = (frame1.shape[1] // 2, frame1.shape[0] // 2)
            mat = cv2.getRotationMatrix2D(center, 0, scale)
            frame1_zoomed = cv2.warpAffine(frame1, mat, (frame1.shape[1], frame1.shape[0]))
            return cv2.addWeighted(
                frame1_zoomed.astype(np.float32), 
                1.0 - progress,
                frame2.astype(np.float32), 
                progress, 
                0.0
            ).astype(np.uint8)
        
        # Default fallback
        return cv2.addWeighted(
            frame1.astype(np.float32), 
            1.0 - progress,
            frame2.astype(np.float32), 
            progress, 
            0.0
        ).astype(np.uint8)

    # def get_audio_intensity(self, time):
    #     """Get normalized audio intensity at given time"""
    #     frame = librosa.time_to_frames(time, sr=self.sr)
    #     if frame >= len(self.onset_env):
    #         return 0
    #     return float(min(1.0, self.onset_env[frame] / np.max(self.onset_env)))
    
    def get_audio_intensity(self, time):
        """Get normalized audio intensity at given time"""
        frame = librosa.time_to_frames(time, sr=self.sr)
        if frame >= len(self.onset_env):
            return 0.0
        
        # Properly handle array to scalar conversion
        max_onset = np.max(self.onset_env)
        if max_onset == 0:
            return 0.0
            
        intensity = float(self.onset_env[frame]) / float(max_onset)
        return min(1.0, intensity)

    def make_video(self):
        """Main method to generate the video"""
        frames = []
        arrangements = list(range(9))  # Start with original arrangement
        current_arrangement = arrangements.copy()
        next_arrangement = arrangements.copy()
        
        time = 0
        transition_in_progress = False
        transition_progress = 0
        transition_duration = 60 / self.tempo  # One beat duration
        
        print(f"Generating video frames (duration: {self.duration:.1f}s, FPS: {self.fps})")
        
        with tqdm(total=int(self.duration * self.fps)) as pbar:
            while time < self.duration:
                # Check for beat or high intensity
                intensity = self.get_audio_intensity(time)
                should_transition = False
                
                # Check for beat based on beat times
                for beat_time in self.beat_times:
                    if abs(time - beat_time) < 0.1:  # Within 100ms of a beat
                        should_transition = True
                        break
                
                if not transition_in_progress and (should_transition or intensity > 0.7):
                    transition_in_progress = True
                    transition_progress = 0
                    # Shuffle next arrangement
                    next_arrangement = current_arrangement.copy()
                    np.random.shuffle(next_arrangement)

                if transition_in_progress:
                    # Choose transition effect based on audio intensity
                    effect = 'crossfade'
                    if intensity > 0.8:
                        effect = 'zoom'
                    elif intensity > 0.5:
                        effect = 'slide'

                    # Create frames for both arrangements
                    frame1 = self.create_grid_frame(current_arrangement, intensity)
                    frame2 = self.create_grid_frame(next_arrangement, intensity)
                    
                    # Create transition frame
                    frame = self.create_transition(frame1, frame2, 
                                                transition_progress, effect)
                    
                    transition_progress += 1 / (transition_duration * self.fps)
                    
                    if transition_progress >= 1.0:
                        transition_in_progress = False
                        current_arrangement = next_arrangement.copy()
                else:
                    frame = self.create_grid_frame(current_arrangement, intensity)

                frames.append(frame)
                time += 1 / self.fps
                pbar.update(1)

        print("Creating video clip...")
        clip = ImageSequenceClip(frames, fps=self.fps)
        
        print("Adding audio...")
        audio = AudioFileClip(self.audio_file)
        final_clip = clip.set_audio(audio)

        print("Writing video file...")
        final_clip.write_videofile(
            self.output_file,
            codec='libx264',
            audio_codec='aac',
            fps=self.fps,
            bitrate='8000k',
            preset='slow',
            threads=4
        )

def main():
    parser = argparse.ArgumentParser(description='Generate audio-reactive panel animation')
    parser.add_argument('input_dir', help='Directory containing 9 panel images')
    parser.add_argument('output_file', help='Output video file path')
    parser.add_argument('audio_file', help='Audio file to sync with')
    parser.add_argument('--sync-mode', choices=['beat', 'intensity', 'segment'],
                        default='beat', help='Audio synchronization mode')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()

    generator = AudioReactivePanelGenerator(
        input_dir=args.input_dir,
        output_file=args.output_file,
        audio_file=args.audio_file,
        sync_mode=args.sync_mode,
        fps=args.fps
    )
    
    generator.make_video()

if __name__ == "__main__":
    main()
