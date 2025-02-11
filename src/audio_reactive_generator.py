import cv2
import numpy as np
from PIL import Image
import os
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import librosa
from tqdm import tqdm

class AudioReactiveGenerator:
    def __init__(self, input_dir, output_file, audio_file, 
                 sync_mode='beat', fps=30, min_transition=0.2, max_transition=1.0):
        self.input_dir = input_dir
        self.output_file = output_file
        self.audio_file = audio_file
        self.sync_mode = sync_mode
        self.fps = fps
        self.min_transition = min_transition
        self.max_transition = max_transition
        
        # Load and analyze audio file
        print("Analyzing audio file...")
        self.y, self.sr = librosa.load(audio_file)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Extract audio features
        #self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.tempo = float(self.tempo)  # Convert to scalar float
        self.beat_times = librosa.frames_to_time(self.beats)
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.spectral = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        
        # Segment analysis for structural changes
        #self.segments = librosa.segment.detect_onsets(onset_envelope=self.onset_env)
        self.segments = librosa.onset.onset_detect(onset_envelope=self.onset_env)

        self.segment_times = librosa.frames_to_time(self.segments)

    def get_image_files(self):
        """Get sorted list of image files from input directory."""
        image_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return sorted(image_files)

    def create_transition(self, img1, img2, progress, effect='crossfade'):
        """Create various transition effects between images."""
        if effect == 'crossfade':
            return cv2.addWeighted(img1, 1 - progress, img2, progress, 0)
        elif effect == 'slide':
            h, w = img1.shape[:2]
            offset = int(w * progress)
            result = np.copy(img1)
            result[:, :w-offset] = img1[:, offset:]
            result[:, w-offset:] = img2[:, :offset]
            return result
        elif effect == 'zoom':
            scale = 1 + progress * 0.2
            center = (img1.shape[1] // 2, img1.shape[0] // 2)
            mat = cv2.getRotationMatrix2D(center, 0, scale)
            img1_zoomed = cv2.warpAffine(img1, mat, (img1.shape[1], img1.shape[0]))
            return cv2.addWeighted(img1_zoomed, 1 - progress, img2, progress, 0)
        return cv2.addWeighted(img1, 1 - progress, img2, progress, 0)

    def get_audio_intensity(self, time):
        """Get normalized audio intensity at given time."""
        frame = librosa.time_to_frames(time, sr=self.sr)
        if frame >= len(self.onset_env):
            return 0
        return min(1.0, self.onset_env[frame] / np.max(self.onset_env))

    def get_frequency_balance(self, time):
        """Get balance between high and low frequencies at given time."""
        frame = librosa.time_to_frames(time, sr=self.sr)
        if frame >= self.spectral.shape[1]:
            return 0
        spec_slice = self.spectral[:, frame]
        low_energy = np.mean(spec_slice[:len(spec_slice)//3])
        high_energy = np.mean(spec_slice[2*len(spec_slice)//3:])
        return high_energy / (low_energy + 1e-10)

    def should_transition(self, time):
        """Determine if a transition should occur based on sync_mode."""
        if self.sync_mode == 'beat':
            return any(abs(t - time) < 0.1 for t in self.beat_times)
        elif self.sync_mode == 'intensity':
            return self.get_audio_intensity(time) > 0.7
        elif self.sync_mode == 'segment':
            return any(abs(t - time) < 0.1 for t in self.segment_times)
        return False

    def apply_audio_reactive_effects(self, frame, time):
        """Apply audio-reactive effects to a frame."""
        intensity = self.get_audio_intensity(time)
        freq_balance = self.get_frequency_balance(time)
        
        # Adjust brightness based on intensity
        frame = cv2.convertScaleAbs(frame, alpha=1 + intensity * 0.3, beta=0)
        
        # Add subtle color effects based on frequency balance
        if freq_balance > 1.5:  # High frequency dominated
            frame = cv2.addWeighted(frame, 1, np.full_like(frame, [0, 0, 50], dtype=np.uint8), 0.2, 0)
        elif freq_balance < 0.5:  # Low frequency dominated
            frame = cv2.addWeighted(frame, 1, np.full_like(frame, [50, 0, 0], dtype=np.uint8), 0.2, 0)
            
        return frame

    def generate_video(self):
        image_files = self.get_image_files()
        num_images = len(image_files)
        
        if num_images == 0:
            raise ValueError("No images found in input directory")

        print(f"Generating video with {num_images} images...")
        print(f"Audio duration: {self.duration:.2f} seconds")
        print(f"Detected tempo: {self.tempo:.1f} BPM")

        # Create frames list for MoviePy
        frames = []
        current_image_idx = 0
        time = 0
        transition_in_progress = False
        transition_progress = 0
        transition_duration = 60 / self.tempo  # One beat duration
        
        # Load first image
        current_img_path = os.path.join(self.input_dir, image_files[current_image_idx])
        current_frame = np.array(Image.open(current_img_path))

        with tqdm(total=int(self.duration * self.fps)) as pbar:
            while time < self.duration:
                if self.should_transition(time) and not transition_in_progress:
                    transition_in_progress = True
                    transition_progress = 0
                    next_image_idx = (current_image_idx + 1) % num_images
                    next_img_path = os.path.join(self.input_dir, image_files[next_image_idx])
                    next_frame = np.array(Image.open(next_img_path))

                if transition_in_progress:
                    # Choose transition effect based on audio features
                    intensity = self.get_audio_intensity(time)
                    effect = 'crossfade'
                    if intensity > 0.8:
                        effect = 'zoom'
                    elif intensity > 0.5:
                        effect = 'slide'

                    frame = self.create_transition(current_frame, next_frame, 
                                                transition_progress, effect)
                    transition_progress += 1 / (transition_duration * self.fps)

                    if transition_progress >= 1.0:
                        transition_in_progress = False
                        current_image_idx = next_image_idx
                        current_frame = next_frame
                else:
                    frame = current_frame.copy()

                # Apply audio-reactive effects
                frame = self.apply_audio_reactive_effects(frame, time)
                frames.append(frame)
                
                time += 1 / self.fps
                pbar.update(1)

        # Create video clip
        print("Creating final video...")
        clip = ImageSequenceClip(frames, fps=self.fps)
        audio = AudioFileClip(self.audio_file)
        final_clip = clip.set_audio(audio)

        # Write video with high quality compression
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
    parser = argparse.ArgumentParser(description='Generate audio-reactive video from image sequence')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_file', help='Output video file path')
    parser.add_argument('audio_file', help='Audio file to sync with')
    parser.add_argument('--sync-mode', choices=['beat', 'intensity', 'segment'],
                        default='beat', help='Audio synchronization mode')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()

    generator = AudioReactiveGenerator(
        input_dir=args.input_dir,
        output_file=args.output_file,
        audio_file=args.audio_file,
        sync_mode=args.sync_mode,
        fps=args.fps
    )
    
    generator.generate_video()

if __name__ == "__main__":
    main()
