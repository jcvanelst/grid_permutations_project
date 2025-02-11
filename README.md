# Grid Permutations Video Generator

This project generates visual permutations of a 3x3 grid and can create videos synchronized with audio.

## Setup

1. Clone this repository
2. Run the setup script:
   ```bash
   cd setup
   chmod +x setup_python_env.sh
   ./setup_python_env.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

## Usage

1. Generate grid permutations:
   ```bash
   python src/grid_generator.py input_images/ output/
   ```

2. Create audio-reactive video:
   ```bash
   python src/audio_reactive_generator.py output/ final_video.mp4 music.mp3
   ```

See docs/usage_guide.md for detailed instructions and options.
