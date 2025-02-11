# Usage Guide

## Grid Generator
The grid generator creates permutations of your images in a 3x3 grid.

### Basic Usage
```bash
python src/grid_generator.py input_images/ output/
```

### Options
- --num_permutations: Limit number of permutations
- --margin: Set margin size (default: 7%)
- --image_size: Set output image size

## Audio-Reactive Generator
Creates videos synchronized with music.

### Basic Usage
```bash
python src/audio_reactive_generator.py images/ output.mp4 music.mp3
```

### Sync Modes
- beat: Sync with music beats
- intensity: Sync with audio intensity
- segment: Sync with song structure

### Examples
1. Beat-synced video:
   ```bash
   python src/audio_reactive_generator.py example/images/ output.mp4 example/audio/sound.m4a --sync-mode beat
   ```

2. High-quality output:
   ```bash
   python src/audio_reactive_generator.py example/images/ output.mp4 example/audio/sound.m4a --fps 60
   ```
