from PIL import Image
import numpy as np
from itertools import permutations
import os
import argparse
import glob

class GridImagePermutationGenerator:
    def __init__(self, output_dir="permutations"):
        self.output_dir = output_dir
        self.positions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        
        # Image settings
        self.image_width = 900  # Width of the final image
        self.image_height = 900  # Height of the final image
        self.margin_percent = 0.02  # 7% margin
        self.bg_color = "white"
        
        # Calculate margins and grid dimensions
        self.margin = int(self.image_width * self.margin_percent)
        self.grid_size = 3
        self.square_size = (self.image_width - (4 * self.margin)) // 3
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_and_resize_image(self, image_path):
        """Load an image and resize it to fit the grid cell."""
        try:
            img = Image.open(image_path)
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize maintaining aspect ratio
            img.thumbnail((self.square_size, self.square_size), Image.Resampling.LANCZOS)
            # Create new image with white background
            new_img = Image.new('RGB', (self.square_size, self.square_size), self.bg_color)
            # Paste resized image centered
            offset = ((self.square_size - img.size[0]) // 2,
                     (self.square_size - img.size[1]) // 2)
            new_img.paste(img, offset)
            return new_img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def create_grid_image(self, images, arrangement):
        """Create a grid image from the arranged image list."""
        # Create a new image with white background
        img = Image.new('RGB', (self.image_width, self.image_height), self.bg_color)
        
        # Place each image in the grid
        for idx, pos in enumerate(arrangement):
            row = idx // 3
            col = idx % 3
            
            # Calculate position for the image
            x = self.margin + (col * (self.square_size + self.margin))
            y = self.margin + (row * (self.square_size + self.margin))
            
            # Paste the image
            if images[pos-1] is not None:  # pos-1 because arrangement uses 1-based indexing
                img.paste(images[pos-1], (x, y))
        
        return img

    def generate_filename(self, arrangement):
        """Create filename based on position-number pairs."""
        filename_parts = [f"{pos}{num}" for pos, num in zip(self.positions, arrangement)]
        return "".join(filename_parts) + ".png"

    def generate_permutations(self, image_paths, num_permutations=None):
        """Generate permutations of the images."""
        # Load and resize all images
        images = [self.load_and_resize_image(path) for path in image_paths]
        
        if len(images) != 9:
            raise ValueError(f"Expected 9 images, got {len(images)}")
        
        if None in images:
            raise ValueError("Some images failed to load")
        
        # Generate permutations
        numbers = list(range(1, 10))  # Numbers 1-9
        all_perms = list(permutations(numbers))
        
        if num_permutations is not None:
            all_perms = all_perms[:num_permutations]
        
        total_perms = len(all_perms)
        print(f"Generating {total_perms} permutations...")
        
        for idx, arrangement in enumerate(all_perms, 1):
            img = self.create_grid_image(images, arrangement)
            filename = self.generate_filename(arrangement)
            img.save(os.path.join(self.output_dir, filename))
            
            if idx % 1000 == 0:
                print(f"Generated {idx}/{total_perms} permutations")

def main():
    parser = argparse.ArgumentParser(description='Generate grid image permutations')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_folder', help='Folder containing input images')
    group.add_argument('--images', nargs=9, help='List of 9 image paths')
    parser.add_argument('--pattern', default='*.png', help='Pattern for matching image files')
    parser.add_argument('--num_permutations', type=int, help='Number of permutations to generate')
    
    args = parser.parse_args()
    
    # Get image paths
    if args.input_folder:
        image_paths = sorted(glob.glob(os.path.join(args.input_folder, args.pattern)))
        if len(image_paths) < 9:
            raise ValueError(f"Found only {len(image_paths)} images in {args.input_folder}")
        image_paths = image_paths[:9]  # Take first 9 images
    else:
        image_paths = args.images
    
    # Create and run generator
    generator = GridImagePermutationGenerator()
    generator.generate_permutations(image_paths, args.num_permutations)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()
