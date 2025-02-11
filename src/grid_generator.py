from PIL import Image, ImageDraw, ImageFont
import numpy as np
from itertools import permutations
import os

class GridPermutationGenerator:
    def __init__(self, output_dir="permutations"):
        self.output_dir = output_dir
        self.positions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        
        # Image settings
        self.image_width = 900  # Width of the final image
        self.image_height = 900  # Height of the final image
        self.margin_percent = 0.015  # 7% margin
        self.bg_color = "white"
        self.square_color = "lightgray"
        self.text_color = "black"
        
        # Calculate margins and grid dimensions
        self.margin = int(self.image_width * self.margin_percent)
        self.grid_size = 3
        self.square_size = (self.image_width - (4 * self.margin)) // 3
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def create_grid_image(self, arrangement):
        # Create a new image with white background
        img = Image.new('RGB', (self.image_width, self.image_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", size=self.square_size // 3)
        except:
            font = ImageFont.load_default()

        # Draw squares and numbers
        for idx, number in enumerate(arrangement):
            row = idx // 3
            col = idx % 3
            
            # Calculate position for the square
            x1 = self.margin + (col * (self.square_size + self.margin))
            y1 = self.margin + (row * (self.square_size + self.margin))
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            
            # Draw square
            draw.rectangle([x1, y1, x2, y2], fill=self.square_color, outline=self.text_color)
            
            # Draw number
            number_text = str(number)
            text_bbox = draw.textbbox((0, 0), number_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = x1 + (self.square_size - text_width) // 2
            text_y = y1 + (self.square_size - text_height) // 2
            draw.text((text_x, text_y), number_text, fill=self.text_color, font=font)

        return img

    def generate_filename(self, arrangement):
        # Create filename based on position-number pairs
        filename_parts = [f"{pos}{num}" for pos, num in zip(self.positions, arrangement)]
        return "".join(filename_parts) + ".png"

    def generate_permutations(self, num_permutations=None):
        numbers = list(range(1, 10))  # Numbers 1-9
        all_perms = list(permutations(numbers))
        
        if num_permutations is not None:
            all_perms = all_perms[:num_permutations]
        
        total_perms = len(all_perms)
        print(f"Generating {total_perms} permutations...")
        
        for idx, arrangement in enumerate(all_perms, 1):
            img = self.create_grid_image(arrangement)
            filename = self.generate_filename(arrangement)
            img.save(os.path.join(self.output_dir, filename))
            
            if idx % 1000 == 0:
                print(f"Generated {idx}/{total_perms} permutations")

def main():
    generator = GridPermutationGenerator()
    
    # Generate first 10 permutations as an example
    # Remove the num_permutations parameter to generate all 362,880 permutations
    generator.generate_permutations(num_permutations=10)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()
