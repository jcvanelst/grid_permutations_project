import cv2
import numpy as np
from PIL import Image
import librosa
#import moviepy.editor as mp
#import moviepy.editor as mp
import moviepy as mp
from moviepy import *
from tqdm import tqdm

def test_imports():
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    print("Pillow version:", Image.__version__)
    print("MoviePy version:", mp.__version__)
    print("Librosa version:", librosa.__version__)
    print("\nAll required packages are successfully installed!")

if __name__ == "__main__":
    test_imports()