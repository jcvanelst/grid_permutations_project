#!/bin/bash
# setup_python_env.sh

# First, ensure Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install pyenv and dependencies
brew install pyenv
brew install openssl readline sqlite3 xz zlib

# Add pyenv to shell (for bash or zsh)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc

# Install Python 3.11 (current stable version good for most libraries)
pyenv install 3.11.7
pyenv global 3.11.7

# Create a new project directory
mkdir -p ~/python_projects/video_generator
cd ~/python_projects/video_generator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install opencv-python
pip install pillow
pip install moviepy
pip install librosa
pip install tqdm
pip install numpy

# Create requirements.txt
pip freeze > requirements.txt

# Create a test script to verify installations
cat > test_setup.py << EOL
import cv2
import numpy as np
from PIL import Image
import librosa
import moviepy.editor as mp
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
EOL

# Run test script
python test_setup.py
