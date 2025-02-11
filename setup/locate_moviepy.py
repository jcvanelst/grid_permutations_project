## locate_moviepy.py
import os
import moviepy
print("Moviepy installation path:", os.path.dirname(moviepy.__file__))
print("Moviepy contents:", os.listdir(os.path.dirname(moviepy.__file__)))
