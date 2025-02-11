# test_moviepy.py
try:
    import moviepy.editor as mp
    print("MoviePy imported successfully")
except Exception as e:
    print(f"Error importing MoviePy: {e}")
    
try:
    from moviepy.editor import VideoFileClip
    print("VideoFileClip imported successfully")
except Exception as e:
    print(f"Error importing VideoFileClip: {e}")

