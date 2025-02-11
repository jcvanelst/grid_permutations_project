import sys
print("Python path:", sys.path)
try:
    import moviepy
    print("Moviepy version:", moviepy.__version__)
    import moviepy.editor
    print("Moviepy.editor imported successfully")
except Exception as e:
    print("Error:", str(e))
