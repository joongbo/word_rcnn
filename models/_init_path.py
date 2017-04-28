import os, sys

'''Set up paths for RCNN.'''
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.getcwd()
libs_path = os.path.join(this_dir, '..', 'libs', 'layers') # Add lib to PYTHONPATH
add_path(libs_path)
libs_path = os.path.join(this_dir, '..', 'libs', 'utils') # Add lib to PYTHONPATH
add_path(libs_path)