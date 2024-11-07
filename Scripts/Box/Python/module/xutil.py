import os

# Directory
def create_dir(dir):
    try:
       os.makedirs(dir)
    except FileExistsError:
       pass
