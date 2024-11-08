import os
import shutil
import json


# Create directory (if not exist)
def create_dir(dir):
    '''
        Usage: create directory (if not exist)
        Required arguments:
            dir: directory name
    '''
    
    try: os.makedirs(dir)
    except FileExistsError: pass


# Copy single file
def copy(srcpath, destpath):
    '''
        Usage: copy single file
        Required arguments:
            srcpath: source pathname
            destpath: destination pathname
    '''
    
    # Split path into directory and file
    srcdir, srcfile = os.path.split(srcpath) # source
    destdir, destfile = os.path.split(destpath) # destination
    
    # Create destination directory (if not exist)
    create_dir(destdir)

    # Copy source file into destination folder (filename unchanged)
    shutil.copy2(srcpath, destdir) # preserve file metadata
    
    # Rename copied file to correct destination filename
    os.rename(f"{destdir}/{srcfile}", destpath)


# Import dictionary from JSON file
def import_dict(jsonpath):
    '''
        Usage: parse JSON data into dictionary
        Required arguments:
            jsonpath: JSON filepath (usually metadata filepath)
        Outputs:
            dictionary
    '''
    
    with open(jsonpath) as file:
    	contents = file.read()

    # JSON data is parsed into dictionary
    return json.loads(contents)
