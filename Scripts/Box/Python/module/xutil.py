import os
import shutil


# Create directory (if not exist)
def create_dir(dir):
    '''
        Usage: create directory (if not exist)
        Required arguments:
            dir: directory name
    '''
    
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


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
