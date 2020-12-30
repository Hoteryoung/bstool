import os
import shutil


def move_file(srcfile, dstfile):
    """move src file to dst file

    Args:
        srcfile (str): source file
        dstfile (str): destination file
    """
    if not os.path.isfile(srcfile):
        print(f"{srcfile} not exist!")
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath) 
        shutil.move(srcfile, dstfile)
        print(f"move {srcfile} -> {dstfile}")

def copy_file(srcfile, dstfile):
    """copy src file to dst file

    Args:
        srcfile (str): source file
        dstfile (str): destination file
    """
    if not os.path.isfile(srcfile):
        print(f"{srcfile} not exist!")
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(srcfile, dstfile)
        print(f"copy {srcfile} -> {dstfile}")