import os
import six


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)

def get_basename(file_path):
    """get basename file name of file or path (no postfix)

    Args:
        file_path (str): input path or file

    Returns:
        str: base name
    """
    basename = os.path.splitext(os.path.basename(file_path))[0]

    return basename

def get_dir_name(file_path):
    """get the dir name

    Args:
        file_path (str): input path of file

    Returns:
        str: dir name
    """
    dir_name = os.path.abspath(os.path.dirname(file_path))

    return dir_name