import os
import six


def mkdir_or_exist(dir_name, mode=0o777):
    """make of check the dir

    Args:
        dir_name (str): directory name 
        mode (str, optional): authority of mkdir. Defaults to 0o777.
    """
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


def get_info_splitted_imagename(img_name):
    """get the information of splitted sub-image from sub-image file name

    Args:
        img_name (str): input image name

    Returns:
        list: file name information
    """
    base_name = get_basename(img_name)
    if base_name.count('__') == 1:
        # urban3d
        sub_fold = None
        ori_image_fn = base_name.split("__")[0]
        coord_x, coord_y = base_name.split("__")[1].split('_')    # top left corner
        coord_x, coord_y = int(coord_x), int(coord_y)
    elif base_name.count('__') == 2:
        # xian_fine
        sub_fold = base_name.split("__")[0].split('_')[1]
        ori_image_fn = base_name.split("__")[1]
        coord_x, coord_y = base_name.split("__")[2].split('_')    # top left corner
        coord_x, coord_y = int(coord_x), int(coord_y)
    elif base_name.count('__') > 2:
        # dalian_fine
        sub_fold = None
        ori_image_fn = base_name.split("__")[1] + '__' + base_name.split("__")[2]
        coord_x, coord_y = base_name.split("__")[3].split('_')    # top left corner
        coord_x, coord_y = int(coord_x), int(coord_y)
    elif base_name.count('__') == 0:
        sub_fold = None 
        ori_image_fn = base_name
        coord_x, coord_y = 0, 0
    return sub_fold, ori_image_fn, (coord_x, coord_y)

def get_files_recursion(root_dir):
    """get files recursion

    Args:
        root_dir (str): input directory

    Returns:
        list: list of files
    """
    all_files = []
    fns = os.listdir(root_dir)
    for fn in fns:
        next_file = os.path.join(root_dir, fn)
        if not os.path.isdir(next_file):
            all_files.append(next_file)
        else:
            all_files += get_files_recursion(next_file)
    
    return all_files

def get_file_names_recursion(root_dir):
    """get file names recursion

    Args:
        root_dir (str): input directory

    Returns:
        list: list of file names
    """
    all_fns = []
    fns = os.listdir(root_dir)
    for fn in fns:
        next_fn = os.path.join(root_dir, fn)
        if not os.path.isdir(next_fn):
            all_fns.append(get_basename(next_fn))
        else:
            all_fns += get_file_names_recursion(next_fn)
    
    return all_fns