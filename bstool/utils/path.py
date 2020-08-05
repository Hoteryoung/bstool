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


def get_info_splitted_imagename(img_name):
    base_name = get_basename(img_name)
    print(base_name)
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

    return sub_fold, ori_image_fn, (coord_x, coord_y)