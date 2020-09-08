import bstool


if __name__ == '__main__':
    all_fns = bstool.get_file_names_recursion('/mnt/lustre/menglingxuan/buildingwolf/traindata2/vis_result')
    
    with open('/mnt/lustre/wangjinwang/documents/vis_fns.txt', 'w') as f:
        for fn in all_fns:
            f.write(fn)