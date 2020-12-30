# -*- encoding: utf-8 -*-
'''
@File    :   get_file_names.py
@Time    :   2020/12/30 21:55:45
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   测试将当前目录下的所有文件名写入指定文件的 demo
'''


import bstool


if __name__ == '__main__':
    all_fns = bstool.get_file_names_recursion('/mnt/lustre/menglingxuan/buildingwolf/traindata2/vis_result')
    
    with open('/mnt/lustre/wangjinwang/documents/vis_fns.txt', 'w') as f:
        for fn in all_fns:
            f.write(fn + '\n')