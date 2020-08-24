import os


if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_folds = {'beijing':  ['arg', 'google', 'ms', 'tdt'],
                 'chengdu':  ['arg', 'google', 'ms', 'tdt'],
                 'haerbin':  ['arg', 'google', 'ms'],
                 'jinan':    ['arg', 'google', 'ms', 'tdt'],
                 'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}

    for city in cities:
        for sub_fold in sub_folds[city]:
            src_root_path = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_18'
            src_path = os.path.join(src_root_path, f"{sub_fold}/annoV2/OffsetField")
            
            dst_root_path = f'/mnt/lustre/wangjinwang/data/buildchange/v0/{city}'
            dst_path = os.path.join(dst_root_path, f"{sub_fold}/offset_field")

            os.symlink(src_path, dst_path)