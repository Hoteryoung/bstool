import os
import pandas
import bstool


if __name__ == '__main__':
    csv_dir = './data/urban3d/weijia/instance_gt_val'
    save_dir = './data/urban3d/weijia'

    csv_fns = ['urban3d_jax_val_orgfootprint_offset_gt_subcsv_merge.csv',
               'urban3d_oma_val_orgfootprint_offset_gt_subcsv_merge.csv']

    merged_csv_file = os.path.join(save_dir, 'urban3d_jax_oma_val_orgfootprint_offset_gt_subcsv_merge.csv')
    first_in = True
    for csv_fn in csv_fns:
        csv_file = os.path.join(csv_dir, csv_fn)

        csv_df = pandas.read_csv(csv_file)

        if first_in:
            csv_df.to_csv(merged_csv_file, index=False)
            first_in = False
        else:
            csv_df.to_csv(merged_csv_file, index=False, header=False, mode='a+')
