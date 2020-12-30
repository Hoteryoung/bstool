# -*- encoding: utf-8 -*-
'''
@File    :   vis_channel_featuremap.py
@Time    :   2020/12/30 22:34:22
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化通道特征
'''


import os
import numpy as np
from matplotlib import pyplot as plt
import bstool

def axis_coord(x, y):
    if x == 0:
        if y == 0:
            return 0, 1
        else:
            return 1, 0
    else:
        if y == 0:
            return 1, 1
        else:
            return 1, 1

def plot_channel_feat(feature_maps, output_file=None):
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    
    x, y = 0, 0
    for angle in [0, 90, 180, 270]:
        feature_map = feature_maps[angle]
        if len(feature_map.shape) > 1:
            channel_feat = np.mean(np.mean(feature_map, axis=1), axis=1)
            # ax[x, y].set_ylim(0, 0.1)
        elif len(feature_map.shape) == 1:
            channel_feat = feature_map
        else:
            raise NotImplementedError
        ax[x, y].plot(channel_feat)
        ax[x, y].set_title(str(angle))
        
        x, y = axis_coord(x, y)

    fig.suptitle('channel vis with different angle')
    plt.savefig(os.path.join(output_file, 'channel_vis_4_angles.png'), bbox_inches='tight', pad_inches=0.1)
    plt.clf()

def plot_same_gap_feat(feature_maps, output_file=None):    
    gap_angles = [90, 180, 270]
    
    for gap_angle in gap_angles:
        for angle1 in [0, 90, 180, 270]:
            for angle2 in [0, 90, 180, 270]:
                if abs(angle1 - angle2) != gap_angle:
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                if len(feature_map.shape) > 1:
                    channel_feat1, channel_feat2 = np.mean(np.mean(feature_maps[angle1], axis=1), axis=1), np.mean(np.mean(feature_maps[angle2], axis=1), axis=1)
                    # ax.set_ylim(-0.1, 0.1)
                elif len(feature_map.shape) == 1:
                    channel_feat1, channel_feat2 = feature_maps[angle1], feature_maps[angle2]
                else:
                    raise NotImplementedError
                
                gap_feature = channel_feat1 - channel_feat2
                print(f"The channel of max output (angle1: {angle1}, angle2: {angle2}): ", np.argmax(np.abs(gap_feature)))
                ax.plot(gap_feature)
                ax.set_title(str(angle1) + " - " + str(angle2))

                plt.savefig(os.path.join(output_file, f'same_gap_vis_{angle1}_{angle2}_channel.png'), bbox_inches='tight', pad_inches=0.1)
                plt.clf()

def plot_same_gap_feat_show_level(feature_maps, output_file=None):    
    gap_angles = [90, 180, 270]
    
    for gap_angle in gap_angles:
        for angle1 in [0, 90, 180, 270]:
            for angle2 in [0, 90, 180, 270]:
                if abs(angle1 - angle2) != gap_angle:
                    continue
                fig, ax = plt.subplots(4, 1, figsize=(8, 8))
                if len(feature_map.shape) > 1:
                    channel_feat1, channel_feat2 = np.mean(np.mean(np.abs(feature_maps[angle1]), axis=1), axis=1), np.mean(np.mean(np.abs(feature_maps[angle2]), axis=1), axis=1)
                    # ax.set_ylim(-0.1, 0.1)
                elif len(feature_map.shape) == 1:
                    channel_feat1, channel_feat2 = feature_maps[angle1], feature_maps[angle2]
                else:
                    raise NotImplementedError
                
                gap_feature = channel_feat1 - channel_feat2
                print(f"The channel of max output (angle1: {angle1}, angle2: {angle2}): ", np.argmax(np.abs(gap_feature)))
                ax[0].plot(channel_feat1)
                ax[0].set_ylim(0, 0.1)

                ax[1].plot(channel_feat2)
                ax[1].set_ylim(0, 0.1)

                ax[2].plot(gap_feature)
                ax[2].set_ylim(-0.1, 0.1)

                ax[3].plot(np.sort(gap_feature)[::-1])
                ax[3].set_ylim(-0.1, 0.1)

                fig.suptitle(str(angle1) + " - " + str(angle2))

                plt.savefig(os.path.join(output_file, f'same_gap_vis_{angle1}_{angle2}_channel_show_level.png'), bbox_inches='tight', pad_inches=0.1)
                plt.clf()


if __name__ == '__main__':
    feat_dir = '/data/buildchange/analysis/offset_features'
    vis_dir = '/data/buildchange/analysis/vis'

    feature_maps = dict()
    for feat_file in os.listdir(feat_dir):
        angle = float(feat_file.split('.npy')[0].split('_R')[1])
        feat_file = os.path.join(feat_dir, feat_file)
        feature_map = np.load(feat_file)
        feature_maps[angle] = feature_map
    
    # plot_channel_feat(feature_maps, vis_dir)
    # plot_same_gap_feat(feature_maps, vis_dir)
    plot_same_gap_feat_show_level(feature_maps, vis_dir)
        