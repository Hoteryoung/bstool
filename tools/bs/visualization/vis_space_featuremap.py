# -*- encoding: utf-8 -*-
'''
@File    :   vis_space_featuremap.py
@Time    :   2020/12/30 22:36:14
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   对空间特征进行可视化
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

def plot_space_feat(feature_maps, output_file=None):
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    
    x, y = 0, 0
    for angle in [0, 90, 180, 270]:
        feature_map = feature_maps[angle]
        if len(feature_map.shape) > 1:
            space_feat = np.mean(feature_map, axis=0)
            # ax[x, y].set_ylim(0, 0.1)
        elif len(feature_map.shape) == 1:
            space_feat = feature_map
        else:
            raise NotImplementedError
        handle = ax[x, y].matshow(space_feat)
        ax[x, y].set_title(str(angle))
        plt.colorbar(handle)
        
        x, y = axis_coord(x, y)

    fig.suptitle('space vis with different angle')
    plt.savefig(os.path.join(output_file, 'space_vis_4_angles.png'), bbox_inches='tight', pad_inches=0.1)
    plt.clf()

def plot_same_gap_feat(feature_maps, output_file=None):    
    gap_angles = [90, 180, 270]
    
    for gap_angle in gap_angles:
        for angle1 in [0, 90, 180, 270]:
            for angle2 in [0, 90, 180, 270]:
                if abs(angle1 - angle2) != gap_angle:
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                feature_maps[angle1] = np.rot90(feature_maps[angle1], k=angle1/90, axes=(1, 2))
                feature_maps[angle2] = np.rot90(feature_maps[angle2], k=angle2/90, axes=(1, 2))
                space_feat1, space_feat2 = np.mean(feature_maps[angle1], axis=0), np.mean(feature_maps[angle2], axis=0)
                
                handle = ax.matshow(space_feat1 - space_feat2)
                ax.set_title(str(angle1) + " - " + str(angle2))
                plt.colorbar(handle)

                plt.savefig(os.path.join(output_file, f'same_gap_vis_{angle1}_{angle2}_space_reverse.png'), bbox_inches='tight', pad_inches=0.1)
                plt.clf()

def plot_same_gap_feat_show_level(feature_maps, output_file=None):    
    gap_angles = [90, 180, 270]
    
    for gap_angle in gap_angles:
        for angle1 in [0, 90, 180, 270]:
            for angle2 in [0, 90, 180, 270]:
                if abs(angle1 - angle2) != gap_angle:
                    continue
                fig, ax = plt.subplots(3, 1, figsize=(16, 12))
                feature_maps[angle1] = np.rot90(feature_maps[angle1], k=angle1/90, axes=(1, 2))
                feature_maps[angle2] = np.rot90(feature_maps[angle2], k=angle2/90, axes=(1, 2))
                space_feat1, space_feat2 = np.mean(np.abs(feature_maps[angle1]), axis=0), np.mean(np.abs(feature_maps[angle2]), axis=0)
                
                gap_feature = space_feat1 - space_feat2

                handle = ax[0].matshow(space_feat1)
                plt.colorbar(handle, ax=ax[0])

                handle = ax[1].matshow(space_feat2)
                plt.colorbar(handle, ax=ax[1])

                handle = ax[2].matshow(gap_feature)
                plt.colorbar(handle, ax=ax[2])

                fig.suptitle(str(angle1) + " - " + str(angle2))
                plt.savefig(os.path.join(output_file, f'same_gap_vis_{angle1}_{angle2}_space_reverse.png'), bbox_inches='tight', pad_inches=0.1)
                plt.clf()

def plot_same_gap_feat_show_each_channel(feature_maps, output_file=None):        
    angle1, angle2 = 0, 90
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    feature_maps[angle1] = np.rot90(feature_maps[angle1], k=angle1/90, axes=(1, 2))
    feature_maps[angle2] = np.rot90(feature_maps[angle2], k=angle2/90, axes=(1, 2))
    space_feat1, space_feat2 = feature_maps[angle1], feature_maps[angle2]

    for channel_idx in range(256):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        gap_feature = space_feat1[channel_idx, ...] - space_feat2[channel_idx, ...]

        handle = ax.matshow(gap_feature)
        colorbar = plt.colorbar(handle, ax=ax)
        colorbar.set_clim(-0.1, 0.1)
        fig.suptitle(str(angle1) + " - " + str(angle2) + ' : ' + str(channel_idx))
        plt.savefig(os.path.join(output_file, 'each_channel', f'same_gap_vis_{angle1}_{angle2}_space_reverse_{channel_idx}.png'), bbox_inches='tight', pad_inches=0.1)
        plt.close()

def plot_max_channel_space_feat(feature_maps, output_file=None, argmax=True):    
    gap_angles = [90, 180, 270]
    
    for gap_angle in gap_angles:
        for angle1 in [0, 90, 180, 270]:
            for angle2 in [0, 90, 180, 270]:
                if abs(angle1 - angle2) != gap_angle:
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                feature_maps[angle1] = np.rot90(feature_maps[angle1], k=angle1/90, axes=(1, 2))
                feature_maps[angle2] = np.rot90(feature_maps[angle2], k=angle2/90, axes=(1, 2))
                space_feat1, space_feat2 = np.mean(feature_maps[angle1], axis=0), np.mean(feature_maps[angle2], axis=0)
                
                channel_feat1, channel_feat2 = np.mean(np.mean(feature_maps[angle1], axis=1), axis=1), np.mean(np.mean(feature_maps[angle2], axis=1), axis=1)
                if argmax:
                    vis_index = np.argmax(np.abs(channel_feat1 - channel_feat2))
                    info = 'max'
                else:
                    vis_index = np.argmin(np.abs(channel_feat1 - channel_feat2))
                    info = 'min'

                space_feat1, space_feat2 = feature_maps[angle1][vis_index, :, :], feature_maps[angle2][vis_index, :, :]

                handle = ax.matshow(space_feat1 - space_feat2)
                ax.set_title(str(angle1) + " - " + str(angle2) + f" channel: {vis_index + 1}")
                plt.colorbar(handle)

                plt.savefig(os.path.join(output_file, f'same_gap_vis_{angle1}_{angle2}_{info}_channel_{vis_index + 1}_space_reverse.png'), bbox_inches='tight', pad_inches=0.1)
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
    
    # plot_space_feat(feature_maps, vis_dir)
    # plot_same_gap_feat(feature_maps, vis_dir)
    # plot_max_channel_space_feat(feature_maps, vis_dir)
    # plot_same_gap_feat_show_level(feature_maps, vis_dir)
    plot_same_gap_feat_show_each_channel(feature_maps, vis_dir)
        