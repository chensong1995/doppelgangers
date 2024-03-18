import os.path as osp

import numpy as np
import torch

from .input_utils import read_image


class PairsDataset(torch.utils.data.Dataset):
    '''
    The Dataset class that loads images and masks from the disk. This class is
    adapted from read_image() in loftr_matches.py of the original Doppelgangers
    implementation.
    '''

    def __init__(self, data_path, pair_path, img_size=1024, df=8, padding=True):
        self.data_path = data_path
        self.pairs_info = np.load(pair_path, allow_pickle=True)
        self.img_size = img_size
        self.df = df
        self.padding = padding

    def __len__(self):
        return self.pairs_info.shape[0]

    def __getitem__(self, idx):
        name0, name1, _, _, _ = self.pairs_info[idx]

        img0_pth = osp.join(self.data_path, name0)
        img1_pth = osp.join(self.data_path, name1)
        img0_raw, mask0 = read_image(img0_pth, self.img_size, self.df, self.padding)
        img1_raw, mask1 = read_image(img1_pth, self.img_size, self.df, self.padding)

        # remove dummy batch dimension
        img0 = img0_raw.squeeze(axis=0)
        mask0 = mask0.squeeze(axis=0)
        img1 = img1_raw.squeeze(axis=0)
        mask1 = mask1.squeeze(axis=0)

        return {
            'idx': idx,
            'image0': img0,
            'image1': img1,
            'mask0': mask0,
            'mask1': mask1
        }
