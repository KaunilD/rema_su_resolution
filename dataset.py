import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

Image.MAX_IMAGE_PIXELS = None

class REMADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, input_pth, target_pth, scale_factor=2, patch_size=256, transform=None):
        self.pairs = []

        self.input_pth = input_pth
        self.target_pth = target_pth
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.transform = transform

        self.gen_pairs()

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def pad_image(image_in, total_stride):
        h, w, c = image_in.shape

        nh = total_stride * GTiffDataset.i_div_up(h, total_stride)
        nw = total_stride * GTiffDataset.i_div_up(w, total_stride)
        image_out = np.zeros((nh, nw, c))
        image_out[:h, :w, :] = image_in
        return image_out

    def gen_pairs(self):

        input = Image.open(self.input_pth)
        target = Image.open(self.target_pth)

        input = REMADataset.pad_image(input, self.patch_size)
        target = REMADataset.pad_image(target, self.patch_size)

        input = np.asarray(input, dtype=np.uint8)
        target = np.asarray(target, dtype=np.uint8)
        # 0 256 510
        #0 512 1024

        h, w, c = image_in.shape

        for i in range(0, h, self.patch_size):
            if i+self.patch_size > h:
                break
                for j in range(0, w, self.patch_size):
                if j + self.patch_size > w:
                    break
                inp = input[i:i+patch_size, j:j+patch_size]
                tar = target[i*2:i*2+patch_size*2, j*2:j*2:patch_size*2]
                



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform:
            sample = self.transform(sample)

        return sample


def create_args():
    parser = argparse.ArgumentParser(
        description="Driver code for REMADataset"
    )

    parser.add_argument(
        "--inp",
        type=str,
        help="input DEM.",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="superresolved target image",
    )

    parser.add_argument(
        "--upfactor",
        type=str,
        help="su target",
    )


    return parser.parse_args()




if __name__=="__main__":
    args = create_args()

    dataset = REMADataset(args.inp, args.out)
