import os
import cv2
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

Image.MAX_IMAGE_PIXELS = None

class REMADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, input_pth, target_pth, split, scale_factor=2, patch_size=256, transform=None):
        self.input_pth = input_pth
        self.target_pth = target_pth
        self.split = split
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.transform = transform

        self.pairs = self.gen_pairs()
        a = int(len(self.pairs)*self.split)
        self.pairs = self.pairs[:a]

    def __len__(self):
        return len(self.pairs)
    def gen_pairs(self):
        inputs, targets   = [], []
        image   = Image.open(self.input_pth)
        target  = Image.open(self.target_pth)
        
        image   = np.asarray(image, dtype=np.uint8)
        target  = np.asarray(target, dtype=np.uint8)
        
        h, w    = image.shape
        counter = 0
        for i in range(0, h, self.patch_size):
            if (i+self.patch_size) > h:
                break
            for j in range(0, w, self.patch_size):
                if (j + self.patch_size) > w:
                    break
                inp = image[i:i+self.patch_size, j:j+self.patch_size]
                inputs.append(inp)
                cv2.imwrite("debug/{}_inp.png".format(counter), inp)
                counter+=1

        h, w    = target.shape
        counter = 0
        for i in range(0, h, self.patch_size*self.scale_factor):
            if (i+self.patch_size*self.scale_factor) > h:
                break
            for j in range(0, w, self.patch_size*self.scale_factor):
                if (j + self.patch_size*self.scale_factor) > w:
                    break
                tar = target[i:i+self.patch_size*self.scale_factor, j:j+self.patch_size*self.scale_factor]
                targets.append(tar)
                cv2.imwrite("debug/{}_tar.png".format(counter), tar)
                counter+=1
 
        assert(len(inputs)==len(targets))
        return list(zip(inputs, targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform:
            inp, out = self.pairs[idx]
            sample = [self.transform(inp), self.transform(out)]

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
