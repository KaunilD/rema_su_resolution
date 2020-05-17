import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class REMADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, input_pth, target_pth, transform=None):
        self.pairs = []

        self.input_pth = input_pth
        self.target_pth = target_pth
        self.transform = transform
        self.gen_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
