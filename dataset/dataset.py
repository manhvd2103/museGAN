import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import pypianoroll
from tqdm import tqdm
import random

class MuseGANDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    data = MuseGANDataset(root_dir="../data/cleaned_data")
    dataloader = DataLoader(dataset=data, batch_size=16, drop_last=True, shuffle=True)
    print(dataloader)

        

