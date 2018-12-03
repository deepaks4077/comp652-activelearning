from typing import Tuple, List, Union, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, SubsetRandomSampler
import numpy as np

class ModifiedDataset(Dataset):

    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index: int):
        data, target = self.dset[index]
        return data, target, index

    def __len__(self):
        return len(self.dset)

    def get_items(self, indices):
        images = torch.zeros(indices.shape[0], 1, 28, 28, dtype = torch.float32)
        labels = torch.zeros(indices.shape[0], 1, dtype = torch.long)

        for i, idx in enumerate(indices):
            images[i], labels[i], _ = self.__getitem__(idx)
        
        return images, labels, indices