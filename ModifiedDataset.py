from typing import Tuple, List, Union, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, SubsetRandomSampler, ConcatDataset
import numpy as np

class ModifiedDataset(Dataset):

    def __init__(self, dset, fake = False, input_size = 28):
        self.dset = dset
        self.input_size = input_size

        if fake:
            self.isfake = 0
        else:
            self.isfake = 1

        self.ismodified = type(self.dset) == type(self)

    def __getitem__(self, index: int):
        
        if self.ismodified or type(self.dset) is ConcatDataset:
            return self.dset[index]
       
        data, target = self.dset[index]
        
        return data, target, index, torch.tensor([self.isfake], dtype = torch.uint8)

    def __len__(self):
        return len(self.dset)

    def get_items(self, indices):
        images = torch.zeros(indices.shape[0], 1, self.input_size, self.input_size, dtype = torch.float32)
        labels = torch.zeros(indices.shape[0], 1, dtype = torch.long)
        isfake = torch.zeros(indices.shape[0], 1, dtype = torch.uint8)

        for i, idx in enumerate(indices):
            images[i], labels[i], _ , isfake[i] = self.__getitem__(idx)
        
        return images, labels, indices, isfake

class TorchDataset(Dataset):

    def __init__(self, name, transforms = None):
        self.name = name
        self.dset = torch.load(name)
        self.transforms = transforms

    def __getitem__(self, index: int):
        data, target = self.dset[0][index], self.dset[1][index]
       
        if self.transforms is not None:
            data = self.transforms(data.cpu().view(1, data.shape[0], data.shape[1]))
           
        return data.view(data.shape[1], data.shape[2]), target

    def __len__(self):
        return len(self.dset[1])

class ModifiedTensorDataset(Dataset):

	def __init__(self, images, labels):

		self.images = images
		self.labels = labels

	def __getitem__(self, index: int):
		data, target = self.images[index], self.labels[index] 
		return data, target

	def __len__(self):
		return self.images.shape[0]
