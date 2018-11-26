#from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import cPickle

import torch
import torchvision
import torchvision.transforms as transforms

import random

#pytorch stuff
class mnist():
	def __init__(self, batch_size):
		# MNIST dataset
		self.num_classes = 10
		self.batch_size = batch_size

		self.train_dataset = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
		self.test_dataset = torchvision.datasets.MNIST(root='../../data/', train=False, transform=transforms.ToTensor())

		# Data loader
		self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

	def get_train_dataset_subset_loader(self, indices):
		#self.train_dataset2 = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
		train_dataset_subset = torch.utils.data.Subset(self.train_dataset, indices)
		train_dataset_subset_loader = torch.utils.data.DataLoader(dataset=train_dataset_subset, batch_size=self.batch_size, shuffle=True)

		return train_dataset_subset_loader

	def get_random_subset_train_dataset_loader(self, num_datapoints):
		random_indices = random.sample(range(0, len(self.train_dataset)), num_datapoints)
		random_subset_train_dataset_loader = self.get_train_dataset_subset_loader(random_indices)
		return random_subset_train_dataset_loader


class cifar10():
	def __init__(self, batch_size):
		# MNIST dataset
		self.num_classes = 10
		self.batch_size = batch_size

		self.train_dataset = torchvision.datasets.CIFAR10(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
		self.test_dataset = torchvision.datasets.CIFAR10(root='../../data/', train=False, transform=transforms.ToTensor())

		# Data loader
		self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

	def get_train_dataset_subset_loader(self, indices):
		#self.train_dataset2 = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
		train_dataset_subset = torch.utils.data.Subset(self.train_dataset, indices)
		train_dataset_subset_loader = torch.utils.data.DataLoader(dataset=train_dataset_subset, batch_size=self.batch_size, shuffle=True)

		return train_dataset_subset_loader

	def get_random_subset_train_dataset_loader(self, num_datapoints):
		random_indices = random.sample(range(0, len(self.train_dataset)), num_datapoints)
		random_subset_train_dataset_loader = self.get_train_dataset_subset_loader(random_indices)
		return random_subset_train_dataset_loader