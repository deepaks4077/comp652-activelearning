# stateful subset sampler that does not return selected indices
# dataloader that iterates until the data has been exhausted
# model that runs with no grad and then with grad
# selector that uses the acquisition function to select indices for training

# selector -> init with acquisition function, run with model outputs
# acquisition function -> empty init / threshold values?, run with model outputs
# dataloder -> standard data loader

# Import comet_ml in the top of your file

from os import listdir
from os.path import isfile, join

from comet_ml import Experiment
from typing import Tuple, List, Union, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import AcquisitionFunctions
from AcquisitionFunctions import Max_Min_Softmax, Random
from ModifiedDataset import ModifiedDataset, TorchDataset
from Selector import Selector
from models import convnet_mnist
from utils import test_model, run_experiment

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_TRIALS = 10

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)

dtsets = []
prefix = "./fake_data/"
fake_subset_indices = [x for x in range(3000)]
real_subset_indices = [x for x in range(30000)]
dtsets = [ModifiedDataset(Subset(TorchDataset(join(prefix, f)), fake_subset_indices), fake = True) for f in listdir(prefix) if isfile(join(prefix, f))]

mnist_dataset = ModifiedDataset(Subset(datasets.MNIST(root='./data/', train=True, 
							transform=transforms.Compose([
								transforms.Resize(64),
								transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True), real_subset_indices), fake = False)

dtsets.append(mnist_dataset)
train_dataset = ModifiedDataset(ConcatDataset(dtsets))

# train_dataset = ModifiedDataset(datasets.MNIST(root='./data/', \
#                                                 train=True, \
#                                                 transform=transforms.Compose([ \
# 							transforms.Resize(64), \
# 							transforms.ToTensor(), \
# 							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), \
#                                                 download=True))

test_dataset = ModifiedDataset(datasets.MNIST(root='./data/', 
                                                train=False, 
                                                transform=transforms.Compose([ \
							transforms.Resize(64), \
							transforms.ToTensor(), \
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
criterion = nn.CrossEntropyLoss()

hyper_params = {"learning_rate": 0.001, "sampling_size": 2000, "selection_size": 500}
experiment = Experiment(api_key="Gncqbz3Rhfy3MZJBcX7xKVJoo", project_name="general", workspace="deepak-sharma-mail-mcgill-ca")
experiment.log_multiple_params(hyper_params)

# myfunctions = [AcquisitionFunctions.Density_Max,
#                 AcquisitionFunctions.Density_Entropy, Max_Min_Softmax, AcquisitionFunctions.Smallest_Margin,
#                 AcquisitionFunctions.Entropy, AcquisitionFunctions.Random]

myfunctions = [AcquisitionFunctions.Density_Max]


# mynames = ["density_max", "density_entropy", "max_min", "smallest_margin", "entropy", "random"]

mynames = ["density_max"]

for j in range(len(myfunctions)):
    print("J = {}".format(j))
    print("Name = {}".format(mynames[j]))
    model = convnet_mnist(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= hyper_params["learning_rate"])
    myselector = Selector(myfunctions[j](selection_size = hyper_params["selection_size"]))
    acc_random = run_experiment(train_dataset, test_dataset, test_loader,  model, hyper_params["sampling_size"], myselector, optimizer, criterion, mynames[j], experiment, NUM_TRIALS)
