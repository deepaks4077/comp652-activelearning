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
from ModifiedDataset import ModifiedDataset, TorchDataset
from Selector import Selector
from models import convnet_mnist
from utils import test_model, run_experiment
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(7)
random.seed(42)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)

dtsets = []
prefix = "./fake_data/"
input_size = 28
fake_subset_indices = [x for x in range(3000)]
real_subset_indices = [x for x in range(30000)]



mytransforms = [transforms.Normalize((-1.,-1.,-1.),(2.,2.,2.)), transforms.ToPILImage(), transforms.Resize(input_size), transforms.ToTensor()]
dtsets = [ModifiedDataset(Subset(TorchDataset(join(prefix, f), transforms = transforms.Compose(mytransforms)), fake_subset_indices), fake = True, input_size = input_size) for f in listdir(prefix) if isfile(join(prefix, f))]

mnist_dataset = ModifiedDataset(Subset(datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True), real_subset_indices), fake = False, input_size = input_size)

dtsets.append(mnist_dataset)
train_dataset = ModifiedDataset(ConcatDataset(dtsets))

test_dataset = ModifiedDataset(datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor()))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
criterion = nn.CrossEntropyLoss()


#Contains most of the parameters needed for an experiment
#CONFIGURABLE STUFF
NUM_TRIALS = 10
max_training_num = 5020
hyper_params = {"learning_rate": 0.001, "sampling_size": int(len(train_dataset)/6), "selection_size": 250, "max_training_num": max_training_num, "NUM_EPOCHS": 1, "bootstrap_samplesize": 20, "reset_model_per_selection": False}



experiment = Experiment(api_key="Gncqbz3Rhfy3MZJBcX7xKVJoo", project_name="amit", workspace="deepak-sharma-mail-mcgill-ca")
#experiment = Experiment(api_key="wD61O90Y34KruZHAyHH7bqWRw", project_name="general", workspace="amitfishy")
experiment.log_parameters(hyper_params)

myfunctions = [AcquisitionFunctions.Random, AcquisitionFunctions.Smallest_Margin, AcquisitionFunctions.Density_Max,AcquisitionFunctions.Density_Entropy, AcquisitionFunctions.SN_Entropy, AcquisitionFunctions.SN_BALD, AcquisitionFunctions.Variation_Ratios, AcquisitionFunctions.Mean_STD]
#myfunctions = [AcquisitionFunctions.Mean_STD]
random_bootstrap_samples = random.sample(range(0, len(train_dataset)), hyper_params["bootstrap_samplesize"])

for j in range(len(myfunctions)):
    print()
    print('-------------------------------------------------------')
    print("Processing function {}".format(j+1))
    print("Name = {}".format(myfunctions[j].name))
    model = convnet_mnist(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= hyper_params["learning_rate"])
    myselector = Selector(myfunctions[j](selection_size = hyper_params["selection_size"]))
    acc_random = run_experiment(train_dataset, test_dataset, test_loader,  model, hyper_params["sampling_size"], myselector, optimizer, criterion, myfunctions[j].name, experiment, max_training_num, NUM_TRIALS, hyper_params["NUM_EPOCHS"], hyper_params["learning_rate"], random_bootstrap_samples, hyper_params["reset_model_per_selection"])
