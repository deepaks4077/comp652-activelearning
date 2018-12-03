# stateful subset sampler that does not return selected indices
# dataloader that iterates until the data has been exhausted
# model that runs with no grad and then with grad
# selector that uses the acquisition function to select indices for training

# selector -> init with acquisition function, run with model outputs
# acquisition function -> empty init / threshold values?, run with model outputs
# dataloder -> standard data loader

# Import comet_ml in the top of your file
from comet_ml import Experiment
from typing import Tuple, List, Union, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import AcquisitionFunctions
from AcquisitionFunctions import Max_Min_Softmax, Random
from ModifiedDataset import ModifiedDataset
from Selector import Selector
from models import convnet_mnist
from utils import test_model, run_experiment

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)



train_dataset = ModifiedDataset(datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True))
test_dataset = ModifiedDataset(datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor()))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
criterion = nn.CrossEntropyLoss()

hyper_params = {"learning_rate": 0.001, "sampling_size": 2000, "selection_size": 500}
experiment = Experiment(api_key="Gncqbz3Rhfy 3MZJBcX7xKVJoo", project_name="general", workspace="deepak-sharma-mail-mcgill-ca")
experiment.log_multiple_params(hyper_params)

myfunctions = [AcquisitionFunctions.Density_Max,
                AcquisitionFunctions.Density_Entropy, Max_Min_Softmax, AcquisitionFunctions.Smallest_Margin,
                AcquisitionFunctions.Entropy, AcquisitionFunctions.Random]

mynames = ["density_max", "density_entropy", "max_min", "smallest_margin", "entropy", "random"]

for j in range(len(myfunctions)):
    print("J = {}".format(j))
    print("Name = {}".format(mynames[j]))
    model = convnet_mnist(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= hyper_params["learning_rate"])
    myselector = Selector(myfunctions[j](selection_size = hyper_params["selection_size"]))
    acc_random = run_experiment(train_dataset, test_dataset, test_loader,  model, hyper_params["sampling_size"], myselector, optimizer, criterion, mynames[j], experiment)