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
from torch.utils.data import Dataset, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

# Create an experiment

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Report any information you need by:
hyper_params = {"learning_rate": 0.001, "sample_size": 1000, "batch_size": 50, "acquisition_func": "Max_Min_Softmax"}
experiment.log_multiple_params(hyper_params)

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

class AcquisitionFunction():
    def __init__(self, selection_size: int):
        self.selection_size = selection_size
        self.softmax_fn = nn.Softmax()
    
    def get_best_sample(self, inp, out):
        raise NotImplementedError

class Random(AcquisitionFunction):
    def get_best_sample(self, input_ids, outputs):
        print(input_ids.shape)
        sample_size = len(input_ids)        
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            res = input_ids[:self.selection_size]

        print(res.shape)
        return res

class Max_Min_Softmax(AcquisitionFunction):
    def get_best_sample(self, input_ids, outputs):
        sample_size = len(input_ids)
            
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            sm_outputs = self.softmax_fn(outputs)
            output_confidence = torch.max(sm_outputs, dim=1)[0] - torch.min(sm_outputs, dim=1)[0]
            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[:self.selection_size]]
            
        return res

class Smallest_Margin(AcquisitionFunction):
    def get_best_sample(self, input_ids, outputs):
        sample_size = len(input_ids)
            
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            sm_outputs = self.softmax_fn(outputs)
            sm_outputs_top_k = torch.topk(sm_outputs, 2)[0]
            output_confidence = sm_outputs_top_k[0] - sm_outputs_top_k[1]
            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[:self.selection_size]]
            
        return res

class Entropy(AcquisitionFunction):
    def get_best_sample(self, input_ids, outputs):
        sample_size = len(input_ids)
            
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            sm_outputs = self.softmax_fn(outputs)
            output_confidence = torch.sum(sm_outputs * torch.log(sm_outputs), dim=1)
            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[:self.selection_size]]
            
        return res

class Density_Max(AcquisitionFunction):
    def get_best_sample(self, input_ids, outputs):
        sample_size = len(input_ids)
            
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            normalized_outputs_matrix = nn.functional.normalize(outputs.reshape(outputs.shape[0], outputs.shape[1] * outputs.shape[2] * outputs.shape[3]), p=2, dim=1)
            sm_outputs = self.softmax_fn(outputs)
            output_uncertainty = 1 - torch.max(sm_outputs, dim=1)[0]
            output_confidence = torch.mean(torch.mm(normalized_outputs_matrix, normalized_outputs_matrix.transpose(dim0=0, dim1=1)), dim=0) * output_uncertainty

            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]
            
        return res

class Density_Entropy(AcquisitionFunction):
    def get_best_sample(self, input_ids, outputs):
        sample_size = len(input_ids)
            
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            normalized_outputs_matrix = nn.functional.normalize(outputs.reshape(outputs.shape[0], outputs.shape[1] * outputs.shape[2] * outputs.shape[3]), p=2, dim=1)
            sm_outputs = self.softmax_fn(outputs)
            output_uncertainty = -torch.sum(sm_outputs * torch.log(sm_outputs), dim=1)
            output_confidence = torch.mean(torch.mm(normalized_outputs_matrix, normalized_outputs_matrix.transpose(dim0=0, dim1=1)), dim=0) * output_uncertainty

            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]
            
        return res

class Selector():
    def __init__(self, func: AcquisitionFunction):
        self.func = func

    def select(self, input_ids, outputs):
        return self.func.get_best_sample(input_ids, outputs)

class convnet_mnist(nn.Module):
    def __init__(self, num_classes):
        super(convnet_mnist, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, self.num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def test_model(model, test_loader):
    model.eval()		
    total_test_samples = test_loader.batch_size
    with torch.no_grad():
        correct = 0
        total = 0
        
        for idx, (images, labels, tl_ind) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print('Test Accuracy on the {} test images: {} %'.format(total_test_samples, test_accuracy))
    return test_accuracy

train_dataset = ModifiedDataset(datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True))
test_dataset = ModifiedDataset(datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor()))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

model = convnet_mnist(10).to(device)
myfunction = Max_Min_Softmax(selection_size = 50)
myselector = Selector(myfunction)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def run_experiment(train_dataset, test_dataset, model, myselector, optimizer, criterion, exp_suffix):
    sampling_set = torch.tensor([x for x in range(len(train_dataset))], dtype = torch.int)
    sampling_size = 1000
    samples = []
    selected = torch.tensor([], dtype = torch.int)
    while len(sampling_set) != 0:
        if(len(sampling_set) < sampling_size):
            samples = sampling_set
        else:
            indices = torch.randperm(len(sampling_set))[:sampling_size].tolist()
            samples = sampling_set[indices]

        images, labels, _ = train_dataset.get_items(samples)
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            training_indices = myselector.select(samples, outputs)

        selected = torch.cat((selected, training_indices), dim = 0)
        model.train()
        
        # Forward pass
        inps, labelz, _ = train_dataset.get_items(selected)
        inps = inps.to(device)
        labelz = labelz.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels.reshape(-1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        accuracy = test_model(model, test_loader)

        experiment.log_metric("acc_{}".format(exp_suffix), accuracy)

        sampling_set = torch.tensor(list((set(sampling_set.tolist()) - set(samples.tolist()))), dtype = torch.int)

run_experiment(train_dataset, test_dataset, model, myselector, optimizer, criterion, "max_min")

model = convnet_mnist(10).to(device)
myfunction = Random(selection_size = 50)
myselector = Selector(myfunction)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

run_experiment(train_dataset, test_dataset, model, myselector, optimizer, criterion, "random")