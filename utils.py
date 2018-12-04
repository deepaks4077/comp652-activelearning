from typing import Tuple, List, Union, Dict
from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import Dataset, SubsetRandomSampler
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_model(model, test_loader):
    model.eval()		
    total_test_samples = test_loader.batch_size * len(test_loader)
    with torch.no_grad():
        correct = 0
        total = 0
        
        for idx, (images, labels, tl_ind, _) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print('Test Accuracy on the {} test images: {} %'.format(total_test_samples, test_accuracy))
    return test_accuracy

def run_experiment(train_dataset, test_dataset, test_loader, model, sampling_size, myselector, optimizer, criterion, exp_suffix, experiment):
    
    print("Running model with function: {}".format(exp_suffix))
    print("Sampling size = {}\n".format(sampling_size))
    sampling_set = torch.tensor([x for x in range(len(train_dataset))], dtype = torch.int)
    samples = []
    selected = torch.tensor([], dtype = torch.int)
    accuracy = []
    i = 1
    while len(sampling_set) != 0:

        print("\nIteration = {}, sample set size = {}".format(i, len(sampling_set)))

        if(len(sampling_set) < sampling_size):
            samples = sampling_set
        else:
            indices = torch.randperm(len(sampling_set))[:sampling_size].tolist()
            samples = sampling_set[indices]

        images, labels, _ , _= train_dataset.get_items(samples)
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model.forward(images)
            training_indices = myselector.select(samples, outputs, images)

        # Contacatenate current and previous selections
        inps1, labelz1, _, isfake = train_dataset.get_items(training_indices)
        inps2, labelz2, _, _ = train_dataset.get_items(selected)
        inps = torch.cat((inps1, inps2), dim = 0)
        labelz = torch.cat((labelz1, labelz2), dim = 0)
        selected = torch.cat((selected, training_indices), dim = 0)
        
        num_fakes = len(np.where(isfake == 0)[0])

        # Set model totrain
        model.train()
        
        # Forward pass
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

        experiment.log_metric("acc_{}".format(exp_suffix), accuracy, i)
        experiment.log_metric("isfake_proportion_{}".format(exp_suffix), num_fakes / training_indices.shape[0], i)

        print("isfake_proportion_ = {}".format(num_fakes))

        sampling_set = torch.tensor(list((set(sampling_set.tolist()) - set(training_indices.tolist()))), dtype = torch.int)

        i += 1

    return accuracy
