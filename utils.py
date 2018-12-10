from typing import Tuple, List, Union, Dict
from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, SubsetRandomSampler
from ModifiedDataset import ModifiedTensorDataset
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_model(model, test_loader, NUM_TRIALS):
    #model.eval()
    model.eval()
    total_test_samples = test_loader.batch_size * len(test_loader)
    with torch.no_grad():
        correct = 0
        total = 0
        
        for idx, (images, labels, tl_ind, _) in enumerate(test_loader):
            images_temp = images
            for nt in range(0):
                images = torch.cat((images,images_temp), dim=0)
            
            images = images.to(device)
            labels = labels.to(device)
            dummy_lbls = torch.zeros(images.shape[0])

            images_dataset = TensorDataset(images, dummy_lbls)
            images_loader = torch.utils.data.DataLoader(dataset=images_dataset, batch_size=1000, shuffle=False)
            
            outputs = torch.FloatTensor([]).to(device)
            for i, (imgs, lbls) in enumerate(images_loader):
                output = model.forward(imgs)
                outputs = torch.cat((outputs, output), dim=0)

            #_, predicted = torch.max(outputs.data, 1)
            batch_size = test_loader.batch_size
            predicted = torch.LongTensor([]).to(device)
            for i in range(batch_size):
                torch.mean(outputs[i::batch_size], dim = 0)
                predicted = torch.cat((predicted, torch.argmax(torch.mean(outputs[i::batch_size], dim=0)).reshape(1)), dim=0 )

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print('Test Accuracy on the {} test images: {} %'.format(total_test_samples, test_accuracy))
    return test_accuracy

def run_experiment(train_dataset, test_dataset, test_loader, model, sampling_size, myselector, optimizer, criterion, exp_suffix, experiment, max_training_num, NUM_TRIALS, NUM_EPOCHS, learn_rate, random_bootstrap_samples, reset_model_per_selection):
    
    print("Running model with function: {}".format(exp_suffix))
    print("Sampling size = {}\n".format(sampling_size))

    sampling_set = torch.tensor([x for x in range(len(train_dataset))], dtype = torch.int)
    sampling_set = torch.tensor(list((set(sampling_set.tolist()) - set(random_bootstrap_samples))), dtype = torch.int)
    samples = []
    selected = torch.tensor([], dtype = torch.int)
    selected = torch.cat((selected, torch.tensor([x for x in random_bootstrap_samples], dtype = torch.int)), dim = 0)
    accuracy = []
    main_iter = 1

    inp, lbl, _, _ = train_dataset.get_items(selected)

    selection_ds = ModifiedTensorDataset(images = inp, labels = lbl)
    selection_dl = torch.utils.data.DataLoader(dataset=selection_ds, batch_size=1000, shuffle=True)

    for n_ep in range(20):
        for idx, (data, target) in enumerate(selection_dl):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target.reshape(-1))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    del inp, lbl, selection_ds, selection_dl

    if reset_model_per_selection:
        torch.save(model, 'bootstrap_model.ckpt')

    while len(sampling_set) != 0:

        print("\nIteration = {}, sample set size = {}".format(main_iter, len(sampling_set)))

        if(len(sampling_set) <= sampling_size):
            samples = sampling_set
        else:
            indices = torch.randperm(len(sampling_set))[:sampling_size].tolist()
            samples = sampling_set[indices]

        images, labels, _ , _= train_dataset.get_items(samples)

        images_temp = images
                
        for nt in range(NUM_TRIALS-1):
            images = torch.cat((images,images_temp), dim=0)

        images = images.to(device)
        #labels = labels.to(device)
        dummy_lbls = torch.zeros(images.shape[0])

        images_dataset = TensorDataset(images, dummy_lbls)
        images_loader = torch.utils.data.DataLoader(dataset=images_dataset, batch_size=1000, shuffle=False)

        with torch.no_grad():
            outputs = torch.FloatTensor([]).to(device)
            for i, (imgs, lbls) in enumerate(images_loader):
                output = model.forward(imgs)
                outputs = torch.cat((outputs, output), dim=0)

            training_indices = myselector.select(samples, outputs, images_temp.to(device)) 

        del images, outputs, images_temp

        # Contacatenate current and previous selections
        inps1, labelz1, _, isfake = train_dataset.get_items(training_indices)
        inps2, labelz2, _, _ = train_dataset.get_items(selected)
        inps = torch.cat((inps1, inps2), dim = 0)
        labelz = torch.cat((labelz1, labelz2), dim = 0)
        selected = torch.cat((selected, training_indices), dim = 0)
        
        num_fakes = len(np.where(isfake == 0)[0])    

        selection_dataset = ModifiedTensorDataset(images = inps, labels = labelz)
        selection_dataloader = torch.utils.data.DataLoader(dataset=selection_dataset, batch_size=1000, shuffle=True)
        
        if reset_model_per_selection:
            model = torch.load('bootstrap_model.ckpt')
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

        for n_ep in range(NUM_EPOCHS):
            for idx, (data, target) in enumerate(selection_dataloader):
                data = data.to(device)
                target = target.to(device)

                outputs = model(data)
                loss = criterion(outputs, target.reshape(-1))
            
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        accuracy = test_model(model, test_loader, NUM_TRIALS)

        experiment.log_metric("acc_{}".format(exp_suffix), accuracy, main_iter)
        experiment.log_metric("isfake_proportion_{}".format(exp_suffix), num_fakes / training_indices.shape[0], main_iter)

        print("isfake_proportion_ = {}".format(num_fakes))

        if len(selected) == max_training_num:
            return accuracy

        sampling_set = torch.tensor(list((set(sampling_set.tolist()) - set(training_indices.tolist()))), dtype = torch.int)

        main_iter += 1

    return accuracy
