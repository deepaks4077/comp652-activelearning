from typing import Tuple, List, Union, Dict
import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AcquisitionFunction():
    def __init__(self, selection_size: int, name = "random"):
        self.selection_size = selection_size
        self.softmax_fn = nn.Softmax(dim = 1)
        self.name = name
    
    def get_best_sample(self, inp_ids, out, inp = None):
        raise NotImplementedError

class Random(AcquisitionFunction):

    name = "Random"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)        
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            res = input_ids[:self.selection_size]

        return res

# class Max_Min_Softmax(AcquisitionFunction):

#     name = "max_min"
#     def get_best_sample(self, input_ids, outputs, inputs = None):
#         sample_size = len(input_ids)
            
#         res = []
#         if (sample_size <= self.selection_size):
#             res = input_ids
#         else:
#             sm_outputs = self.softmax_fn(outputs)
#             output_confidence = torch.max(sm_outputs, dim=1)[0] - torch.min(sm_outputs, dim=1)[0]
#             _, sorted_indices = torch.sort(output_confidence)
#             res = input_ids[sorted_indices.tolist()[:self.selection_size]]
            
#         return res

class Smallest_Margin(AcquisitionFunction):

    name = "Smallest_Margin"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
        
        mean_outputs = torch.FloatTensor([]).to(device)
        for i in range(sample_size):
            mean_outputs = torch.cat((mean_outputs, torch.mean(outputs[i::sample_size], dim=0).reshape(1, -1)), dim=0)

        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            sm_outputs = self.softmax_fn(mean_outputs)
            sm_outputs_top_k = torch.topk(sm_outputs, 2)[0]
            output_confidence = sm_outputs_top_k[:, 0] - sm_outputs_top_k[:, 1]
            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[:self.selection_size]]
            
        return res

# class Entropy(AcquisitionFunction):

#     name = "entropy"
#     def get_best_sample(self, input_ids, outputs, inputs = None):
#         sample_size = len(input_ids)
            
#         res = []
#         if (sample_size <= self.selection_size):
#             res = input_ids
#         else:
#             sm_outputs = self.softmax_fn(outputs)
#             output_confidence = torch.sum(sm_outputs * torch.log(sm_outputs), dim=1)
#             _, sorted_indices = torch.sort(output_confidence)
#             res = input_ids[sorted_indices.tolist()[:self.selection_size]]
            
#         return res

class Density_Max(AcquisitionFunction):

    name = "Density_Max"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)

        mean_outputs = torch.tensor([], dtype = torch.float32).to(device)

        for i in range(sample_size):
            mean_outputs = torch.cat((mean_outputs, torch.mean(outputs[i::sample_size], dim=0).reshape(1, -1)), dim=0)

        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            normalized_input_matrix = nn.functional.normalize(inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]), p=2, dim=1)
            sm_outputs = self.softmax_fn(mean_outputs)
            output_uncertainty = 1 - torch.max(sm_outputs, dim=1)[0]
            output_confidence = torch.mean(torch.mm(normalized_input_matrix, normalized_input_matrix.transpose(dim0=0, dim1=1)), dim=0) * output_uncertainty

            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]
            
        return res

class Density_Entropy(AcquisitionFunction):

    name = "Density_Entropy"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
        
        mean_outputs = torch.FloatTensor([]).to(device)
        for i in range(sample_size):
            mean_outputs = torch.cat((mean_outputs, torch.mean(outputs[i::sample_size], dim=0).reshape(1, -1)), dim=0)            
        
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            normalized_input_matrix = nn.functional.normalize(inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]), p=2, dim=1)
            sm_outputs = self.softmax_fn(mean_outputs)
            output_uncertainty = -torch.sum(sm_outputs * torch.log(sm_outputs), dim=1)
            output_confidence = torch.mean(torch.mm(normalized_input_matrix, normalized_input_matrix.transpose(dim0=0, dim1=1)), dim=0) * output_uncertainty

            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]
            
        return res


#STOCHASTIC NETWORK BASED STUFF
class SN_Entropy(AcquisitionFunction):
    name = "SN_Entropy"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
        outputs = self.softmax_fn(outputs)
        
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            mean_outputs = torch.FloatTensor([]).to(device)
            for i in range(sample_size):
                mean_outputs = torch.cat((mean_outputs, torch.mean(outputs[i::sample_size], dim=0).reshape(1, -1)), dim=0)

            output_confidence = -torch.sum(mean_outputs * torch.log(mean_outputs), dim=1)
            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]

        return res

class SN_BALD(AcquisitionFunction):
    name = "SN_BALD"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
        outputs = self.softmax_fn(outputs)

        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            #find entropy of means
            mean_outputs = torch.FloatTensor([]).to(device)
            for i in range(sample_size):
                mean_outputs = torch.cat((mean_outputs, torch.mean(outputs[i::sample_size], dim=0).reshape(1, -1)), dim=0)
            output_confidence = -torch.sum(mean_outputs * torch.log(mean_outputs), dim=1)

            #find mean of entropies
            outputs = -torch.sum(outputs * torch.log(outputs), dim=1)
            mean_outputs = torch.FloatTensor([]).to(device)
            for i in range(sample_size):
                mean_outputs = torch.cat((mean_outputs, torch.mean(outputs[i::sample_size], dim=0).reshape(1)), dim=0)
            output_confidence = output_confidence - mean_outputs

            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]

        return res


class Variation_Ratios(AcquisitionFunction):
    name = "Variation_Ratios"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
        outputs = self.softmax_fn(outputs)

        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            mean_outputs = torch.FloatTensor([]).to(device)
            for i in range(sample_size):
                mean_outputs = torch.cat((mean_outputs, torch.mean(outputs[i::sample_size], dim=0).reshape(1, -1)), dim=0)

            output_confidence = 1 - torch.max(mean_outputs, dim=1)[0]
            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]

        return res


class Mean_STD(AcquisitionFunction):
    name = "Mean_STD"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
        outputs = self.softmax_fn(outputs)

        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            std_outputs = torch.FloatTensor([]).to(device)
            for i in range(sample_size):
                std_outputs = torch.cat((std_outputs, (torch.mean(outputs[i::sample_size]**2, dim=0) - torch.mean(outputs[i::sample_size], dim=0)**2).reshape(1, -1)**0.5), dim=0)

            output_confidence = torch.mean(std_outputs, dim=1)
            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]

        return res
