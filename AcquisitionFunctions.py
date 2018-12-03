from typing import Tuple, List, Union, Dict
import torch
import torch.nn as nn
import numpy as np

class AcquisitionFunction():
    def __init__(self, selection_size: int, name = "random"):
        self.selection_size = selection_size
        self.softmax_fn = nn.Softmax(dim = 1)
        self.name = name
    
    def get_best_sample(self, inp_ids, out, inp = None):
        raise NotImplementedError

class Random(AcquisitionFunction):

    name = "random"
    def get_best_sample(self, input_ids, outputs, inputs = None):
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

    name = "max_min"
    def get_best_sample(self, input_ids, outputs, inputs = None):
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

    name = "smallest_margin"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
            
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            sm_outputs = self.softmax_fn(outputs)
            sm_outputs_top_k = torch.topk(sm_outputs, 2)[0]
            output_confidence = sm_outputs_top_k[:, 0] - sm_outputs_top_k[:, 1]
            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[:self.selection_size]]
            
        return res

class Entropy(AcquisitionFunction):

    name = "entropy"
    def get_best_sample(self, input_ids, outputs, inputs = None):
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

    name = "density_max"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
            
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            normalized_input_matrix = nn.functional.normalize(inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]), p=2, dim=1)
            sm_outputs = self.softmax_fn(outputs)
            output_uncertainty = 1 - torch.max(sm_outputs, dim=1)[0]
            output_confidence = torch.mean(torch.mm(normalized_input_matrix, normalized_input_matrix.transpose(dim0=0, dim1=1)), dim=0) * output_uncertainty

            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]
            
        return res

class Density_Entropy(AcquisitionFunction):

    name = "density_entropy"
    def get_best_sample(self, input_ids, outputs, inputs = None):
        sample_size = len(input_ids)
            
        res = []
        if (sample_size <= self.selection_size):
            res = input_ids
        else:
            normalized_input_matrix = nn.functional.normalize(inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]), p=2, dim=1)
            sm_outputs = self.softmax_fn(outputs)
            output_uncertainty = -torch.sum(sm_outputs * torch.log(sm_outputs), dim=1)
            output_confidence = torch.mean(torch.mm(normalized_input_matrix, normalized_input_matrix.transpose(dim0=0, dim1=1)), dim=0) * output_uncertainty

            _, sorted_indices = torch.sort(output_confidence)
            res = input_ids[sorted_indices.tolist()[-self.selection_size:]]
            
        return res