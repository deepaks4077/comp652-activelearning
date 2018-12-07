from typing import Tuple, List, Union, Dict
import torch
import torch.nn as nn
import numpy as np
from AcquisitionFunctions import AcquisitionFunction

class Selector():
    def __init__(self, func: AcquisitionFunction):
        self.func = func

    def select(self, input_ids, outputs, inputs, NUM_TRIALS):
        return self.func.get_best_sample(input_ids, outputs, inputs, NUM_TRIALS)