import torch
from torch.utils.data import ConcatDataset
import numpy as np

class Vehicle:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset
        self.current_bs = -1
        self.previous_bs = -1


    def add_samples(self, newdataset):     
        self.dataset = ConcatDataset([self.dataset, newdataset])
