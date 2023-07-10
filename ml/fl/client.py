import torch

from typing import Dict, Tuple, List, Union, Optional, Any
from collections import OrderedDict
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from ml.utils.train_utils import train, test
from ml.utils.helpers import get_std

from torch.utils.data import ConcatDataset

import numpy as np

class Client:
    def __init__(self, id):
        self.id = id
        self.vehicle_list = []
        self.IS = 0
        self.DQ = None
        self.dataset = None
        self.trainloader = None
        self.testloader = None
        self.model = None
        self.optimizer = None
        self.epochs = None
        self.lr = None
        self.criterion = None
        self.device = None
        self.test_size = None
        self.batch_size = None


    def init_learning_parameters(self, params: Dict[str, Union[bool, str, int, float]], model):
        """
        This function initializes the learning parameters
        of each  client.
        """
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.model = model
        self.device = params["device"]
        self.test_size = params['test_size']
        self.batch_size = params['batch_size']

        # Get Criterion
        from ml.utils.helpers import get_criterion
        self.criterion = get_criterion(params['criterion'])

        # Get Optimizer
        from ml.utils.helpers import get_optim
        self.optimizer = get_optim(model, params['optimizer'], self.lr)

        

    def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
        """
        Setting model parameters of the client
        """
        if not isinstance(parameters, torch.nn.Module):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.load_state_dict(parameters.state_dict(), strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """
        Getting model parameters of the client
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    

    def update(self): 
        """
        Perform local training for specified epochs.
        """    
        train_history = train(self.model,self.train_loader, self.device, self.criterion, self.optimizer, self.epochs,False)
    
    def evaluate(self, test_loader):
        """
        Evaluaate on local test set.
        """
        acc, f1 = test(self.model,test_loader,self.criterion, self.device)
        return acc, f1
    
    def register(self, vehicle):
        """
        Register a vehicle to the bs (client)
        """
        vehicle.current_bs = self.id
        self.vehicle_list.append(vehicle)
        self.IS = self.IS + 1
        

    def unregister(self, vehicle):
        """
        Unregister a vehicle from the bs (client)
        """
        for mycar in self.vehicle_list:
            if mycar.id == vehicle.id:
                self.vehicle_list.remove(vehicle)
        
        vehicle.previous_bs = self.id

        return vehicle

    def reconfirm(self, vehicle):
        """
        If the vehicle remains at the same bs
        reconfirm its presence.
        """
        for mycar in self.vehicle_list:
            if mycar.id == vehicle.id:
                mycar.previous_bs = mycar.current_bs

    def refresh(self):
        """
        Update client's data loaders after adding
        new clients.
        """
        self.dataset = self.vehicle_list[0].dataset
        for vehicle in self.vehicle_list[1:]:
            self.dataset = ConcatDataset([self.dataset, vehicle.dataset])

        # Train - Test Split
        train_set, val_set = random_split(self.dataset, [int(len(self.dataset)*(1 - self.test_size)), int(len(self.dataset)*self.test_size)])
        
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)

        self.DQ = get_std(self.train_loader)



        