import torch

from typing import Dict, Tuple, List, Union, Optional, Any
from collections import OrderedDict
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import numpy as np

class Client:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset
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


    def init_parameters(self, params: Dict[str, Union[bool, str, int, float]]):  # default parameters
        self.epochs = params["epochs"]
        self.optimizer = params["optimizer"]
        self.lr = params["lr"]
        self.criterion = params["criterion"]
        self.device = params["device"]
        self.test_size = params['test_size']
        self.batch_size = params['batch_size']

        # Train - Test Split
        train_set, val_set = random_split(self.dataset, (len(self.dataset)*(1 - self.test_size), len(self.dataset)*self.test_size))
        
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)

        

    def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
        if not isinstance(parameters, torch.nn.Module):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.load_state_dict(parameters.state_dict(), strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    

    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                optimizer.step()

                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
        return mm.results
    
    def evaluate(self):
        if self.args._train_only: # `args.test_fraction` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.test_set))
        return mm.results
