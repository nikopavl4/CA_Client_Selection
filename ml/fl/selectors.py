import random
import torch.nn as nn

class RandomSelector:
    def __init__(self, fraction):
        self.fraction = fraction


    def sample_clients(self, client_list):  # default parameters
        available_clients = client_list
        if len(available_clients) == 0:
            print(f"Cannot sample clients. The number of available clients is zero.")
            return []
        num_selection = int(self.fraction * len(available_clients))
        if num_selection == 0:
            num_selection = 1
        if num_selection > len(available_clients):
            num_selection = len(available_clients)
        sampled_clients = random.sample(available_clients, num_selection)
        print(f"Parameter c={self.fraction}. Sampled {num_selection} client(s): {[cl.id for cl in sampled_clients]}")
        return sampled_clients
    
class L2Selector:
    def __init__(self, fraction):
        self.fraction = fraction
        self.dist_function = nn.MSELoss()


    def sample_clients(self, client_list):  # default parameters
        available_clients = client_list
        for cl in available_clients:
            for param_tensor in cl.state_dict():
                loss = self.dist_function(cl.model.state_dict()[param_tensor], )


        return sampled_clients
    

class CASelector:
    def __init__(self, fraction):
        self.fraction = fraction


    def sample_clients(self, client_list):  # default parameters
        available_clients = client_list
        if len(available_clients) == 0:
            print(f"Cannot sample clients. The number of available clients is zero.")
            return []
        num_selection = int(self.fraction * len(available_clients))
        if num_selection == 0:
            num_selection = 1
        if num_selection > len(available_clients):
            num_selection = len(available_clients)
        sampled_clients = random.sample(available_clients, num_selection)
        print(f"Parameter c={self.fraction}. Sampled {num_selection} client(s): {sampled_clients}")
        for i in sampled_clients:
            print(i)
        return sampled_clients