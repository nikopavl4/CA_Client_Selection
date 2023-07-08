import random
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from config import federated_args, str2bool
from ml.utils.train_utils import train, test

# Load arguments
args = federated_args()
print(f"Script arguments: {args}\n")

# Enable Cuda if available
if torch.cuda.is_available():
    device = args.device
else:
    device = 'cpu'

# ensure reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load dataset
from dataset.load_dataset import load_MNIST
trainset, testset = load_MNIST()

# Print Dataset Details
print("== MNIST ==")
in_dim = 1
num_classes = len(torch.unique(torch.as_tensor(trainset.targets)))
print(f'Input Dimensions: {in_dim}')
print(f'Num of Classes: {num_classes}')
print(f'Train Samples: {len(trainset)}')
print(f'Test Samples: {len(testset)}')
print(f'Num of Clients: {args.clients}')
print(f'Train samples per client: {int((len(trainset)/args.clients)*(1-args.test_size))}')
print(f'Test samples per client: {int((len(trainset)/args.clients)*(args.test_size))}')
print("===============")

# Create Clients - each client has its own id, trainloader, testloader, model, optimizer
from ml.utils.fed_utils import create_fed_clients
client_list = create_fed_clients(trainset, args.clients)

# Initialize model, optimizer, criterion
# Get Model
from ml.models.cnn import CNN
model = CNN()
model.to(device)

# Initialize Fed Clients
from ml.utils.fed_utils import initialize_fed_clients
client_list = initialize_fed_clients(client_list, args, copy.deepcopy(model))

# Initiazlize Server with its own strategy, global test, global model, global optimizer, client selection 
from ml.fl.server import Server
Fl_Server = Server(args, testset, copy.deepcopy(model))

for round in range(args.fl_rounds+1):
    print(f"FL Round: {round}")
    client_list = Fl_Server.update(client_list)
    acc, f1 = Fl_Server.evaluate()
    print(f'Round {round} - Server Accuracy: {acc}, Server F1: {f1}.')