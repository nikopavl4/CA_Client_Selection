import random
import copy
import numpy as np
import torch
import time
from config import federated_args, str2bool
from ml.utils.fed_utils import update_fed_vehicles, move_vehicles
from ml.utils.helpers import zero_IS

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
from dataset.load_dataset import load_MNIST, load_CIFAR10
if args.dataset == 'MNIST':
    trainset1, trainset2, testset = load_MNIST()
    print("== MNIST ==")
    in_dim = 1
    linp = 7
elif args.dataset == 'CIFAR10':
    trainset1, trainset2, testset = load_CIFAR10()
    print("== CIFAR10 ==")
    in_dim = 3
    linp = 8
else:
    print("No correct dataset specified! Exiting...")
    exit()

# Print Dataset Details
num_classes = len(torch.unique(torch.as_tensor(testset.targets)))
print(f'Input Dimensions: {in_dim}')
print(f'Num of Classes: {num_classes}')
print(f'Train Samples: {len(trainset1)}')
print(f'Train Samples to distribute later: {len(trainset2)}')
print(f'Test Samples: {len(testset)}')
print(f'Num of Clients: {args.vehicles}')
print(f'Initial Train samples per client: {int((len(trainset1)/args.vehicles)*(1-args.test_size))}')
print(f'Initial Test samples per client: {int((len(trainset1)/args.vehicles)*(args.test_size))}')
print("===============")

# Create vehicles
from ml.utils.fed_utils import create_fed_vehicles
vehicle_list = create_fed_vehicles(trainset1, args.vehicles)

# Create Clients - randomly assign vehicles from vehicle_list to clients
from ml.utils.fed_utils import create_fed_clients
client_list = create_fed_clients(vehicle_list, args.clients)

# Initialize model, optimizer, criterion
# Get Model
from ml.models.cnn import CNN
model = CNN(in_dim, num_classes, linp)
model.to(device)

# Initialize Fed Clients
from ml.utils.fed_utils import initialize_fed_clients
client_list = initialize_fed_clients(client_list, args, copy.deepcopy(model))

# Perform clients refresh to create their perspective dataset and loaders
from ml.utils.fed_utils import refresh_fed_clients
client_list = refresh_fed_clients(client_list)

# Initiazlize Server with its own strategy, global test, global model, global optimizer, client selection 
from ml.fl.server import Server
Fl_Server = Server(args, testset, copy.deepcopy(model))

total_accs = []
total_f1s = []
start = time.time()
for round in range(args.fl_rounds):
    print(f"FL Round: {round}")
    client_list = Fl_Server.update(client_list)
    acc, f1 = Fl_Server.evaluate()
    print(f'Round {round} - Server Accuracy: {acc}, Server F1: {f1}.')
    total_accs.append(acc)
    total_f1s.append(f1)
 
    client_list = zero_IS(client_list)
    client_list = move_vehicles(vehicle_list, client_list, args.mobility)
    client_list, trainset2 = update_fed_vehicles(client_list, trainset2)
    client_list = refresh_fed_clients(client_list)

stop = time.time()
training_time = stop - start
print(f"Training time: {training_time} sec")

results = total_accs
results.append(training_time)
results.extend(total_f1s)

# Save Results to csv file
import os
import csv
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, f'results/{args.dataset}/{args.seed}_{args.selector}_{args.fraction}.csv')
# Example.csv gets created in the current working directory
with open (filename,'w',newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ',')
    my_writer.writerow(results)