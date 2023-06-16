import random
from typing import Tuple
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

# Create Clients - each client has its own id, trainloader, testloader, model, optimizer
client_list = create_clients()

# Initiazlize Server with its own strategy, global test, global model, global optimizer, client selection 
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

# Print Dataset Details
print("== MNIST ==")
in_dim = 1
num_classes = len(torch.unique(torch.as_tensor(train_loader.dataset.targets)))
print(f'Input Dimensions: {in_dim}')
print(f'Num of Classes: {num_classes}')
print(f'Train Samples: {len(trainset)}')
print(f'Test Samples: {len(testset)}')
print(f'Num of Clients: {args.clients}')
print(f'Train samples per client: {(len(trainset)/args.clients)*(1-args.test_size)}')
print(f'Test samples per client: {(len(trainset)/args.clients)*(args.test_size)}')

# for i in range fed rounds perform a server update

# print results

# Get Model
from ml.models.cnn import CNN
model = CNN()
model.to(device)

# Get Criterion
from ml.utils.helpers import get_criterion
criterion = get_criterion(args.criterion)

# Get Optimizer
from ml.utils.helpers import get_optim
optimizer = get_optim(model, args.optimizer, args.lr)

#Train, Test
train_history = train(model,train_loader, args.device, criterion, optimizer, args.epochs, True)
acc, f1 = test(model,test_loader,criterion, args.device)

print(f'Final Results on Test - Accuracy: {acc}, F1: {f1}.')