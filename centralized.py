import random
from typing import Tuple
import numpy as np
import torch
import time
import torch.nn.functional as F
from config import centralized_args, str2bool
from ml.utils.train_utils import train, test

# Load arguments
args = centralized_args()
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
from dataset.load_dataset import load_MNIST_cen, load_CIFAR10_cen
if args.dataset == 'MNIST':
    trainset, testset = load_MNIST_cen()
    print("== MNIST ==")
    in_dim = 1
    linp = 7
elif args.dataset == 'CIFAR10':
    trainset, testset = load_CIFAR10_cen()
    print("== CIFAR10 ==")
    in_dim = 3
    linp = 8
else:
    print("No correct dataset specified! Exiting...")
    exit()

# Create Dataloaders
from torch.utils.data import DataLoader
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

# Print Dataset Details
num_classes = len(torch.unique(torch.as_tensor(train_loader.dataset.targets)))
print(f'Input Dimensions: {in_dim}')
print(f'Num of Classes: {num_classes}')
print(f'Train Samples: {len(trainset)}')
print(f'Test Samples: {len(testset)}')


# Get Model
from ml.models.cnn import CNN
model = CNN(in_dim, num_classes, linp)
model.to(device)

# Get Criterion
from ml.utils.helpers import get_criterion
criterion = get_criterion(args.criterion)

# Get Optimizer
from ml.utils.helpers import get_optim
optimizer = get_optim(model, args.optimizer, args.lr)

#Train, Test
start = time.time()
train_history = train(model,train_loader, args.device, criterion, optimizer, args.epochs,True)
stop = time.time()
training_time = stop - start
print(f"Training time: {training_time} sec")
acc, f1 = test(model,test_loader,criterion, args.device)

print(f'Final Results on Test - Accuracy: {acc}, F1: {f1}.')