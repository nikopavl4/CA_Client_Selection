import torchvision
import torchvision.datasets as datasets
from torch.utils.data import random_split

def load_MNIST():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    half_data  = int(len(mnist_trainset)/2)
    split_list = [half_data, len(mnist_trainset)-half_data]
    mnist_trainset1, mnist_trainset2 = random_split(mnist_trainset, split_list)


    print(f"MNIST - Loaded {len(mnist_trainset1)} images as trainset, {len(mnist_trainset1)} images as secondary trainset and {len(mnist_testset)} images as testset.")

    return mnist_trainset1, mnist_trainset2, mnist_testset

def load_CIFAR10():
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    half_data  = int(len(cifar_trainset)/2)
    split_list = [half_data, len(cifar_trainset)-half_data]
    cifar_trainset1, cifar_trainset2 = random_split(cifar_trainset, split_list)


    print(f"CIFAR10 - Loaded {len(cifar_trainset1)} images as trainset, {len(cifar_trainset1)} images as secondary trainset and {len(cifar_testset)} images as testset.")

    return cifar_trainset1, cifar_trainset2, cifar_testset