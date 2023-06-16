import torchvision
import torchvision.datasets as datasets

def load_MNIST():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    print(f"MNIST - Loaded {len(mnist_trainset)} images as trainset and {len(mnist_testset)} images as testset.")

    return mnist_trainset, mnist_testset