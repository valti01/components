import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms



def get_in_training_loaders(in_dataset, batch_size):

    if in_dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'TI':
        dataset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader

def split_data(dataset, size=1000, seed=100):
    size=1000
    n = len(dataset)
    rnd_state = np.random.RandomState(seed=seed)
    indices = np.arange(0, n)
    rnd_state.shuffle(indices)
    indices = indices[:size]
    subset = Subset(dataset, indices=indices)
    print(len(subset))
    return subset


def get_in_testing_loader(in_dataset, batch_size):

    if in_dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())
        testset = split_data(testset, size=1000)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif in_dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'TI':
        testset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/val', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader