from argparse import ArgumentParser
import numpy as np
from resnet import ResNet50
import torch
import torchvision
import torchvision.transforms as transforms
from model_normalization import Cifar10Wrapper
from foolbox.attacks import L2PGD
from foolbox import PyTorchModel
import csv
import os

def eval_acc(net, loader, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        i = 0
        for data in loader:
            print(i, end="\r")
            i = i + 1
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nAccuracy of the network on the 10000 test images: %.2f %%' % (
            100 * correct / total))
    return correct / total


def eval_robust_acc(net, loader, device, attack, eps):
    correct = 0
    total = 0
    net.eval()
    fb_model = PyTorchModel(net, bounds=(0, 1), device=device)
    # with torch.no_grad():
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        images = torch.as_tensor(images)
        _, advs, success = attack(fb_model, images, labels, epsilons=eps)
        outputs = net(advs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(i, correct / total, end="\r")

    print('\nAccuracy of the network on the 10000 test images: %.2f %%' % (
            100 * correct / total))
    return correct / total


def main(params, device):
    print(params)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
                                             shuffle=False, num_workers=2)
    model = ResNet50(10)
    print("Load model")
    model.load_state_dict(torch.load(params.fname))
    model = Cifar10Wrapper(model)
    model.to(device)
    acc=eval_acc(model, testloader, device)
    attack = L2PGD(steps=params.steps, abs_stepsize=params.abs_stepsize, random_start=True)
    if params.n_restarts > 1:
        attack = attack.repeat(params.n_restarts)
    a_acc=eval_robust_acc(model, testloader, device, attack, params.eps)
    if params.out_fname is not None:
        is_exist=os.path.exists(params.out_fname)
        with open(params.out_fname,mode='a+') as out_f:
            args=vars(params)
            writer=csv.DictWriter(out_f,fieldnames=sorted({'acc','a_acc'}.union(args.keys())))
            if not is_exist:
                writer.writeheader()
            writer.writerow({**args,'acc':acc,'a_acc':a_acc})


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--out_fname", type=str)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--n_restarts", type=int, default=10)
    parser.add_argument("--abs_stepsize", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=50)
    FLAGS = parser.parse_args()
    np.random.seed(9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    main(FLAGS, device)
