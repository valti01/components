import cv2
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from typing import Callable, Optional, Tuple, Any
import numpy as np
from PIL import Image

import torch
import torchvision

from random import choice
import random


class CompressedDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
               Args:
                   index (int): Index

               Returns:
                   tuple: (image, target) where target is index of the target class.
               """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CompressImage:

    def __call__(self, pic: torch.Tensor, target: int):
        if self.targetlabel == "random":
            random_gen = random.Random(self.randomseed)
            numbers = [i for i in range(10) if i != target]
            label = random_gen.choice(numbers)
        elif self.targetlabel == "own":
            label = target
        elif self.targetlabel == "all":
            label = 0
        else:
            label = int(self.targetlabel)
        pic = np.transpose(pic.numpy(), (1, 2, 0))
        # pic = pic.numpy()
        pic = pic.reshape(3072)

        pre = np.dot(pic, self.V[label].T)

        left = 1
        right = 3072
        best_componentnumber = 3072
        # if cant be done with 1 comp, return 1 comp

        if self.mindistance is not None:

            while left <= right:
                mid = (left + right) // 2
                picr = np.dot(np.dot(pic, self.V[label][:mid, :].T), self.V[label][:mid, :])
                if self.distancenorm == "L2":
                    distance = np.sqrt(np.sum((pic - picr) ** 2))
                elif self.distancenorm == "Linf":
                    distance = np.max(np.abs(pic - picr))
                if distance <= self.mindistance:
                    best_componentnumber = mid
                    best_distance = distance
                    right = mid - 1
                else:
                    left = mid + 1
            best_componentnumber -= 1
            best_componentnumber = max(best_componentnumber, 1)
            picr = np.dot(np.dot(pic, self.V[label][:best_componentnumber, :].T),
                          self.V[label][:best_componentnumber, :])
        else:
            picr = np.dot(pre[:self.n_comps], self.V[label][:self.n_comps, :])
        # distance = np.sqrt(np.sum((pic - picr) ** 2))

        picr = picr.reshape((32, 32, 3))
        pic_tensor = torch.from_numpy(picr.transpose((2, 0, 1))).float()
        return pic_tensor

    def __init__(self, n_comps, targetlabel, randomseed, mindistance, distancenorm):
        random.seed(5)
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                transform=transforms.ToTensor())
        x_train = train_set.data
        y_train = train_set.targets
        x_test = test_set.data
        y_test = test_set.targets
        x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32) / 255
        y_train = np.array(y_train)
        V = []
        if targetlabel == "all":
            M = np.dot(x_train.T, x_train)
            U, S, Vtemp = np.linalg.svd(M)
            V.append(Vtemp)
        else:
            for i in range(np.max(y_train) + 1):
                current = x_train[y_train == i]
                M = np.dot(current.T, current)
                U, S, Vtemp = np.linalg.svd(M)
                V.append(Vtemp)
        self.targetlabel = targetlabel
        self.V = V
        self.n_comps = n_comps
        self.randomseed = randomseed
        self.mindistance = mindistance
        self.distancenorm = distancenorm
        # current_time = time.time()
        # runtime = current_time - start_time
        # hours, remainder = divmod(runtime,3600)
        # minute, seconds = divmod(remainder, 60)
        # print(f"runtime: {hours} hours {minute} minutes {seconds} seconds")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TensorAndCompression:
    def __init__(self, n_comps, label, randomseed, mindistance, distancenorm="Linf"):
        self.to_tensor = transforms.ToTensor()
        self.compress_image = CompressImage(n_comps, label, randomseed, mindistance, distancenorm)

    def __call__(self, img, target):
        img_tensor = self.to_tensor(img)
        compressed_img = self.compress_image(img_tensor, target)
        return compressed_img


def split_data(dataset, size=1000, seed=100):
    size = 1000
    n = len(dataset)
    rnd_state = np.random.RandomState(seed=seed)
    indices = np.arange(0, n)
    rnd_state.shuffle(indices)
    indices = indices[:size]
    subset = Subset(dataset, indices=indices)
    return subset


def food_loader(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (32, 32))
    return img


def get_out_training_loaders(batch_size):
    trainset_out = torchvision.datasets.ImageFolder(root='data/food-101/images/', loader=food_loader,
                                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                                  transforms.RandomChoice(
                                                                                      [transforms.RandomApply([
                                                                                                                  transforms.RandomAffine(
                                                                                                                      90,
                                                                                                                      translate=(
                                                                                                                      0.15,
                                                                                                                      0.15),
                                                                                                                      scale=(
                                                                                                                      0.85,
                                                                                                                      1),
                                                                                                                      shear=None)],
                                                                                                              p=0.6),
                                                                                       transforms.RandomApply([
                                                                                                                  transforms.RandomAffine(
                                                                                                                      0,
                                                                                                                      translate=None,
                                                                                                                      scale=(
                                                                                                                      0.5,
                                                                                                                      0.75),
                                                                                                                      shear=30)],
                                                                                                              p=0.6),
                                                                                       transforms.RandomApply(
                                                                                           [transforms.AutoAugment()],
                                                                                           p=0.9), ]),
                                                                                  transforms.ToTensor(), ]))
    trainloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)

    valset_out = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    valloader_out = DataLoader(valset_out, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader_out, valloader_out


def bird_loader(path):
    path = path.split('/')
    if path[-1][0:2] == '._':
        path[-1] = path[-1][2:]
    path = '/'.join(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (32, 32))
    return img


def flower_loader(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (32, 32))
    return img


def get_out_testing_datasets(out_names, comps=100, targetlabel=1, randomseed=1, mindistance=(1.25 * 2 * (8 / 255))):
    out_datasets = []
    returned_out_names = []

    for name in out_names:

        if name == 'mnist':
            mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                               transform=transforms.Compose([transforms.ToTensor(),
                                                                             transforms.Resize(32),
                                                                             transforms.Lambda(
                                                                                 lambda x: x.repeat(3, 1, 1)),
                                                                             ]))
            returned_out_names.append(name)
            out_datasets.append(mnist)

        elif name == 'tiny_imagenet':
            tiny_imagenet = torchvision.datasets.ImageFolder(root='data/tiny-imagenet-200/test',
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Resize(32)]))

            returned_out_names.append(name)
            out_datasets.append(tiny_imagenet)

        elif name == 'places':
            places365 = torchvision.datasets.Places365(root='data/', split='val', small=True, download=False,
                                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                                     transforms.Resize(32)]))

            returned_out_names.append(name)
            out_datasets.append(places365)

        elif name == 'LSUN':
            LSUN = torchvision.datasets.ImageFolder(root='data/LSUN_resize/', transform=transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(LSUN)

        elif name == 'iSUN':
            iSUN = torchvision.datasets.ImageFolder(root='data/iSUN/', transform=transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(iSUN)

        elif name == 'birds':
            birds = torchvision.datasets.ImageFolder(root='data/images/', loader=bird_loader,
                                                     transform=transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(birds)

        elif name == 'flowers':
            flowers = torchvision.datasets.ImageFolder(root='data/flowers/', loader=flower_loader,
                                                       transform=transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(flowers)

        elif name == 'coil':
            coil_100 = torchvision.datasets.ImageFolder(root='data/coil/',
                                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                                      transforms.Resize(32)]))

            returned_out_names.append(name)
            out_datasets.append(coil_100)
        elif name == 'cifar10-compressed':
            cifar10compressed = CompressedDataset(root='./data', train=False, download=True,
                                                  transform=TensorAndCompression(comps, targetlabel,
                                                                                 randomseed=randomseed,
                                                                                 mindistance=mindistance))
            returned_out_names.append(name)
            cifar10compressed = split_data(cifar10compressed, size=1000)
            out_datasets.append(cifar10compressed)

        else:
            print(name, ' dataset is not implemented.')

    return returned_out_names, out_datasets
