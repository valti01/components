import sys
from argparse import ArgumentParser
from typing import Callable, Optional, Tuple, Any

import eagerpy as ep
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from resnet import ResNet50
import torch
import torchvision
import torchvision.transforms as transforms
from model_normalization import Cifar10Wrapper
from torch.utils.data import Subset
from foolbox.attacks import L2PGD
from foolbox import PyTorchModel, Model
import csv
import os
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.datasets import cifar10 as ds
from datasets import UniformNoiseDataset, SmoothedNoise, ImageNetMinusC10
import random
from random import choice
import pandas as pd


class FaceLandmarksDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, preds, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.preds = preds

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
               Args:
                   index (int): Index

               Returns:
                   tuple: (image, target) where target is index of the target class.
               """
        img, target, pred = self.data[index], self.targets[index], self.preds[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img, target, pred)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CompressImage:

    def __call__(self, pic: torch.Tensor, target: int, pred):

        if self.targetlabel == "random":
            random_gen = random.Random(self.randomseed)
            numbers = [i for i in range(10) if i != target]
            label = random_gen.choice(numbers)
        elif self.targetlabel == "second":
            sorted_indexes = np.argsort(pred)[::-1]
            label = sorted_indexes[1]
        elif self.targetlabel == "own":
            label = target
        elif self.targetlabel == "all":
            label = 0
        else:
            label = int(self.targetlabel)
        pic = np.transpose(pic.numpy(), (1, 2, 0))
        pic = pic.reshape(3072)
        picr = np.dot(np.dot(pic, self.V[label][:self.n_comps, :].T), self.V[label][:self.n_comps, :])
        picr = picr.reshape((1, 32, 32, 3))
        pic_tensor = torch.from_numpy(picr.transpose((0, 3, 1, 2))).float()
        return pic_tensor

    def __init__(self, n_comps, targetlabel, randomseed=10):
        random.seed(5)
        (x_train, y_train), (x_test, y_test) = ds.load_data()
        x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32) / 255
        y_train = y_train.flatten()
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CustomTransform:
    def __init__(self, n_comps, label, randomseed=10):
        self.to_tensor = transforms.ToTensor()
        self.compress_image = CompressImage(n_comps, label, randomseed)

    def __call__(self, img, target, pred):
        img_tensor = self.to_tensor(img)
        compressed_img = self.compress_image(img_tensor, target, pred)
        return compressed_img


def predict(net, loader, device):
    preds = []
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            print(i, end="\r")
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            preds.append(outputs.cpu().numpy())

    print("")
    return np.concatenate(preds, axis=0)


def get_label_vector(loader):
    label_vec = []
    for (images, labels) in loader:
        label_vec.append(labels.numpy())
    return np.concatenate(label_vec, axis=0)


def predict_adv(net, loader, device, attack, eps, is_ood, n_restart):
    preds = []
    adv_images = []
    net.eval()
    fb_model = PyTorchModel(net, bounds=(0, 1), device=device)
    # with torch.no_grad():
    adv_return = None
    for i, (images, labels) in enumerate(loader):
        if is_ood:
            labels = torch.zeros_like(labels) - 1
        else:
            labels = torch.ones_like(labels)
        images, labels = images.to(device), labels.to(device)
        # images = torch.as_tensor(images)
        score_best = None
        # adv_best = None
        pred_best = None
        is_improved = 0
        for j in range(n_restart):
            _, advs_j, success = attack(fb_model, images, labels, epsilons=eps)
            pred_j = net(advs_j)
            score_j = attack.score(pred_j, labels).raw
            score_j = score_j.detach().cpu().numpy()
            pred_j = pred_j.detach().cpu().numpy()
            advs_j = advs_j.detach().cpu().numpy()
            if score_best is None:
                score_best = score_j
                adv_best = advs_j
                pred_best = pred_j
                print(advs_j.shape)
                # dv_return[i]=advs_j
            else:
                is_improved = score_j > score_best
                if np.any(is_improved):
                    score_best[is_improved] = score_j[is_improved]
                    pred_best[is_improved] = pred_j[is_improved]
                    adv_best[is_improved] = advs_j[is_improved]

            # del pred_j, score_j, advs_j
            print(i, j, np.mean(is_improved), np.mean(-score_best * labels.detach().cpu().numpy()), end="\r")

        adv_images.append(adv_best)
        preds.append(pred_best)

    print("advimages", np.concatenate(adv_images, axis=0).shape)
    return np.concatenate(preds, axis=0), np.concatenate(adv_images, axis=0)


def split_data(ds, size=1000, seed=100):
    size = 1000
    n = len(ds)
    rnd_state = np.random.RandomState(seed=seed)
    indices = np.arange(0, n)
    rnd_state.shuffle(indices)
    indices = indices[:size]

    split = Subset(ds, indices=indices)
    return split


def main(params, device):
    print(params)
    model = ResNet50(10)
    print("Load model")
    model.load_state_dict(torch.load(params.fname))
    model = Cifar10Wrapper(model)
    model.to(device)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    fulltestloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size, shuffle=False, num_workers=1)
    full_pred_in = predict(model, fulltestloader, device)

    if params.size != 10000:
        testset = split_data(testset, size=params.size, seed=params.seed)

    testloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
                                             shuffle=False, num_workers=1)

    if params.pred_fname is not None:
        newmodel = ResNet50(10)
        newmodel.load_state_dict(torch.load(params.pred_fname))
        newmodel = Cifar10Wrapper(newmodel)
        newmodel.to(device)
        full_pred_in = predict(newmodel, fulltestloader, device)

    l = [x for (x, y) in testloader]
    x_test = torch.cat(l, 0)
    x_test = x_test.detach().numpy()

    componentnumber = params.init_comps
    transform = CustomTransform(componentnumber, params.targetlabel, randomseed=params.randomseed)
    compressed = FaceLandmarksDataset(root='./data', train=False, preds=full_pred_in, download=True,
                                      transform=transform)
    compressed = split_data(compressed, size=1000)

    while componentnumber > 1:
        l = [x for (x, y) in compressed]
        x_test_ood = torch.cat(l, 0)
        x_test_ood = x_test_ood.detach().numpy()

        # distance = np.max(np.abs(x_test - x_test_ood), axis=(1,2,3))
        distance = np.sqrt(np.sum((x_test - x_test_ood) ** 2, axis=(1, 2, 3)))
        if np.min(distance) >= 1:
            break
        print(f"current number:{componentnumber} current minimum distance: {np.min(distance)}")
        componentnumber -= 1
        transform.compress_image.n_comps = componentnumber

    print(f"components needed using {params.targetlabel} to get at least that distance: ", componentnumber)
    print("Percentage of distances greater than 1: ", np.mean(distance > 1))
    print("Min distance: ", np.min(distance))
    print("Max distance: ", np.max(distance))
    print("Average distance: ", np.mean(distance))
    print("Median distance: ", np.median(distance))
    stats = []
    stats.append({'components': componentnumber,
                  'label': params.targetlabel,
                  'greater_than_1': np.mean(distance > 1),
                  'min_distance': np.min(distance),
                  'max_distance': np.max(distance),
                  'average_distance': np.mean(distance),
                  'median_distance': np.median(distance)
                  })
    with open(params.out_fname, mode='a+', newline='') as out_f:
        args = vars(params)
        fieldnames = sorted(set(stats[0].keys()).union(args.keys()))
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if not os.path.exists(params.out_fname):
            writer.writeheader()
        for i in range(len(stats)):
            writer.writerow({**args, **stats[i]})
    sys.exit(0)


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--pred_fname", type=str)
    parser.add_argument("--out_fname", type=str, required=True)
    parser.add_argument("--init_comps", type=int, default=300)
    parser.add_argument("--targetlabel", type=str, default="all", help="Options: all, random, second, own, 0-9")
    parser.add_argument("--randomseed", type=int, default=10, help="For targetlabel: random")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--size", type=int, default=1000)

    FLAGS = parser.parse_args()
    np.random.seed(1009)
    device = torch.device(("cuda:" + str(FLAGS.gpu)) if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    main(FLAGS, device)
