import sys
from argparse import ArgumentParser
from typing import Callable, Optional, Tuple, Any

import eagerpy as ep
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import softmax

from resnet import ResNet50
import torch
import torchvision
import torchvision.transforms as transforms
from model_normalization import Cifar10Wrapper
from torch.utils.data import Subset
from foolbox.attacks import L2PGD, LinfPGD
from foolbox import PyTorchModel, Model
import csv
import os
from sklearn.metrics import roc_curve, auc
#from datasets import UniformNoiseDataset, SmoothedNoise, ImageNetMinusC10
import random
from random import choice
import pandas as pd
import time

class CompressedDataset(torchvision.datasets.CIFAR10):
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
        #pic = pic.numpy()
        pic = pic.reshape(3072)

        pre = np.dot(pic, self.V[label].T)

        left = 1
        right = 3072
        best_componentnumber = 3072
        #if cant be done with 1 comp, return 1 comp

        if self.mindistance is not None:

            while left <= right:
                mid = (left + right) // 2
                imgr = np.dot(np.dot(pic, self.V[label][:mid, :].T), self.V[label][:mid, :])
                if self.distancenorm == "L2":
                    distance = np.sqrt(np.sum((pic - imgr) ** 2))
                elif self.distancenorm == "Linf":
                    distance = np.max(np.abs(pic - imgr))
                else:
                    print("unsupported distance norm")
                if distance <= self.mindistance:
                    best_componentnumber = mid
                    best_distance = distance
                    right = mid - 1
                else:
                    left = mid + 1
            best_componentnumber -= 1
            best_componentnumber = max(best_componentnumber,1)
            picr = np.dot(np.dot(pic, self.V[label][:best_componentnumber, :].T),
                          self.V[label][:best_componentnumber, :])
        else:
            picr = np.dot(pre[:self.n_comps], self.V[label][:self.n_comps, :])
        #distance = np.sqrt(np.sum((pic - picr) ** 2))

        picr = picr.reshape((32, 32, 3))
        pic_tensor = torch.from_numpy(picr.transpose((2, 0, 1))).float()
        return pic_tensor

    def __init__(self, n_comps, targetlabel, randomseed, mindistance, distancenorm):
        random.seed(5)
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
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
        current_time = time.time()
        runtime = current_time - start_time
        hours, remainder = divmod(runtime,3600)
        minute, seconds = divmod(remainder, 60)
        print(f"runtime: {hours} hours {minute} minutes {seconds} seconds")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TensorAndCompression:
    def __init__(self, n_comps, label, randomseed, mindistance, distancenorm):
        self.to_tensor = transforms.ToTensor()
        self.compress_image = CompressImage(n_comps, label, randomseed, mindistance, distancenorm)

    def __call__(self, img, target, pred):
        img_tensor = self.to_tensor(img)
        compressed_img = self.compress_image(img_tensor, target, pred)
        return compressed_img

def print_distances(distance):
    print("distance shape: ", distance.shape)
    print("Percentage of distances greater than 1: ", np.mean(distance > 1))
    print("Min distance: ", np.min(distance))
    print("Max distance: ", np.max(distance))
    print("Average distance: ", np.mean(distance))
    print("Median distance: ", np.median(distance))


def save_images(distance, x_test,x_test_ood,params,pred_in,pred_out,attack_distance,x_test_ood_adv,pred_a_out,x_test_adv,pred_a_in,top_10_indices):
    #return
    max_index = np.argmax(distance)
    max_x_test = x_test[max_index]
    max_x_test_ood = x_test_ood[max_index]

    if not os.path.exists(f'{params.imgdir}/max(x,xc)/'):
        os.makedirs(f'{params.imgdir}/max(x,xc)')
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(max_x_test.transpose(1, 2, 0).reshape(32, 32, 3))
    axes[0].set_axis_off()
    axes[0].set_title(f"x {parse_label(np.argmax(pred_in[max_index]))}, {np.max(softmax(pred_in[max_index])): .2f}")
    axes[1].imshow(max_x_test_ood.transpose(1, 2, 0).reshape(32, 32, 3))
    axes[1].set_axis_off()
    axes[1].set_title(
        f"xc {parse_label(np.argmax(pred_out[max_index]))}, {np.max(softmax(pred_out[max_index])): .2f}")
    plt.savefig(f"{params.imgdir}/max(x,xc)/figure_max_{params.comps}_comps.png")
    plt.close()

    min_index = np.argmin(distance)
    min_x_test = x_test[min_index]
    min_x_test_ood = x_test_ood[min_index]

    if not os.path.exists(f'{params.imgdir}/min(x,xc)/'):
        os.makedirs(f'{params.imgdir}/min(x,xc)')
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(min_x_test.transpose(1, 2, 0).reshape(32, 32, 3))
    axes[0].set_axis_off()
    axes[0].set_title(f"x {parse_label(np.argmax(pred_in[min_index]))}, {np.max(softmax(pred_in[min_index])): .2f}")
    axes[1].imshow(min_x_test_ood.transpose(1, 2, 0).reshape(32, 32, 3))
    axes[1].set_axis_off()
    axes[1].set_title(
        f"xc {parse_label(np.argmax(pred_out[min_index]))}, {np.max(softmax(pred_out[min_index])): .2f}")
    plt.savefig(f"{params.imgdir}/min(x,xc)/figure_min_{params.comps}_comps.png")
    plt.close()

    max_index = np.argmax(attack_distance)
    max_x_test = x_test[max_index]
    max_x_test_ood_adv = x_test_ood_adv[max_index]

    if not os.path.exists(f'{params.imgdir}/max(x,xca)/'):
        os.makedirs(f'{params.imgdir}/max(x,xca)')
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(max_x_test.transpose(1, 2, 0).reshape(32, 32, 3))
    axes[0].set_axis_off()
    axes[0].set_title(f"x {parse_label(np.argmax(pred_in[max_index]))}, {np.max(softmax(pred_in[max_index])): .2f}")
    axes[1].imshow(max_x_test_ood_adv.transpose(1, 2, 0).reshape(32, 32, 3))
    axes[1].set_axis_off()
    axes[1].set_title(
        f"xca {parse_label(np.argmax(pred_a_out[max_index]))},{np.max(softmax(pred_a_out[max_index])): .2f}")
    plt.savefig(f"{params.imgdir}/max(x,xca)/figure_max_{params.comps}_comps.png")
    plt.close()

    # min distance image

    min_index = np.argmin(attack_distance)
    min_x_test = x_test[min_index]
    min_x_test_ood_adv = x_test_ood_adv[min_index]

    if not os.path.exists(f'{params.imgdir}/min(x,xca)/'):
        os.makedirs(f'{params.imgdir}/min(x,xca)')
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(min_x_test.transpose(1, 2, 0).reshape(32, 32, 3))
    axes[0].set_axis_off()
    axes[0].set_title(f"x {parse_label(np.argmax(pred_in[min_index]))}, {np.max(softmax(pred_in[min_index])): .2f}")
    axes[1].imshow(min_x_test_ood_adv.transpose(1, 2, 0).reshape(32, 32, 3))
    axes[1].set_axis_off()
    axes[1].set_title(
        f"xca {parse_label(np.argmax(pred_a_out[min_index]))}, {np.max(softmax(pred_a_out[min_index])): .2f}")
    plt.savefig(f"{params.imgdir}/min(x,xca)/figure_min_{params.comps}_comps.png")
    plt.close()

    if not os.path.exists(f'{params.imgdir}/{params.comps}'):
        os.makedirs(f'{params.imgdir}/{params.comps}')

    for i in range(100):
        diff = x_test_adv[i] - x_test[i]
        diff = diffscale(diff)

        diffood = x_test_ood_adv[i] - x_test_ood[i]
        diffood = diffscale(diffood)
        fig, axes = plt.subplots(2, 3)
        axes[0, 0].imshow(x_test[i].transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0, 0].set_axis_off()
        axes[0, 0].set_title(f"x {parse_label(np.argmax(pred_in[i]))}, {np.max(softmax(pred_in[i])): .2f}")
        axes[0, 2].imshow(x_test_adv[i].transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0, 2].set_axis_off()
        axes[0, 2].set_title(f"xa {parse_label(np.argmax(pred_a_in[i]))}, {np.max(softmax(pred_a_in[i])): .2f}")
        axes[1, 0].imshow(x_test_ood[i].transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1, 0].set_axis_off()
        axes[1, 0].set_title(f"xc {parse_label(np.argmax(pred_out[i]))}, {np.max(softmax(pred_out[i])): .2f}")
        axes[1, 2].imshow(x_test_ood_adv[i].transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1, 2].set_axis_off()
        axes[1, 2].set_title(
            f"xca {parse_label(np.argmax(pred_a_out[i]))}, {np.max(softmax(pred_a_out[i])): .2f}")
        axes[0, 1].imshow(diff.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0, 1].set_axis_off()
        axes[0, 1].set_title("diff(x,xa)")
        axes[1, 1].imshow(diffood.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1, 1].set_axis_off()
        axes[1, 1].set_title("diff(xc,xca)")

        plt.savefig(f"{params.imgdir}/{params.comps}/figure_{i}.png", bbox_inches='tight')
        plt.close()

    # top 10 x_test_ood_adv
    if not os.path.exists(f"{params.imgdir}/{params.comps}/top10confidence"):
        os.makedirs(f"{params.imgdir}/{params.comps}/top10confidence")

    for i in range(10):
        index = top_10_indices[i]
        diff = x_test_adv[index] - x_test[index]
        diff = diffscale(diff)

        diffood = x_test_ood_adv[index] - x_test_ood[index]
        diffood = diffscale(diffood)
        fig, axes = plt.subplots(2, 3)
        axes[0, 0].imshow(x_test[index].transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0, 0].set_axis_off()
        axes[0, 0].set_title(f"x {parse_label(np.argmax(pred_in[index]))}, {np.max(softmax(pred_in[index])): .2f}")
        axes[0, 2].imshow(x_test_adv[index].transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0, 2].set_axis_off()
        axes[0, 2].set_title(
            f"xa {parse_label(np.argmax(pred_a_in[index]))}, {np.max(softmax(pred_a_in[index])): .2f}")
        axes[1, 0].imshow(x_test_ood[index].transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1, 0].set_axis_off()
        axes[1, 0].set_title(
            f"xc {parse_label(np.argmax(pred_out[index]))}, {np.max(softmax(pred_out[index])): .2f}")
        axes[1, 2].imshow(x_test_ood_adv[index].transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1, 2].set_axis_off()
        axes[1, 2].set_title(
            f"xca {parse_label(np.argmax(pred_a_out[index]))}, {np.max(softmax(pred_a_out[index])): .2f}")
        axes[0, 1].imshow(diff.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0, 1].set_axis_off()
        axes[0, 1].set_title("diff(x,xa)")
        axes[1, 1].imshow(diffood.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1, 1].set_axis_off()
        axes[1, 1].set_title("diff(xc,xca)")

        plt.savefig(f"{params.imgdir}/{params.comps}/top10confidence/figure_{i}.png", bbox_inches='tight')
        plt.close()



def predict(net, loader, device):
    preds = []
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            print(i, end="\r")
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            preds.append(outputs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def get_label_vector(loader):
    label_vec = []
    for (images, labels) in loader:
        label_vec.append(labels.numpy())
    return np.concatenate(label_vec, axis=0)

def parse_label(number):
    if number == 0:
        return "airplane"
    elif number == 1:
        return "automobile"
    elif number == 2:
        return "bird"
    elif number == 3:
        return "cat"
    elif number == 4:
        return "deer"
    elif number == 5:
        return "dog"
    elif number == 6:
        return "frog"
    elif number == 7:
        return "horse"
    elif number == 8:
        return "ship"
    elif number == 9:
        return "truck"


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

    return np.concatenate(preds, axis=0), np.concatenate(adv_images, axis=0)


def split_data(dataset, size=1000, seed=100):
    size=1000
    n = len(dataset)
    rnd_state = np.random.RandomState(seed=seed)
    indices = np.arange(0, n)
    rnd_state.shuffle(indices)
    indices = indices[:size]
    subset = Subset(dataset, indices=indices)
    return subset


def get_ood_datasets(targetlabel, comps, randomseed, batch_size, srcds, ood_names, pred_in, mindistance,distancenorm  ,size=1000):
    ood_ds = []
    if ood_names is None or np.isin('svhn', ood_names):
        testsetsvhn = torchvision.datasets.SVHN(root='./data', split='test', download=True,
                                                transform=transforms.ToTensor())
        testsetsvhn = split_data(testsetsvhn, size=size)
        testloadersvhn = torch.utils.data.DataLoader(testsetsvhn, batch_size=batch_size,
                                                     shuffle=False, num_workers=2)
        ood_ds.append(('svhn', testloadersvhn))
        print('Add: ', ood_ds[-1][0])
    if ood_names is None or np.isin('cifar100', ood_names):
        testsetc100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                    transform=transforms.ToTensor())
        testsetc100 = split_data(testsetc100, size=size)
        testloaderc100 = torch.utils.data.DataLoader(testsetc100, batch_size=batch_size,
                                                     shuffle=False, num_workers=2)
        ood_ds.append(('cifar100', testloaderc100))
        print('Add: ', ood_ds[-1][0])
    if ood_names is None or np.isin('cifar10-compressed', ood_names):
        testsetc10 = CompressedDataset(root='./data', train=False, preds=pred_in, download=True,
                                       transform=TensorAndCompression(comps, targetlabel, randomseed, mindistance, distancenorm))
        testsetc10 = split_data(testsetc10, size=size)

        testloaderc10 = torch.utils.data.DataLoader(testsetc10, batch_size=batch_size, shuffle=False, num_workers=1)

        ood_ds.append(('cifar10-compressed', testloaderc10))
        print('Add: ', ood_ds[-1][0])
    if ood_names is None or np.isin('lsun', ood_names):
        testsetlsun = torchvision.datasets.LSUN(root='./data', classes=['classroom_train'],
                                                transform=transforms.Compose([
                                                    transforms.CenterCrop(224),
                                                    transforms.Resize(32),
                                                    transforms.ToTensor()
                                                ]))
        testsetlsun = split_data(testsetlsun, size=size)
        testloaderlsun = torch.utils.data.DataLoader(testsetlsun, batch_size=batch_size,
                                                     shuffle=False, num_workers=2)
        ood_ds.append(('lsun', testloaderlsun))
        print('Add: ', ood_ds[-1][0])
    # if ood_names is None or np.isin('uniform', ood_names):
    #     un = UniformNoiseDataset(size)
    #     testloaderun = torch.utils.data.DataLoader(un, batch_size=batch_size,
    #                                                shuffle=False, num_workers=2)
    #     ood_ds.append(('uniform', testloaderun))
    #     print('Add: ', ood_ds[-1][0])
    #
    # if ood_names is None or np.isin('smoothed_noise', ood_names):
    #     sn = SmoothedNoise(srcds)
    #     testloadersn = torch.utils.data.DataLoader(sn, batch_size=batch_size,
    #                                                shuffle=False, num_workers=2)
    #     ood_ds.append(('smoothed_noise', testloadersn))
    #     print('Add: ', ood_ds[-1][0])
    # if ood_names is None or np.isin('imagenet', ood_names):
    #     imagenet = ImageNetMinusC10()
    #     testloaderin = torch.utils.data.DataLoader(imagenet, batch_size=batch_size,
    #                                                shuffle=False, num_workers=2)
    #     ood_ds.append(('imagenet', testloaderin))
    #     print('Add: ', ood_ds[-1][0])

    return ood_ds


def msp(l):
    probs = np.exp(l)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return np.max(probs, axis=1)


def ml(l):
    return np.max(l, axis=1)


def lse(l):
    return np.log(np.sum(np.exp(l), axis=1))


def ul(l):
    return lse(l) - np.mean(l, axis=1)


class L2SCOREPGD(L2PGD):

    def __init__(self, *, rel_stepsize: float = 0.025, abs_stepsize: Optional[float] = None, steps: int = 50,
                 random_start: bool = True, loss='msp'):
        self.loss = loss
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

    def score(self, logits, labels):
        logits_tensor = ep.astensor(logits)
        if self.loss == 'msp':
            return -ep.max(ep.softmax(logits_tensor, axis=1), axis=1) * labels
        elif self.loss == 'ml':
            return -ep.max(logits_tensor, axis=1) * labels
        elif self.loss == 'ul':
            lse_score = ep.log(ep.sum(ep.exp(logits_tensor), axis=1))
            mean_logit = ep.mean(logits_tensor, axis=1)
            return (mean_logit - lse_score) * labels
        elif self.loss == 'lse':
            lse_score = ep.log(ep.sum(ep.exp(logits_tensor), axis=1))
            return -(lse_score * labels)
        else:
            raise Exception('Loss:{0} not supported'.format(self.loss))

    def get_loss_fn(self, model: Model, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        _self = self

        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.sum(_self.score(logits, labels))

        return loss_fn

class LinfSCOREPGD(LinfPGD):

    def __init__(self, *, rel_stepsize: float = 0.025, abs_stepsize: Optional[float] = None, steps: int = 50,
                 random_start: bool = True, loss='msp'):
        self.loss = loss
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

    def score(self, logits, labels):
        logits_tensor = ep.astensor(logits)
        if self.loss == 'msp':
            return -ep.max(ep.softmax(logits_tensor, axis=1), axis=1) * labels
        elif self.loss == 'ml':
            return -ep.max(logits_tensor, axis=1) * labels
        elif self.loss == 'ul':
            lse_score = ep.log(ep.sum(ep.exp(logits_tensor), axis=1))
            mean_logit = ep.mean(logits_tensor, axis=1)
            return (mean_logit - lse_score) * labels
        elif self.loss == 'lse':
            lse_score = ep.log(ep.sum(ep.exp(logits_tensor), axis=1))
            return -(lse_score * labels)
        else:
            raise Exception('Loss:{0} not supported'.format(self.loss))

    def get_loss_fn(self, model: Model, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        _self = self

        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.sum(_self.score(logits, labels))

        return loss_fn

def old_algorithm(global_min_distance, componentnumber, targetlabel, randomseed, distance_norm, full_pred_in, original_images):
    transform = TensorAndCompression(componentnumber, targetlabel, randomseed=randomseed,
                                     mindistance=None, distancenorm=distance_norm)
    compressed = CompressedDataset(root='./data', train=False, preds=full_pred_in, download=True,
                                   transform=transform)
    compressed = split_data(compressed, size=1000)
    
    
    componentnumber = 1
    while componentnumber < 3072:
        l = [x for (x, y) in compressed]
        x_test_ood = torch.cat(l, 0)
        x_test_ood = x_test_ood.detach().numpy()
        x_test_ood = x_test_ood.reshape(1000, 3, 32, 32)
        if distance_norm == "Linf":
            distance = np.max(np.abs(original_images - x_test_ood), axis=(1, 2, 3))  # linf
        if distance_norm == "L2":
            distance = np.sqrt(np.sum((original_images - x_test_ood) ** 2, axis=(1, 2, 3)))  # l2

        if np.min(distance) <= global_min_distance:
            print(f"current number:{componentnumber} current minimum distance: {np.min(distance)}")
            break

        print(f"current number:{componentnumber} current minimum distance: {np.min(distance)}", end="\r")
        componentnumber += 1
        transform.compress_image.n_comps = componentnumber


def calc_auc(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    return auc(fpr, tpr)

def main(params, device):
    print(params)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    fulltestloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
                                                 shuffle=False, num_workers=1)
    testset = split_data(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
                                             shuffle=False, num_workers=1)





    #model = ResNet50(10)
    #print("Load model")
    #model.load_state_dict(torch.load(params.fname))
    #model = Cifar10Wrapper(model)
    #model.to(device)


    model_name = params.model_name
    from resnet import ResNet, BasicBlock, PreActResNet, PreActBlock

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "ramp":
        backbone = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=10) # PreActResNet-18
        model = torch.nn.Sequential(backbone)
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        model_path = "ramp.pt"
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = {f"module.0.{k}": v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
        model.eval()
        del checkpoint
    elif model_name == "hat":
        backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)  # ResNet-18
        model = torch.nn.Sequential(backbone)
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        model_path = "hat.pt"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        del checkpoint
    elif model_name == "ratio":
        model = ResNet50(10)
        print("Load model")
        model.load_state_dict(torch.load(params.fname))
        model = Cifar10Wrapper(model)
        model.to(device)

    if params.distance_norm == "Linf":
        attack_in = LinfSCOREPGD(steps=params.steps,
                               abs_stepsize=2.5 * params.eps_in / params.steps,
                               random_start=True, loss=params.obj)

        params.abs_stepsize_in = 2.5 * params.eps_in / params.steps
        attack_out = LinfSCOREPGD(steps=params.steps,
                                abs_stepsize=2.5 * params.eps_out / params.steps,
                                random_start=True, loss=params.obj)
        params.abs_stepsize_out = 2.5 * params.eps_out / params.steps
    elif params.distance_norm == "L2":
        attack_in = L2SCOREPGD(steps=params.steps,
                               abs_stepsize=2.5 * params.eps_in / params.steps,
                               random_start=True, loss=params.obj)

        params.abs_stepsize_in = 2.5 * params.eps_in / params.steps
        attack_out = L2SCOREPGD(steps=params.steps,
                                abs_stepsize=2.5 * params.eps_out / params.steps,
                                random_start=True, loss=params.obj)
        params.abs_stepsize_out = 2.5 * params.eps_out / params.steps

    # attack = L2PGD(steps=params.steps, abs_stepsize=params.abs_stepsize, random_start=True)


    labels = get_label_vector(loader=testloader)
    pred_in = predict(model, testloader, device)
    full_pred_in = predict(model, fulltestloader, device)

    #
    l = [x for (x, y) in testloader]
    x_test = torch.cat(l, 0)
    x_test = x_test.detach().numpy()

    componentnumber = params.comps
    final_componentnumber=0
    if params.global_min_distance is not None:

        transform = TensorAndCompression(componentnumber, params.targetlabel, randomseed=params.randomseed, mindistance=None, distancenorm=params.distance_norm)
        compressed = CompressedDataset(root='./data', train=False, preds=full_pred_in, download=True,
                                       transform=transform)
        compressed = split_data(compressed, size=1000)
        componentnumber = 1
        for compnumber in [1] + list(range(10, 150, 10)):
            transform.compress_image.n_comps = compnumber
            l = [x for (x, y) in compressed]
            x_test_ood = torch.cat(l, 0)
            x_test_ood = x_test_ood.detach().numpy()
            x_test_ood = x_test_ood.reshape(1000, 3, 32, 32)

            if params.distance_norm == "Linf":
                distance = np.max(np.abs(x_test - x_test_ood), axis=(1, 2, 3))  # linf
            if params.distance_norm == "L2":
                distance = np.sqrt(np.sum((x_test - x_test_ood) ** 2, axis=(1, 2, 3)))  # l2

            if np.min(distance) <= params.global_min_distance:
                break

            componentnumber = compnumber




        transform.compress_image.n_comps = componentnumber

        while componentnumber < 3072:
            print(f"compnumber in full search: {componentnumber}")
            l = [x for (x, y) in compressed]
            x_test_ood = torch.cat(l, 0)
            x_test_ood = x_test_ood.detach().numpy()
            x_test_ood = x_test_ood.reshape(1000,3,32,32)
            if params.distance_norm == "Linf":
                distance = np.max(np.abs(x_test - x_test_ood), axis=(1,2,3)) #linf
            if params.distance_norm == "L2":
                distance = np.sqrt(np.sum((x_test - x_test_ood) ** 2, axis=(1, 2, 3))) #l2

            if np.min(distance) <= params.global_min_distance:
                if componentnumber==1:
                    final_componentnumber=1
                break

            print(f"current number:{componentnumber} current minimum distance: {np.min(distance)}")
            final_componentnumber = componentnumber
            componentnumber += 1
            transform.compress_image.n_comps = componentnumber

        print(f"final: {final_componentnumber}")
        transform.compress_image.n_comps = final_componentnumber

    #

    if params.pred_fname is not None:
        newmodel = ResNet50(10)
        newmodel.load_state_dict(torch.load(params.pred_fname))
        newmodel = Cifar10Wrapper(newmodel)
        newmodel.to(device)
        full_pred_in = predict(newmodel, fulltestloader, device)

    ood_datasets = get_ood_datasets(params.targetlabel, final_componentnumber, params.randomseed, params.batch_size, testset,
                                    params.ood_filter,mindistance=params.adaptive_min_distance, pred_in=full_pred_in,distancenorm=params.distance_norm)
    print("Attack in distribution...")
    pred_a_in, x_test_adv = predict_adv(model, testloader, device, attack_in, params.eps_in, False, params.n_restarts)
    score_fns = [('msp', msp), ('ml', ml), ('lse', lse), ('ul', ul)]
    y_in = np.ones(pred_in.shape[0])
    stats = []
    for (name, ood_loader) in ood_datasets:
        pred_out = predict(model, ood_loader, device)
        pred_a_out, x_test_ood_adv = predict_adv(model, ood_loader, device, attack_out, params.eps_out, True,
                                                 params.n_restarts)

        pred_softmax = np.max(softmax(pred_a_out, axis=1), axis=1)
        sorted_indices = np.argsort(-pred_softmax, axis=0)
        top_10_indices = sorted_indices[:10]
        #top_10_values = pred_a_out[top_10_indices, 1]

        l = [x for (x, y) in testloader]
        x_test = torch.cat(l, 0)
        x_test = x_test.detach().numpy()
        l = [x for (x, y) in ood_loader]
        x_test_ood = torch.cat(l, 0)
        x_test_ood = x_test_ood.detach().numpy()



        predicted_labels = np.argmax(pred_a_out, axis=1)

        if params.distance_norm == "Linf":
            distance = np.max(np.abs(x_test - x_test_ood), axis=(1, 2, 3))  # linf
        if params.distance_norm == "L2":
            distance = np.sqrt(np.sum((x_test - x_test_ood) ** 2, axis=(1, 2, 3)))

        print_distances(distance)

        attack_distance = np.sqrt(np.sum((x_test - x_test_ood_adv) ** 2,
                                         axis=(1, 2, 3)))

        print_distances(attack_distance)
        accuracy = np.mean(predicted_labels == labels)


        if params.imgdir is not None:
            save_images(distance, x_test,x_test_ood,params,pred_in,pred_out,attack_distance,x_test_ood_adv,pred_a_out,x_test_adv,pred_a_in,top_10_indices)

        for (score_name, score_fn) in score_fns:
            score_in = score_fn(pred_in)
            score_a_in = score_fn(pred_a_in)
            score_out = score_fn(pred_out)
            score_a_out = score_fn(pred_a_out)
            # print(y_in.shape, score_out.shape, score_in.shape)
            y_true = np.concatenate((y_in, np.zeros_like(score_out)), axis=0)
            pred_score = np.concatenate((score_in, score_out), axis=0)
            pred_a_score = np.concatenate((score_in, score_a_out), axis=0)
            pred_aa_score = np.concatenate((score_a_in, score_a_out), axis=0)
            stats.append({'model_fname': params.fname,
                          'components': 'adaptive' if params.adaptive_min_distance is not None else final_componentnumber,
                          'score_fn': score_name,
                          'ds': name,
                          'auc': calc_auc(y_true, pred_score),
                          'w-auc': calc_auc(y_true, pred_a_score),
                          'a-auc': calc_auc(y_true, pred_aa_score),
                          'greater_than_1': np.mean(distance > 1),
                          'min_distance': np.min(distance),
                          'max_distance': np.max(distance),
                          'average_distance': np.mean(distance),
                          'median_distance': np.median(distance),
                          '(after_attack)_greater_than_0.5': np.mean(attack_distance > 0.5),
                          '(after_attack)_min_distance': np.min(attack_distance),
                          '(after_attack)_max_distance': np.max(attack_distance),
                          '(after_attack)_average_distance': np.mean(attack_distance),
                          '(after_attack)_median_distance': np.median(attack_distance),
                          'adaptive_min_distance': params.adaptive_min_distance,
                          'accuracy': accuracy,
                          'global_min_distance': params.global_min_distance,
                          'distance_norm': params.distance_norm,
                          'targetlabel': params.targetlabel,
                          'model_name':model_name
                          })
            print(stats[-1])
        end_time = time.time()
        runtime = end_time - start_time
        hours, remainder = divmod(runtime,3600)
        minute, seconds = divmod(remainder, 60)
        print(f"runtime: {hours} hours {minute} minutes {seconds} seconds")

    if params.out_fname is not None:
        is_exist = os.path.exists(params.out_fname)
        with open(params.out_fname, mode='a+') as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=stats[0].keys())
            if not is_exist:
                writer.writeheader()
            writer.writerows(stats)


def diffscale(diff):
    diffmin = np.min(diff)
    diffmax = np.max(diff)
    diff = (diff - diffmin) / (diffmax - diffmin)
    return diff


if __name__ == '__main__':
    print("A")
    parser = ArgumentParser(description='Main entry point')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fname", type=str, default="ratio_025.pth")
    parser.add_argument("--pred_fname", type=str)
    parser.add_argument("--ood_filter", type=str, default="cifar10-compressed")
    parser.add_argument("--out_fname", type=str)
    parser.add_argument("--obj", type=str, default="ul")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--comps", type=int, default=1)
    parser.add_argument("--targetlabel", type=str, default="1")
    parser.add_argument("--n_restarts", type=int, default=10)
    parser.add_argument("--randomseed", type=int, default=10)
    parser.add_argument("--imgdir", type=str)
    # parser.add_argument("--abs_stepsize", type=float, default=0.1)
    parser.add_argument("--eps_in", type=float, default=8/255) #TODO: 8/255
    parser.add_argument("--eps_out", type=float, default=8/255) #TODO: 8/255
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--adaptive_min_distance", type=float)
    parser.add_argument("--global_min_distance",type=float)
    parser.add_argument("--distance_norm", type=str, default="L2")
    parser.add_argument("--model_name",type=str,default="hat")
    start_time = time.time()
    FLAGS = parser.parse_args()
    np.random.seed(9)
    device = torch.device(("cuda:" + str(FLAGS.gpu)) if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    main(FLAGS, device)
