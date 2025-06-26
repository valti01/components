from typing import Callable, Optional, Tuple, Any
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from tensorflow.keras.datasets import cifar10 as ds
from random import choice


class FaceLandmarksDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root: str,preds , train: bool = True, transform: Optional[Callable] = None,
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
            label = choice([i for i in range(0, 10) if i not in [target]]) # TODO: saját random generátor, seeddel
        elif self.targetlabel == "second":
            sorted_indexes = np.argsort(pred)[::-1]
            label = sorted_indexes[1]
        elif self.targetlabel == "own":
            label = target
        else:
            label = int(self.targetlabel)
        pic = np.transpose(pic.numpy(), (1, 2, 0))
        pic = pic.reshape(3072)
        picr = np.dot(np.dot(pic, self.V[label][:self.n_comps, :].T), self.V[label][:self.n_comps, :])
        picr = picr.reshape((32, 32, 3))
        pic_tensor = torch.from_numpy(picr.transpose((2, 0, 1))).float()
        return pic_tensor

    def __init__(self, n_comps, targetlabel):

        (x_train, y_train), (x_test, y_test) = ds.load_data()
        x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32) / 255
        y_train = y_train.flatten()
        V = []
        for i in range(np.max(y_train) + 1):
            current = x_train[y_train == i]
            M = np.dot(current.T, current)
            U, S, Vtemp = np.linalg.svd(M)
            V.append(Vtemp)
        self.targetlabel = targetlabel
        self.V = V
        self.n_comps = n_comps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class CustomTransform:
    def __init__(self, n_comps, label):
        self.to_tensor = transforms.ToTensor()
        self.compress_image = CompressImage(n_comps, label)

    def __call__(self, img, target, pred):
        img_tensor = self.to_tensor(img)
        compressed_img = self.compress_image(img_tensor, target, pred)
        return compressed_img