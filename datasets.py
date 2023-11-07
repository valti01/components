from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torch
import numpy as np
from scipy import ndimage
from PIL import Image


class UniformNoiseDataset(Dataset):
    def __init__(self, n_samples, size=(32, 32, 3), low=0, high=1, seed=9) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.rnd = np.random.RandomState(seed)
        self.low = low
        self.high = high
        self.size = size
        self.transform = transforms.ToTensor()

    def __getitem__(self, index: int) -> T_co:
        return self.transform(np.array(self.rnd.uniform(self.low, self.high, self.size), dtype=np.float32)), 0

    def __len__(self) -> int:
        return self.n_samples


class SmoothedNoise(Dataset):
    def __init__(self, src_ds, seed=9, size=(32, 32, 3), low=0, high=1, min_sigma=1.0, max_sigma=2.5) -> None:
        super().__init__()
        self.src_ds = src_ds
        self.n_samples = len(self.src_ds)
        self.low = low
        self.high = high
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.size = size
        self.rnd = np.random.RandomState(seed)
        self.transform = transforms.ToTensor()

    def __getitem__(self, index: int) -> T_co:
        p = self.rnd.uniform(0, 1)
        if p <= 0.5:
            sample, y = self.src_ds[index]
            shape = sample.shape
            flatten = torch.flatten(sample)
            idxs = torch.randperm(flatten.shape[0])
            sample = torch.reshape(flatten[idxs], shape)
            # sample=torch.reshape(flatten,shape)
            sample = torch.moveaxis(sample, 0, -1)
        else:
            sample = self.rnd.uniform(self.low, self.high, self.size)

        smoothedimage = np.array(sample)
        smoothedimage = ndimage.gaussian_filter(smoothedimage, self.rnd.uniform(self.min_sigma, self.max_sigma))
        mi = np.min(smoothedimage)
        ma = np.max(smoothedimage)
        normalized_img = (smoothedimage - mi) / (ma - mi) * self.high + self.low
        # normalized_img=sample
        # img=np.moveaxis(normalized_img,-1,0)
        return self.transform(np.array(normalized_img, dtype=np.float32)), 0

    def __len__(self) -> int:
        return self.n_samples


class ImageNetMinusC10(Dataset):
    def __init__(self, fname="./data/imagenet_data/test_imagenet_cifar_exclusive_1k.npy",
                 transforms=transforms.Compose([transforms.Resize(32),
                                                transforms.ToTensor()])) -> None:
        super().__init__()
        self.fname = fname
        self.transforms = transforms
        self.x = np.load(fname)

    def __getitem__(self, index: int) -> T_co:
        img = Image.fromarray(self.x[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, 0

    def __len__(self) -> int:
        return len(self.x)
