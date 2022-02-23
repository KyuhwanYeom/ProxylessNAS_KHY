import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision.datasets import CIFAR10
from utils import Cutout

class Cifar10DataProvider():

    def __init__(self, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None, cutout=False):
        train_transforms = self.build_train_transform(
            distort_color, resize_scale)
        train_dataset = CIFAR10(root='data', train=True,
                                transform=train_transforms, download=True)
        valid_dataset = CIFAR10(root='data', train=False, transform=transforms.Compose([
            transforms.Resize(self.resize_value),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            self.normalize,
        ]))
        if cutout:
            train_transforms.transforms.append(Cutout(n_holes=1, length=16))
            
        if valid_size is not None:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True
            )

        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True
            )

        self.test = torch.utils.data.DataLoader(
            CIFAR10('data', train=False, transform=transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                self.normalize,
            ])), batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                self.image_size, scale=(resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        return train_transforms

    @property
    def resize_value(self):
        return 32

    @property
    def image_size(self):
        return 32
