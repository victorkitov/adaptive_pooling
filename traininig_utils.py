from torch.utils.tensorboard import SummaryWriter
import typing as tp
from pathlib import Path
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
import torchvision


class DatasetLoader:
    def __init__(self, dataset_name: str, batch_size: int):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_classes = None

    def create_dataloaders(self) -> tp.Dict[str, DataLoader]:
        if self.dataset_name == 'CIFAR10':
            self.num_classes = 10
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            loaders = {
                'train': DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=4),
                'test': DataLoader(test_data,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   num_workers=4),
            }
            return loaders
        elif self.dataset_name == "mnist":
            self.num_classes = 10
            transform = transforms.Compose([transforms.ToTensor(),
                                            torchvision.transforms.Pad(2, padding_mode="edge")])
                                            # transforms.Normalize((0.5), (0.5, 0.5, 0.5))])

            train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            loaders = {
                'train': DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=4),
                'test': DataLoader(test_data,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   num_workers=4),
            }
            return loaders
        elif self.dataset_name == "fashion_mnist":
            self.num_classes = 10
            transform = transforms.Compose([transforms.ToTensor(),
                                            torchvision.transforms.Pad(2, padding_mode="edge")])

            train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            loaders = {
                'train': DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=4),
                'test': DataLoader(test_data,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   num_workers=4),
            }
            return loaders
        elif self.dataset_name == "SVHN":
            self.num_classes = 10
            transform = transforms.Compose([transforms.ToTensor()])
                                            # torchvision.transforms.Pad(2, padding_mode="edge")])

            train_data = datasets.SVHN(root='data/', split="train", download=True, transform=transform)
            test_data = datasets.SVHN(root='data/', split="test", download=True, transform=transform)
            loaders = {
                'train': DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=4),
                'test': DataLoader(test_data,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   num_workers=4),
            }
            return loaders
        elif self.dataset_name == "CIFAR100":
            self.num_classes = 100
            transform = transforms.Compose([transforms.ToTensor()])
                                            # torchvision.transforms.Pad(2, padding_mode="edge")])

            train_data = datasets.CIFAR100(root='data/', train=True, download=True, transform=transform)
            test_data = datasets.CIFAR100(root='data/', train=False, download=True, transform=transform)
            loaders = {
                'train': DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=4),
                'test': DataLoader(test_data,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   num_workers=4),
            }
            return loaders
        elif self.dataset_name == "KMNIST":
            self.num_classes = 100
            transform = transforms.Compose([transforms.ToTensor(),
                                            torchvision.transforms.Pad(2, padding_mode="edge")])

            train_data = datasets.KMNIST(root='data/', train=True, download=True, transform=transform)
            test_data = datasets.KMNIST(root='data/', train=False, download=True, transform=transform)
            loaders = {
                'train': DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=4),
                'test': DataLoader(test_data,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   num_workers=4),
            }
            return loaders


class TBLogger:
    def __init__(self, name: str):
        self.writer = SummaryWriter(log_dir='./final/' + name + '_' + datetime.now().strftime("%H:%M:%S"))

    def add_scalar(self, name, value, iteration):
        self.writer.add_scalar(name, value, iteration)
