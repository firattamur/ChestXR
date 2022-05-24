import os
import json
import torch
from torch.utils.data import ConcatDataset
from dataset.chestxr14_dataset import ChestXRDataset
from torchvision import datasets, transforms
from tqdm import tqdm


def load_transforms(config):
    """
    Preprocessing transformation for train and test datasets.
    """

    imgsize = config.img_size

    transforms_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(padding=torch.randint(imgsize // 2, (1,)).item()),
            transforms.Resize((imgsize, imgsize)),
            lambda x: (1 - x),
        ]
    )

    transforms_test = transforms.Compose(
        [
            transforms.Resize((imgsize, imgsize)),
            lambda x: (1 - x),
        ]
    )

    return transforms_train, transforms_test


def load_train_dataset(config):
    """

    Load train dataset and apply transformations.
    """

    dtrain = ChestXRDataset(config)

    # load data loader
    dloadtrain = torch.utils.data.DataLoader(dtrain, batch_size=config.NBATCH,
                                            num_workers=config.NWORKERS, shuffle=True)

    return dloadtrain


def load_valid_dataset(config):
    """

    Load validation dataset and apply transformations.
    """

    dvalid = ChestXRDataset(config)

    # load data loaders
    dloadvalid = torch.utils.data.DataLoader(dvalid, batch_size=config.NBATCH,
                                             num_workers=config.NWORKERS, shuffle=True)

    return dloadvalid


def load_test_dataset(config):
    """

    Load test dataset and apply transformations.
    """

    dtest = ChestXRDataset(config)

    # load data loaders
    dloadtest = torch.utils.data.DataLoader(dtest, batch_size=1,
                                            num_workers=config.NWORKERS, shuffle=True)

    return dloadtest

