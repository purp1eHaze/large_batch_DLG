import numpy as np
import errno
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.sampling import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS

def get_data(dataset, data_root, iid, num_users):
    ds = dataset 
    
    if ds == 'cifar10':
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        # transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
        #                                       transforms.RandomHorizontalFlip(),
        #                                       transforms.ColorJitter(brightness=0.25, contrast=0.8),
        #                                       transforms.ToTensor(),
        #                                       normalize,
        #                                       ])  
        # transform_test = transforms.Compose([transforms.CenterCrop(32),
        #                                      transforms.ToTensor(),
        #                                      normalize,
        #                                      ])
        transform_train = transforms.Compose([ transforms.ToTensor()])  
        transform_test = transforms.Compose([ transforms.ToTensor()])  

        train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
    
    if ds == 'cifar100':
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        # transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
        #                                       transforms.RandomHorizontalFlip(),
        #                                       transforms.ColorJitter(brightness=0.25, contrast=0.8),
        #                                       transforms.ToTensor(),
        #                                       normalize,
        #                                       ])  
        # transform_test = transforms.Compose([transforms.CenterCrop(32),
        #                                      transforms.ToTensor(),
        #                                      normalize,
        #                                      ])
        transform_train = transforms.Compose([ transforms.ToTensor()])  
        transform_test = transforms.Compose([ transforms.ToTensor()])  

        train_set = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
    
    if iid:
        dict_users = cifar_iid(train_set, num_users)
 

    return train_set, test_set, dict_users

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

