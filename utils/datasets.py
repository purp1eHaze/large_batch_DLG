import numpy as np
import errno
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from utils.metrics import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS
import skimage as skimage
import skimage.transform as transform
import random


class Resize(object):

    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample):
        # 图像
        # print(sample.shape)
        # print(sample)
        # exit()

        # 使用skitimage.transform对图像进行缩放
        image_new = transform.resize(sample, self.output_size)
        image_new = torch.from_numpy(image_new)
        return image_new


def get_data(dataset, data_root, normalized):
    ds = dataset 
    
    if ds == 'cifar10':
    

        if normalized == False:
            transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
            ])    

        train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform
                                                )
    
    if ds == 'mnist':
        
        if normalized == False:
            transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081,)),
            ])   

        train_set = torchvision.datasets.MNIST(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.MNIST(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform
                                                )

    if ds == "imagenet":

        imagenet_root = "/home/lbw/Code/dlg-master/imgtest"
        
        #transform = transforms.Compose([ transforms.ToTensor()])
        if normalized == False:
            transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]) 

        image_datasets = {x: datasets.ImageFolder(imagenet_root+ '/', transform)
                      for x in ['train', 'val', 'test']}
        train_set = image_datasets['train']

        # n = len(train_set)  
        #  #按比例取随机数列表
        # train_set = torch.utils.data.Subset(train_set, n_test) 

        test_set = image_datasets['test']                             
    
    return train_set, test_set

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

