import numpy as np
import errno
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from utils.sampling import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS
import skimage as skimage
import skimage.transform as transform

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

    if ds == "imagenet":
        root = '/home/lbw/Data/'

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def get_imagenet(root, train = True, transform = None, target_transform = None):
            if train:
                root = os.path.join(data_root, 'ILSVRC2012/train')
            else:
                root = os.path.join(data_root, 'ILSVRC2012/val')
            return datasets.ImageFolder(root = root,
                                    transform = transform,
                                    target_transform = target_transform)
        transform = transforms.Compose([ transforms.ToTensor(), normalize, Resize((3, 224, 224)) ])

        train_set = get_imagenet(root=root, train=True, transform = transform) #, target_transform= None)
        test_set = get_imagenet(root=root, train=False, transform = transform) #, target_transform= None) 
        # size = (256, 340)
        # for i in train_set:
        #     i[0] = skimage.transform.resize(i[0], size)
        #     i[1] = i[1] 
        # for i in test_set:
        #     i = skimage.transform.resize(i[0], size)
        #     i[1] = i[1]                                                  
    
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

