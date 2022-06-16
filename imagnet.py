import os
import torch
import torchvision.datasets as datasets
from torch.utils.data import Subset
# class CustomSubset(Subset):
#      '''A custom subset class'''


root = '/home/lbw/Data/ILSVRC2012'
def get_imagenet(root, train = True, transform = None, target_transform = None):
    if train:
        root = os.path.join(root, 'train')
    else:
        root = os.path.join(root, 'val')
    return datasets.ImageFolder(root = root,
                               transform = transform,
                               target_transform = target_transform)

train = get_imagenet(root=root, train=True, transform = None, target_transform= None)
test = get_imagenet(root=root, train=False, transform = None, target_transform= None)

print(train)
print(test)
print(train.classes)
print(test.classes)
print(test.class_to_idx)

# print(test.imgs)