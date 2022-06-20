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


# cp -r /home/jjlin/datasets/ILSVRC2012/train/n01440764/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n01860187/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n02097474/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n02165105/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n02669723/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n03042490/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n03532672/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n03887697/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n04243546/  /home/lbw/Data/ILSVRC2012/train/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/train/n04552348/  /home/lbw/Data/ILSVRC2012/train/
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n01440764/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n01860187/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n02097474/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n02165105/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n02669723/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n03042490/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n03532672/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n03887697/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n04243546/  /home/lbw/Data/ILSVRC2012/val/ &
# cp -r /home/jjlin/datasets/ILSVRC2012/val/n04552348/  /home/lbw/Data/ILSVRC2012/val/
