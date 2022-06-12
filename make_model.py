# -*- coding: utf-8 -*-
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import copy
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from utils.training import local_update, test, accuracy, fed_avg

from skimage.exposure import rescale_intensity
from models.vision import LeNet, AlexNet, ResNet18, weights_init
from utils.args import parser_args
from utils.datasets import get_data
from utils.sampling import label_to_onehot, cross_entropy_for_onehot

def MSE(A, B):
    A = rescale_intensity(1.0 * A, out_range=(0, 1))
    B = rescale_intensity(1.0 * B, out_range=(0, 1))
    return np.mean((A - B) ** 2)

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def TVloss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


if __name__ == '__main__':

    args = parser_args()
    tt = transforms.ToPILImage()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Running on %s" % device)
    
    # prepare dataset
    if args.dataset == "cifar10":
        num_classes = 10
    if args.dataset == "cifar100":
        num_classes = 100

    dst, test_set, dict_users = get_data(dataset=args.dataset,
                                                    data_root = args.data_root,
                                                    iid = True,
                                                    num_users = 10)
    local_train_ldr = DataLoader(dst, batch_size = 16, shuffle=False, num_workers=2)
    
    # prepare model 
    if args.model == "lenet":
        net = LeNet().to(device)
    if args.model == "alexnet":
        net = AlexNet(num_classes=num_classes).to(device)
    if args.model == "resnet":
        net = ResNet18(num_classes=num_classes).to(device)

    # #torch.manual_seed(1234)
    # #net.apply(weights_init)
    # # train and save model in different epoch    
    if not os.path.exists("model_time/"+args.model+"/"+args.dataset):
        os.makedirs("model_time/"+args.model+"/"+args.dataset)
    for i in range(100):  
        local_update(local_train_ldr, net, 0.1)
        checkpoint={'epoch': i,
			'model': net.state_dict()}
        torch.save(checkpoint, "/home/lbw/Code/model_time/"+args.model+"/"+args.dataset+"/"+ 'model_'+str(i)+'.pth')

    exit()


        
        



