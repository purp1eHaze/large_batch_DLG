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
from models.vision import LeNet, AlexNet_Imagenet, AlexNet_Cifar, ResNet18, weights_init
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
    if args.dataset == "imagenet":
        num_classes = 20
        input_size = 224
    if args.dataset == "cifar10":
        num_classes = 10
        input_size = 32
    if args.dataset == "cifar100":
        num_classes = 100
        input_size = 32
    
    dst, test_set, dict_users = get_data(dataset=args.dataset,
                                                    data_root = args.data_root,
                                                    iid = True,
                                                    num_users = 10)
    local_train_ldr = DataLoader(dst, batch_size = 32, shuffle=True, num_workers=2)

    # prepare model 
    if args.model == "lenet":
        net = LeNet(input_size=input_size).to(device)
    if args.model == "alexnet":
        if args.dataset == "imagenet":
            net = AlexNet_Imagenet(num_classes=num_classes, input_size = input_size).to(device) # pretrained = True
        else: 
            net = AlexNet_Cifar(num_classes=num_classes, input_size = input_size).to(device)
    if args.model == "resnet":
        if args.dataset == "imagenet":
            net = ResNet18(num_classes=num_classes, imagenet = True).to(device)
        else:
            net = ResNet18(num_classes=num_classes, imagenet = False).to(device)
        
    # learning_optimizer = torch.optim.SGD(net.parameters(), 0.01, 
    #                         momentum=0.9,
    #                         weight_decay=0.0005)
     
    learning_optimizer = torch.optim.Adam(net.parameters(),
                0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)

    img_index = args.index

    # make image folder
    for j in range(args.batch_size):
        img_path = os.path.join('assets', str(j)) 
        if not os.path.exists(img_path): 
            os.makedirs(img_path) 
    
    # load image and label 
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()
    criterion = cross_entropy_for_onehot


    model_time = []
    dir = "/home/lbw/Code/model_time/"+args.model+"/"+args.dataset+"/"

    if not os.path.exists("/home/lbw/Code/model_time/"+args.model+"/"+args.dataset):
        os.makedirs("/home/lbw/Code/model_time/"+args.model+"/"+args.dataset)

    for i in range(100):  
        local_update(local_train_ldr, net, learning_optimizer)
        loss, acc = test(net, local_train_ldr)
        print("training loss: %.4f"  % loss)
        print("training acc: %.3f"  % acc)
        checkpoint={'epoch': i,
			'model': net.state_dict()}
        torch.save(checkpoint, "/home/lbw/Code/model_time/"+args.model+"/"+args.dataset+"/"+ 'model_'+str(i)+'.pth')

    exit()

    for i in range(20):  
        local_update(local_train_ldr, net, learning_optimizer)
        loss, acc = test(net, local_train_ldr)
        print("training loss: %.4f"  % loss)
        print("training acc: %.3f"  % acc)

        model_time.append(net.state_dict())
  
   
        



