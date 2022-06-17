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

def diff(x_collection, y_collection, x_avg, y_avg):

    x_diff = []
    y_diff = []
    for x_sample in x_collection:
        x_diff.append(x_sample.unsqueeze(0) - x_avg)
    for y_sample in y_collection:
        y_diff.append(y_sample.unsqueeze(0) - y_avg)
    
    return x_diff, y_diff

def avg(x_collection):

    x =  x_collection[0].unsqueeze(0)
    for x_sample in x_collection[1:]:
        x = torch.cat([x, x_sample.unsqueeze(0)], dim=0)

    x_avg = torch.mean(x, dim=0, keepdim = False)
    
    return x_avg

if __name__ == '__main__':

    args = parser_args()
    tt = transforms.ToPILImage()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Running on %s" % device)
    
    # prepare dataset
    if args.dataset == "imagenet":
        num_classes = 10
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
    local_train_ldr = DataLoader(dst, batch_size = 64, shuffle=False, num_workers=2)
    test_ldr = DataLoader(test_set, batch_size = 64, shuffle=False, num_workers=2)
    
    # prepare model 
    if args.model == "lenet":
        net = LeNet(input_size=input_size).to(device)
    if args.model == "alexnet":
        net = AlexNet(num_classes=num_classes, input_size = input_size).to(device)
    if args.model == "resnet":
        net = ResNet18(num_classes=num_classes, input_size=input_size).to(device)

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

    gt_data = dst[img_index][0].to(device) # 0 for label, 1 for label
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    data_size = gt_data.size()
    gt_data = gt_data.view(1, *data_size)
    
    # print(gt_data.shape)
    # print(dst[img_index][0].view(1, *data_size).shape)
    # print(dst[img_index][0].shape)
    # print(dst[img_index+1][0].shape)
    # print(dst[img_index+2][0].shape)
    # print(dst[img_index+3][0].shape)
    # print(dst[img_index+4][0].shape)
    # plt.figure(figsize=(30, 20))
    # plt.imshow(tt(dst[img_index+4][0])) 
    # plt.savefig("test.jpg")
    # plt.close()    
    # exit()

    # batch selection to be attacked
    for i in range(args.batch_size-1):
        gt_data = torch.cat([gt_data, dst[img_index+i+1][0].view(1, *data_size).to(device)], dim=0)
        gt_label = torch.cat([gt_label, torch.Tensor([dst[img_index+i+1][1]]).long().view(1, ).to(device)], dim=0)

    gt_onehot_label = label_to_onehot(gt_label, num_classes= 10)

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    # set optimizer for deep leakage
    #optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = 1)

    history = []
    x_avg = dummy_data.clone().detach()
    y_avg = dummy_label.clone().detach()

    model_time = []
    dir = "/home/lbw/Code/model_time/"+args.model+"/"+args.dataset+"/"

    # for i in range(1, 40, 1):
    #     model_dict = torch.load(dir+ "model_"+str(i)+".pth")['model']
    #     model_time.append(model_dict)

    for i in range(10):  
        local_update(local_train_ldr, net, 0.1)
        loss, acc = test(net, local_train_ldr)
        print("training loss: %.4f"  % loss)
        print("training acc: %.3f"  % acc)

        model_time.append(net.state_dict())
  
    for iters in range(args.epochs): # default =300
    
        x_collection = []
        y_collection = []
        # gradient_x = []
        # gradient_y = []
        loss = []
            
        for i in range(len(model_time)):
            
            #print("1")

            # load model 
            net.load_state_dict(model_time[i]) 
            # load data
            dummy_data = x_avg.requires_grad_(True)
            dummy_label = y_avg.requires_grad_(True)
            # set optimizer for deep leakage
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = args.lr)
            pred = net(gt_data)
            y = criterion(pred, gt_onehot_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            def closure():
                optimizer.zero_grad()
                dummy_pred = net(dummy_data) 
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                tvloss = 0.01 * TVloss(dummy_data)
                tvloss.backward() 
                return grad_diff  

            # dummy_data = dummy_data + dummy_data.grad 
            optimizer.step(closure)
            x_collection.append(dummy_data.detach())
            y_collection.append(dummy_label.detach())
            loss.append(closure().item())

        avg_loss = 0 
        for i in range(len(loss)):
            avg_loss += loss[i]
        avg_loss /= len(loss)

        diff_x, diff_y = diff(x_collection, y_collection, x_avg, y_avg)
    
        x_avg = avg(x_collection)
        y_avg = avg(y_collection)   
    
        if (iters+1) % args.epoch_interval == 0:  #default = 10
            #rec_mse = MSE(gt_data.cpu().detach().numpy(), dummy_data.cpu().detach().numpy())
            print(iters, "gradloss: %.4f"  % avg_loss)
            history.append(dummy_data.cpu())

    subimage_size = int(args.epochs/args.epoch_interval)
    # save image
    for i in range(args.batch_size):
        plt.figure(figsize=(30, 20))
        for j in range(subimage_size): # default = 30            
            plt.subplot(int(subimage_size/10), 10, j+1)
            plt.imshow(tt(history[j][i]))
            # plt.title("iter=%d" % (i * 100))
            # plt.axis('off')      
        plt.savefig("assets/"+str(i)+"/"+"rec.jpg")
        plt.close()         
        



