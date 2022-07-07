# -*- coding: utf-8 -*-
import argparse
from locale import normalize
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
from utils.training import local_update, test, accuracy
from models.vision import LeNet, LeNet_Imagenet, AlexNet_Imagenet, AlexNet_Cifar, ResNet18
from utils.args import parser_args
from utils.datasets import get_data
from utils.metrics import label_to_onehot, cross_entropy_for_onehot, TVloss, TVloss_l1, MSE
from utils.config import Adam_Config, LBFGS_Config, SGD_Config, Geiping_Config

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def trimmed_avg(x_collection):
    x = x_collection[0].reshape((-1, 1))
    for x_sample in x_collection[1:]:
        x = torch.cat([x, x_sample.reshape((-1, 1))], dim=1)
    med, _ = torch.sort(x, axis=1)
    trimmed_num = args.trimmed_num
    med = torch.mean(med[:, trimmed_num:-trimmed_num], axis=1).reshape(
        (x_collection[0].shape))

    return med

def median(x_collection):
    x = x_collection[0].reshape((-1, 1))
    for x_sample in x_collection[1:]:
        x = torch.cat([x, x_sample.reshape((-1, 1))], dim=1)
    X = torch.median(x, dim=1)
    return X.values.reshape(x_collection[0].shape)

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
    
    dst, test_set = get_data(dataset=args.dataset, data_root = args.data_root, normalized = args.normalized)
    local_train_ldr = DataLoader(dst, batch_size = 32, shuffle=False, num_workers=2) 
   
    model_time = []
    # prepare model 
    def get_model_fn():
        if args.model == "lenet":
            if args.dataset == "imagenet":
                model_fn = LeNet_Imagenet(input_size=input_size).to(device)
            else:
                model_fn = LeNet(input_size=input_size).to(device)
        if args.model == "alexnet":
            if args.dataset == "imagenet":
                model_fn = AlexNet_Imagenet(num_classes=num_classes, input_size = input_size).to(device) # pretrained = True
            else: 
                model_fn = AlexNet_Cifar(num_classes=num_classes, input_size = input_size).to(device)
        if args.model == "resnet":
            if args.dataset == "imagenet":
                #net = ResNet18(num_classes=num_classes, imagenet = True).to(device)
                model_fn = torchvision.models.resnet18(num_classes =10, pretrained=False).to(device)
            else:
                model_fn = ResNet18(num_classes=num_classes, imagenet = False).to(device)
        return model_fn
    
    dir = "/home/lbw/Code/model_time/"+args.model+"/"+args.dataset+"/"

    if args.mode == "random": 
        for i in range(args.attack_iters):
            net = get_model_fn()
            model_time.append(net.state_dict())

    if args.mode == "trained":
        net = get_model_fn()
        learning_optimizer = torch.optim.SGD(net.parameters(), 0.0001, 
                        momentum=0.9,
                        weight_decay=0.0005)
        # learning_optimizer = torch.optim.Adam(net.parameters(),
        #             0.0001,
        #             betas=(0.9, 0.999),
        #             eps=1e-08,
        #             weight_decay=0,
        #             amsgrad=False)
        for i in range(args.attack_iters):   
            local_update(local_train_ldr, net, learning_optimizer)
            loss, acc = test(net, local_train_ldr)
            print("training loss: %.4f"  % loss)
            print("training acc: %.3f"  % acc)
            model_time.append(net.state_dict())
    
    if args.mode == "load_trained":
        for i in range(1, 2, 1):
            model_dict = torch.load(dir+ "model_"+str(i)+".pth")['model']
            model_time.append(model_dict)

    # make image folder
    if not os.path.exists("images/"+args.model+"_"+args.dataset):
        os.makedirs("images/"+args.model+"_"+args.dataset) 
    # histgram folder
    if not os.path.exists("hists/"+args.model+"_"+args.dataset):
        os.makedirs("hists/"+args.model+"_"+args.dataset)
    
    # load image and label 
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()
    criterion = cross_entropy_for_onehot

    img_index = args.index
    gt_data = dst[img_index][0].to(device) # 0 for label, 1 for label
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    data_size = gt_data.size()
    gt_data = gt_data.view(1, *data_size)
    
    # batch selection to be attacked
    for i in range(args.batch_size-1):
        gt_data = torch.cat([gt_data, dst[img_index+i+1][0].view(1, *data_size).to(device)], dim=0)
        gt_label = torch.cat([gt_label, torch.Tensor([dst[img_index+i+1][1]]).long().view(1, ).to(device)], dim=0)

    gt_onehot_label = label_to_onehot(gt_label, num_classes= num_classes)

    # generate dummy data and label
    dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
    #dummy_label = torch.rand(gt_onehot_label.size()).to(device).requires_grad_(True)
    dummy_label = gt_onehot_label


    history = []
    x_avg = dummy_data.clone().detach()
    #y_avg = dummy_label.clone().detach()

    # set optimizer for deep leakage
    #optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = 1)

    # store original gradient
    original_dy_dx = []
    for i in range(len(model_time)):
        net.load_state_dict(model_time[i])
        net.eval()
        pred = net(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx.append(list((_.detach().clone() for _ in dy_dx)))

    if args.optim == "adam": 
        config = Adam_Config 
    if args.optim == "LBFGS":     
        config = LBFGS_Config
    if args.optim == "SGD":
        config = SGD_Config
    if args.optim == "geiping":
        config = Geiping_Config
  
    for iters in range(args.epochs): # default =300
    
        x_collection = []
        #y_collection = []
        loss = []

        # if iters == int(args.epochs * 3/8) or iters == int(args.epochs * 5/8) or iters == int(args.epochs * 7/8):
        #     args.lr = args.lr * 0.3

        for i in range(len(model_time)):
            

            # load model
            net.load_state_dict(model_time[i])

            # load data
            dummy_data = x_avg.requires_grad_(True)
            #dummy_label = y_avg.requires_grad_(True)

            # set optimizer for deep leakage
            # optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = args.lr)
            optimizer = torch.optim.LBFGS([dummy_data], lr = args.lr,  max_iter=20)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[args.epochs // 2.667, args.epochs // 1.6,

                                                                args.epochs // 1.142],
                                                    gamma=0.3)  # 3/8 5/8 7/8

            # if i==0 and iters ==0:
            #     for j in range(len(original_dy_dx)):
            #         plot_his(original_dy_dx[j])
            #         plt.savefig('hists/' + args.model + '_' + args.dataset + '/' + str(iters) + "_"+str(j) + '.png')
            #         plt.close()   
            
            def closure():
                optimizer.zero_grad()
                dummy_pred = net(dummy_data)
                #dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                # dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
                dummy_loss = criterion(dummy_pred, gt_onehot_label) 
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx[0:], original_dy_dx[i][0:]):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                # tvloss = 0.01 * TVloss_l1(dummy_data) + 0.01 * TVloss(dummy_data)
                # tvloss.backward() 
                return grad_diff  

            # dummy_data = dummy_data + dummy_data.grad 
            optimizer.step(closure)
            scheduler.step()
            x_collection.append(dummy_data.detach())
            #y_collection.append(dummy_label.detach())
            loss.append(closure().item())

        avg_loss = 0 
        for i in range(len(loss)):
            avg_loss += loss[i]
        avg_loss /= len(loss)

        x_avg = median(x_collection)
        blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=0.5)
        x_avg = blurrer(x_avg)
        #y_avg = avg(y_collection)   
    
        if (iters+1) % args.epoch_interval == 0:  #default = 10
            rec_mse = MSE(gt_data.cpu().detach().numpy(), dummy_data.cpu().detach().numpy())
            tvloss = TVloss(dummy_data)
            print(iters, "gradloss: %.4f"  % avg_loss, "mseloss: %.5f" % rec_mse, "tvloss-%.5f" % tvloss)
            history.append(dummy_data.cpu())

    subimage_size = int(args.epochs/args.epoch_interval)
    # save image
    for i in range(args.batch_size):
        plt.figure(figsize=(30, 30))
        plt.imshow(tt(history[-1][i]))
        plt.savefig("images/"+str(i)+"_whole.jpg")
        plt.close()

        plt.figure(figsize=(30, 20))
        for j in range(subimage_size): # default = 30            
            plt.subplot(int(subimage_size/10), 10, j+1)
            plt.imshow(tt(history[j][i]))
            # plt.title("iter=%d" % (i * 100))
            # plt.axis('off')      
        plt.savefig("images/"+str(i)+".jpg")
        plt.close() 
        





