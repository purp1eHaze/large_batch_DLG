# -*- coding: utf-8 -*-
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import copy
import inversefed
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from utils.training import local_update, test, accuracy, fed_avg
from skimage.exposure import rescale_intensity
from models.vision import LeNet, LeNet_Imagenet, AlexNet_Imagenet, AlexNet_Cifar#, ResNet18
from utils.args import parser_args
from utils.datasets import get_data
from utils.metrics import label_to_onehot, cross_entropy_for_onehot, Classification, psnr, reconstruction_costs

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def MSE(A, B):
    MSE = 0
    for j in range(args.batch_size):
        A1 = rescale_intensity(1.0 * A[j], out_range=(0, 1))
        B1 = rescale_intensity(1.0 * B[j], out_range=(0, 1))
        MSE += np.mean((A1 - B1) ** 2)
    return np.mean(MSE)

# def MSE(A, B):
#     A = rescale_intensity(1.0 * A, out_range=(0, 1))
#     B = rescale_intensity(1.0 * B, out_range=(0, 1))
#     return np.mean((A - B) ** 2)

def plot_his(A):
    plt.hist(torch.flatten(A).cpu().detach().numpy(), bins='auto', density=True)

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def TVloss_l1(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    # count_h = _tensor_size(x[:, :, 1:, :])
    # count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])).sum()
    w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1])).sum()
    return torch.sum(h_tv + w_tv) / batch_size

def TVloss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def TV(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

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
    local_train_ldr = DataLoader(dst, batch_size = 32, shuffle=False, num_workers=2)
   
    model_dict = []
    # prepare model 
    for i in range(10):
        if args.model == "lenet":
            if args.dataset == "imagenet":
                net = LeNet_Imagenet(input_size=input_size).to(device)
            else:
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
        
        model_dict.append(net.state_dict())
    net.eval()
    #print(next(net.parameters()).dtype)

        
    learning_optimizer = torch.optim.SGD(net.parameters(), 0.0001, 
                            momentum=0.9,
                            weight_decay=0.0005)
     
    # learning_optimizer = torch.optim.Adam(net.parameters(),
    #             0.0001,
    #             betas=(0.9, 0.999),
    #             eps=1e-08,
    #             weight_decay=0,
    #             amsgrad=False)


    # make image folder
    for j in range(args.batch_size):
        img_path = 'images/' 
        if not os.path.exists(img_path): 
            os.makedirs(img_path) 

    if not os.path.exists("hists/"+args.model+"_"+args.dataset):
        os.makedirs("hists/"+args.model+"_"+args.dataset)
    
    # load image and label 
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()
    criterion = cross_entropy_for_onehot

    
    # batch selection to be attacked
    ground_truth, labels = [], []
    target_id_ = 1
    print(target_id_)
    while len(labels) < args.batch_size:
        img, label = local_train_ldr.dataset[target_id_]
        target_id_ += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=device))
            ground_truth.append(img.to(device))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)  
    img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

    # generate dummy data and label
    if args.normalized == True:
        dm = torch.as_tensor([0.4802, 0.4481, 0.3975])[:, None, None].to(device)
        ds = torch.as_tensor([0.2302, 0.2265, 0.2262])[:, None, None].to(device)
    #dummy_data = torch.rand(gt_data.size())
    dummy_data = torch.randn((args.batch_size, *img_shape)).to(device).requires_grad_(True) 
    #dummy_label = torch.rand(gt_onehot_label.size()).to(device).requires_grad_(True) # known for convinience

    assert labels.shape[0] == args.batch_size

    # set optimizer for deep leakage
    #optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = 1)
    history = []
    x_avg = dummy_data.clone().detach()
    #y_avg = dummy_label.clone().detach()

    model_time = []
    dir = "/home/lbw/Code/model_time/"+args.model+"/"+args.dataset+"/"
    
    # print(net.state_dict()["features.2.conv.weight"][0])

    # for i in range(1, 2, 1):
    #     model_dict = torch.load(dir+ "model_"+str(i)+".pth")['model']
    #     # print(model_dict["features.2.conv.weight"][0])
    #     # exit()
    #     model_time.append(model_dict)

    for i in range(1):  
        # local_update(local_train_ldr, net, learning_optimizer)
        # loss, acc = test(net, local_train_ldr)
        # print("training loss: %.4f"  % loss)
        # print("training acc: %.3f"  % acc)
        # model_time.append(net.state_dict())
        model_time.append(model_dict[i])

# ----------------------------------------------------geiping mod------------------------------------------------------------

    config = dict(
        signed=True,  # Gradient Sign args.signed
        boxed= True,   #args.boxed,
        cost_fn=args.cost_fn,
        indices="def",
        weights="equal",
        lr=0.1,
        optim="adam",   #args.optimizer,
        restarts=1, #args.restarts,
        max_iterations=15000,
        total_variation=1e-4, #args.tv,
        init="randn",
        filter="none",
        lr_decay=True,
        scoring_choice="loss",
    )
    print(config)
    print(torch.backends.cudnn.benchmark)
    print(torch.backends.cudnn.deterministic)
    
    net.zero_grad()
    loss_fn = Classification() 
    target_loss, _, _ = loss_fn(net(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, net.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    full_norm = torch.stack([g.norm() for g in input_gradient]).mean()

    # Original Geiping Mod
    # rec_machine = inversefed.GradientReconstructor(net, (dm, ds), config, num_images=args.batch_size)
    # output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=False)
    # exit()


    #net.load_state_dict(model_time[i]) 
    dummy_data = dummy_data.requires_grad_(True)
    #dummy_label = y_avg.requires_grad_(True)

    # set optimizer for deep leakage
    if args.optim == 'adam':
        optimizer = torch.optim.Adam([dummy_data], lr=args.lr)
    elif args.optim == 'sgd':  # actually gd
        optimizer = torch.optim.SGD([dummy_data],  lr=args.lr) # momentum=0.9,  nesterov=True,
    elif args.optim == 'LBFGS':
        optimizer = torch.optim.LBFGS([dummy_data])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[args.epochs // 2.667, args.epochs  // 1.6,

                                                                    args.epochs  // 1.142], gamma=0.3)   # 3/8 5/8 7/8

    loss_dlg = torch.nn.CrossEntropyLoss(reduction='mean')
   
    for iters in range(args.epochs): # default =300
        
        loss = []
        def closure():    
            optimizer.zero_grad()
            net.zero_grad()
            loss = loss_dlg(net(dummy_data), labels)
            gradient = torch.autograd.grad(loss, net.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient, cost_fn=args.cost_fn, indices="def", weights="equal")
            rec_loss += 1e-4 * TV(dummy_data)
            rec_loss.backward()   
            # if self.config['signed']:
            dummy_data.grad.sign_()
            return rec_loss
    
        # dummy_data = dummy_data + dummy_data.grad 
        optimizer.step(closure)
        scheduler.step()
        loss.append(closure().item())

        # boxed
        dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

        avg_loss = 0 
        for i in range(len(loss)):
            avg_loss += loss[i]
        avg_loss /= len(loss)

        if (iters+1) % args.epoch_interval== 0:  #default = 10
            #denormal_ground_truth  = torch.clamp(ground_truth * ds + dm, 0, 1) # denormalized
            #denormal_dummy_data  = torch.clamp(dummy_data * ds + dm, 0, 1) # denormalized
            rec_mse = MSE(ground_truth.cpu().detach().numpy(), dummy_data.cpu().detach().numpy())
            tvloss = TVloss(dummy_data)
            print(iters, "gradloss: %.4f"  % avg_loss, "mseloss: %.5f" % rec_mse, "tvloss-%.5f" % tvloss)
            
            #history.append(denormal_ground_truth.cpu())
            if ((iters+1)/args.epochs > 0.7):
                history.append(dummy_data.cpu())

    subimage_size = len(history)
    # save image
    for i in range(args.batch_size):
        plt.figure(figsize=(30, 20))
        for j in range(subimage_size): # default = 30            
            plt.subplot(int(subimage_size/1), 1, j+1)
            plt.imshow(tt(history[j][i]))
            # plt.title("iter=%d" % (i * 100))
            # plt.axis('off')      
        plt.savefig("images/"+str(i)+".jpg")
        plt.close() 
    
    exit()



# ----------------------------------------------------multi-epoch mod------------------------------------------------------------

    for iters in range(args.epochs): # default =300
    
        x_collection = []
        y_collection = []
        # gradient_x = []
        # gradient_y = []
        loss = []
        loss_dlg = torch.nn.CrossEntropyLoss(reduction='mean')
            
        for i in range(len(model_time)):
            
            # load model 
            net.load_state_dict(model_time[i]) 
            net.zero_grad()
            loss_fn = Classification() 
            target_loss, _, _ = loss_fn(net(ground_truth), labels)
            input_gradient = torch.autograd.grad(target_loss, net.parameters())
            input_gradient = [grad.detach() for grad in input_gradient]
            full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
            #print(f"Full gradient norm is {full_norm:e}.")

            # load averaged data
            dummy_data = x_avg.requires_grad_(True)
            #dummy_label = y_avg.requires_grad_(True)

            # set optimizer for deep leakage
            if args.optim == 'adam':
                optimizer = torch.optim.Adam([dummy_data], lr=args.lr)
            elif args.optim == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([dummy_data], lr=args.lr) # momentum=0.9,  nesterov=True,
            elif args.optim == 'LBFGS':
                optimizer = torch.optim.LBFGS([dummy_data])
            else:
                raise ValueError()

            if args.normalized == True:
                dm = torch.as_tensor([0.4802, 0.4481, 0.3975])[:, None, None].to(device)
                ds = torch.as_tensor([0.2302, 0.2265, 0.2262])[:, None, None].to(device)

            #print(x_trial.shape)
            def closure():    
                optimizer.zero_grad()
                net.zero_grad()
                loss = loss_dlg(net(dummy_data), labels)

                gradient = torch.autograd.grad(loss, net.parameters(), create_graph=True)
                rec_loss = reconstruction_costs([gradient], input_gradient, cost_fn=args.cost_fn, indices="def", weights="equal")
                rec_loss += 1e-4 * TV(dummy_data)
                rec_loss.backward()   
                # if self.config['signed']:
                #     x_trial.grad.sign_()
                return rec_loss
        
            # dummy_data = dummy_data + dummy_data.grad 
            optimizer.step(closure)
            x_collection.append(dummy_data.detach())
            #y_collection.append(dummy_label.detach())
            loss.append(closure().item())


        # boxed
        dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

        avg_loss = 0 
        for i in range(len(loss)):
            avg_loss += loss[i]
        avg_loss /= len(loss)

        x_avg = avg(x_collection)
        #y_avg = avg(y_collection)   
    
        if (iters+1) % args.epoch_interval == 0:  #default = 10
            rec_mse = MSE(ground_truth.cpu().detach().numpy(), dummy_data.cpu().detach().numpy())
            tvloss = TVloss(dummy_data)
            print(iters, "gradloss: %.4f"  % avg_loss, "mseloss: %.5f" % rec_mse, "tvloss-%.5f" % tvloss)
            dummy_data  = torch.clamp(dummy_data * ds + dm, 0, 1) # denormalized
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
        plt.savefig("images/"+str(i)+".jpg")
        plt.close() 





