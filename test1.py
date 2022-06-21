# -*- coding: utf-8 -*-
import argparse
import numpy as np
from PIL import Image
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
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from utils.args import parser_args
from models.vision import LeNet, AlexNet_Imagenet, AlexNet_Cifar, ResNet18, weights_init

print(torch.__version__, torchvision.__version__)

from utils.datasets import get_data
from utils.sampling import label_to_onehot, cross_entropy_for_onehot


def plot_his(A):
    plt.hist(torch.flatten(A).cpu().detach().numpy(), bins='auto', density=True)


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


def avg(x_collection):
    x = x_collection[0].unsqueeze(0)
    for x_sample in x_collection[1:]:
        x = torch.cat([x, x_sample.unsqueeze(0)], dim=0)

    x_avg = torch.mean(x, dim=0, keepdim=False)

    return x_avg


# def trimmed(grad_li, num_machines, num_byz, device, grad_ag_list=None, add_ag=False, t=0):
#     param_list_ = []
#     for each_param_list in grad_li:
#         each_param_array = each_param_list.squeeze()
#         param_list_.append(each_param_array)
#
#     grad_array = torch.cat([x.reshape((-1, 1)) for x in param_list_], dim=1)
#     med, _ = torch.sort(grad_array, axis=1)
#     # trimmed_num = int(num_machines * 0.4)
#     trimmed_num = num_byz
#     med = torch.mean(med[:, trimmed_num:-trimmed_num], axis=1)
#
#     return med

def trimmed_avg(x_collection):
    X = torch.cat([x.reshape((-1, 1)) for x in x_collection[0:]], dim=1)
    med, _ = torch.sort(X, axis=1)
    trimmed_num = args.trimmed_num

    med = torch.mean(med[:, trimmed_num:-trimmed_num], axis=1).reshape(
        (1, x_collection[0].shape[1], x_collection[0].shape[2], x_collection[0].shape[3]))

    return med


if __name__ == '__main__':

    args = parser_args()
    print(args)
    device = "cpu"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = "cuda"
        print("Running on %s" % device)
    if not os.path.exists('imgs/' + args.model + '-' + args.dataset):
        os.makedirs('imgs/' + args.model+ '-' + args.dataset)
    if not os.path.exists('save_model/' + args.model + '-' + args.dataset):
        os.makedirs('save_model/' + args.model + '-' + args.dataset)

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
                                         data_root="/home/lbw/Data",
                                         iid=True,
                                         num_users=10)
    local_train_ldr = DataLoader(dst, batch_size=64, shuffle=False, num_workers=2)
    test_ldr = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    # prepare model
    if args.model == "lenet":
        net = LeNet(input_size=input_size).to(device)
    if args.model == "alexnet":
        if args.dataset == "imagenet":
            net = AlexNet_Imagenet(num_classes=num_classes, input_size=input_size).to(device)
        else:
            net = AlexNet_Cifar(num_classes=num_classes, input_size=input_size).to(device)
    if args.model == "resnet":
        net = ResNet18(num_classes=num_classes, input_size=input_size).to(device)

    img_index = args.index

    model_time = []
    # save model in different epoch
    # for i in range(args.attack_iters + args.start_attack_iters):
    #     local_update(local_train_ldr, net, args.lr)
    #     loss, acc = test(net, local_train_ldr)
    #     print(i)
    #     print("training loss: %.4f" % loss)
    #     print("training acc: %.3f" % acc)
    #     torch.save(net.state_dict(), 'save_model/' + args.model_name + '-' + args.dataset + '/' + str(i))
    #
    # print('load_model')

    ###load model

    learning_optimizer = torch.optim.Adam(net.parameters(),
            0.0001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False)

    # for i in range(10):
    #     local_update(local_train_ldr, net, learning_optimizer)
    #     if i % 1 == 0:
    #         model_time.append(net.state_dict())
    
    dir = "/home/lbw/Code/model_time/"+args.model+"/"+args.dataset+"/"

    for i in range(1, 20, 1):
        model_dict = torch.load(dir+ "model_"+str(i)+".pth")['model']
        model_time.append(model_dict)

    # load image and label
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()
    criterion = cross_entropy_for_onehot
    gt_data = dst[img_index][0].to(device)  # 0 for label, 1 for label
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    data_size = gt_data.size()
    gt_data = gt_data.view(1, *data_size)

    # batch_selection
    if args.batch_size != 1:
        for i in range(args.batch_size - 1):
            gt_data = torch.cat([gt_data, dst[img_index + i + 1][0].view(1, *data_size).to(device)], dim=0)
            gt_label = torch.cat([gt_label, torch.Tensor([dst[img_index + i][1]]).long().view(1, ).to(device)], dim=0)

    gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

    # generate dummy data and label
    dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.rand(gt_onehot_label.size()).to(device).requires_grad_(True)

    # set optimizer for deep leakage
    # optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = 1)

    history = []
    x_avg = dummy_data.clone().detach()
    y_avg = dummy_label.clone().detach()
    original_dy_dx = []

    for i in range(len(model_time)):
        net.load_state_dict(model_time[i])
        pred = net(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx.append(list((_.detach().clone() for _ in dy_dx)))

        
    lr = 1
    tv_para = 0.01
    for i in range(len(original_dy_dx[0])):
        plot_his(original_dy_dx[0][i])
        plt.savefig('imgs/' + args.model + '-' + args.dataset + '/' + str(i) + '.png')
        plt.close()
    print('ok')
    for iters in range(args.epochs):

        x_collection = []
        y_collection = []
        # gradient_x = []
        # gradient_y = []
        loss = []

        for i in range(len(model_time)):

            # load model
            net.load_state_dict(model_time[i])

            # load data
            dummy_data = x_avg.requires_grad_(True)
            dummy_label = y_avg.requires_grad_(True)
            # if iters % 500 == 0:
            #     lr = lr * 1
            #     tv_para = tv_para * 1
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)


            # pred = net(gt_data)
            # y = criterion(pred, gt_onehot_label)
            # dy_dx = torch.autograd.grad(y, net.parameters())
            # original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            def closure():

                optimizer.zero_grad()
                dummy_pred = net(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx[i]):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                tvloss = tv_para * TVloss(dummy_data)
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

        x_avg = avg(x_collection)
        y_avg = avg(y_collection)

        if iters % 10 == 0:
            # rec_mse = MSE(gt_data.cpu().detach().numpy(), dummy_data.cpu().detach().numpy())
            rec_mse = MSE(gt_data.cpu().detach().numpy(), dummy_data.cpu().detach().numpy())
            tvloss = tv_para * TVloss(dummy_data)
            print(iters, "gradloss-%.5f" % avg_loss, "mseloss-%.5f" % rec_mse, "tvloss-%.5f" % tvloss)
            for i in range(args.batch_size):
                history.append(tt(x_avg[i].cpu()))

    plt.figure(figsize=(12, np.sqrt(args.bs) * 8))
    for j in range(args.bs):
        for i in range(int(args.epochs / args.save_iterval)):
            plt.subplot(int(args.epochs * args.bs / args.save_iterval / 10), 10,
                        int(args.epochs / args.save_iterval) * j + i + 1)
            plt.imshow(history[args.bs * i + j])
            plt.axis('off')
    plt.title("rec_MSE-%.4f" % (rec_mse))
    plt.savefig('imgs/' + args.model + '-' + args.dataset + '/' + 'bs_' + str(args.bs) + 'attack_iter' + str(
        args.attack_iters) + 'start_' + str(args.start_attack_iters) + '.png')
    # plt.close()
