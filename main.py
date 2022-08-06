# -*- coding: utf-8 -*-
import argparse
from locale import normalize
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import copy
import os, math
import inversefed
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from utils.training import local_update, test, accuracy
from models.vision import LeNet, LeNet_mnist, AlexNet_Imagenet, AlexNet_Cifar, ResNet18
from utils.args import parser_args
from utils.datasets import get_data
from utils.metrics import label_to_onehot, cross_entropy_for_onehot, TVloss, MSE, reconstruction_costs, calculate_ssim, Classification
from utils.config import Setup_Config
from skimage.metrics import structural_similarity as ssim
import skimage as skimage

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
        device_ids = [ 1, 2,  3]
        print("Running on %s" % device)
    
    # prepare dataset
    if args.dataset == "imagenet":
        num_classes = 10
        input_size = 224
        img_shape = (3, 224, 224)
    if args.dataset == "cifar10":
        num_classes = 10
        input_size = 32
        img_shape = (3, 32, 32)
    if args.dataset == "mnist":
        num_classes = 10
        input_size = 28
        img_shape = (1, 28, 28)

    config = Setup_Config(args)

    lr = config["lr"]
    # if config['normalized'] == True:
    if args.dataset == "imagenet":
        dm = torch.as_tensor([0.4802, 0.4481, 0.3975])[:, None, None].to(device)
        ds = torch.as_tensor([0.2302, 0.2265, 0.2262])[:, None, None].to(device)
    if args.dataset == "cifar10":
        dm = torch.as_tensor([0.507, 0.487, 0.441])[:, None, None].to(device)
        ds = torch.as_tensor([0.267, 0.256, 0.276])[:, None, None].to(device)
    if args.dataset == "mnist":
        dm = torch.as_tensor([0.1307]).to(device)
        ds = torch.as_tensor([0.3081]).to(device)

    dst, test_set = get_data(dataset=args.dataset, data_root = args.data_root, normalized = config['normalized'])
    local_train_ldr = DataLoader(dst, batch_size = 32, shuffle=False, num_workers=2) 
   
    model_time = []
    # prepare model 
    def get_model_fn():
        if args.model == "lenet":
            if args.dataset == "mnist":
                model_fn = LeNet_mnist()
            else:
                model_fn = LeNet(input_size=input_size)
        if args.model == "alexnet":
            if args.dataset == "imagenet":
                model_fn = AlexNet_Imagenet(num_classes=num_classes, input_size = input_size) # pretrained = True
            else: 
                model_fn = AlexNet_Cifar(num_classes=num_classes, input_size = input_size)
        if args.model == "resnet":
            if args.dataset == "imagenet":
                # model_fn = torchvision.models.resnet18(True)
                #net = ResNet18(num_classes=num_classes, imagenet = True).to(device)
                model_fn = torchvision.models.resnet18(num_classes =10, pretrained=False)
            else:
                model_fn = ResNet18(num_classes=num_classes, imagenet = False)
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

    net = net.to(device)

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

    if args.optim in ["geiping", "BN", "GC", "gaussian", "Zhu"]:
        gt_data, gt_label = [], []
        target_id_ = args.index

        while len(gt_label) < args.bs:
            img, label = local_train_ldr.dataset[target_id_]

            target_id_ += 1
            #if label not in gt_label :
            gt_label.append(torch.as_tensor((label,), device= "cuda"))
            gt_data.append(img.to(device))

        gt_data = torch.stack(gt_data)
        gt_label = torch.cat(gt_label)
        
        # gt_data = torch.as_tensor(
        #         np.array(Image.open("auto.jpg").resize((224, 224), Image.BICUBIC)) / 255, device ="cuda", dtype=torch.float
        #     )
        # gt_data = gt_data.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
        history = []

    # Dataloader for RGIA
    else:
        gt_data = dst[img_index][0].to(device) # 0 for label, 1 for label
        gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
        gt_label = gt_label.view(1, )
        data_size = gt_data.size()
        gt_data = gt_data.view(1, *data_size)
        
        # batch selection to be attacked
        for i in range(args.bs-1):
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

        # store original gradient
        original_dy_dx = []
        for i in range(len(model_time)):
            net.load_state_dict(model_time[i])
            net.eval()
            pred = net(gt_data)
            y = criterion(pred, gt_onehot_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx.append(list((_.detach().clone().cpu() for _ in dy_dx)))

    if args.optim in ["geiping", "BN", "GC", "gaussian", "Zhu"]: #bassline 
        net.eval()
        net.zero_grad()
        loss_fn = Classification() 

        bn_l, bn_prior = [], []
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_l.append(inversefed.gradientinversion.BNStatisticsHook(module))

        target_loss, _, _ = loss_fn(net(gt_data), gt_label)
        input_gradient = torch.autograd.grad(target_loss, net.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
        print(f"Full gradient norm is {full_norm:e}.")

        for i, mod in enumerate(bn_l):
            mean_var = mod.mean_var[0].detach(), mod.mean_var[1].detach()
            bn_prior.append(mean_var)

        rec_machine = inversefed.GradientReconstructor(net, (dm, ds), config, num_images=args.bs, bn_prior=bn_prior)
        # need some reshaping 
        
        history, stats = rec_machine.reconstruct(input_gradient, gt_label, img_shape= img_shape, dryrun=False)
        
   
        if config['normalized'] == True:
            history = torch.clamp(history * ds + dm, 0, 1) # denormalized
            gt_data  = torch.clamp(gt_data * ds + dm, 0, 1)

        for i in range(args.bs):
            plt.figure()
            plt.imshow(tt(history[i]))
            plt.savefig("images/"+ args.model + '_' + args.dataset + '/' + "optim_" +args.optim + "_mode_" +args.mode + "_iter_" + str(args.attack_iters) + "_" +str(i)+"_whole.png")
            plt.close()

        test_mse, test_psnr, test_ssim = [], [], []

        test_mse = MSE(history.cpu().numpy(), gt_data.cpu().numpy())

        for i in range(args.bs):
            #mse = (history[i].cpu() - gt_data[i].cpu()).pow(2).mean().item()
            test_psnr.append(10 * math.log(1/test_mse[i], 10))
            # calculate ssim
            a = []
            for j in range(args.bs):
                if args.dataset != "mnist":
                    a.append(ssim(history[i].cpu().numpy(), gt_data[j].cpu().numpy(), win_size=3))
                else:  
                    a.append(ssim(history[i].squeeze(0).cpu().numpy(), gt_data[j].squeeze(0).cpu().numpy()))
            if args.dataset != "mnist":
                test_ssim.append(ssim(history[i].cpu().numpy(), gt_data[np.argsort(a)[0]].cpu().numpy(), win_size=3))
            else:  
                test_ssim.append(ssim(history[i].squeeze(0).cpu().numpy(), gt_data[np.argsort(a)[0]].squeeze(0).cpu().numpy()))

            # if args.dataset != "mnist":
            #     test_ssim.append(ssim(history[i].cpu().numpy(), gt_data[i].cpu().numpy(), win_size=3))
            # else:  
            #     test_ssim.append(ssim(history[i].squeeze(0).cpu().numpy(), gt_data[i].squeeze(0).cpu().numpy()))

    # Proposed RGIA

    else:
        for iters in range(config['epochs']): # default =300
        
            x_collection = []
            #y_collection = []
            loss = []

            if iters == int(config['epochs'] * 3/8) or iters == int(config['epochs'] * 5/8) or iters == int(config['epochs'] * 7/8):
                lr = lr * config['lr_decay']

            for i in range(len(model_time)):
                
                # load model
                net.load_state_dict(model_time[i])
                # load data
                dummy_data = x_avg.requires_grad_(True)
                #dummy_label = y_avg.requires_grad_(True)

                # set optimizer for deep leakage
                # optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = args.lr)
                optimizer = torch.optim.LBFGS([dummy_data], lr = lr,  max_iter=20)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[config['epochs'] // 2.667, config['epochs'] // 1.6,
                                                                    config['epochs'] // 1.142],
                                                        gamma=config['lr_decay'])  # 3/8 5/8 7/8

                # if i==0 and iters ==0:
                #     for j in range(len(original_dy_dx)):
                #         plot_his(original_dy_dx[j])
                #         plt.savefig('hists/' + args.model + '_' + args.dataset + '/' + str(iters) + "_"+str(j) + '.png')
                #         plt.close()

                target_dy_dx = []
                for grad in original_dy_dx[i]:        
                    target_dy_dx.append(grad.cuda())

                def closure():
                    optimizer.zero_grad()
                    dummy_pred = net(dummy_data)
                    dummy_loss = criterion(dummy_pred, gt_onehot_label) 
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    s = 1
                    grad_diff = 0
                    if config['cost_fn'] == "l2":
                        for gx, gy in zip(dummy_dy_dx[0:], target_dy_dx[0:]):
                            grad_diff += ((gx - gy) ** 2).sum() * s
                            s += 2

                    if config['cost_fn'] == "sim":
                        grad_diff = reconstruction_costs(dummy_dy_dx, target_dy_dx, cost_fn="sim", indices="def", weights="equal")

                    # grad_diff = grad_diff * 1e-5 
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

            if config['normalized'] == True:
                x_avg.data = torch.max(torch.min(x_avg, (1 - dm) / ds), -dm / ds)

            blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=0.5)
            x_avg = blurrer(x_avg)
            #y_avg = avg(y_collection)   
        
            if (iters+1) % config['interval'] == 0:  #default = 10
                rec_mse = MSE(gt_data.cpu().detach().numpy(), dummy_data.cpu().detach().numpy())
                tvloss = TVloss(dummy_data)
                psnr = 10*math.log(0.1)
                print(iters, "gradloss: %.4f"  % avg_loss, "mseloss: %.5f" % 1, "tvloss: %.5f" % tvloss, "PSNR: %.2f" % psnr)
                if config['normalized'] == True:
                    denormal_dummy_data  = torch.clamp(dummy_data * ds + dm, 0, 1) # denormalized
                    history.append(denormal_dummy_data.detach().cpu())
                else:
                    history.append(dummy_data.detach().cpu())

        for i in range(args.bs):
            plt.figure()
            plt.imshow(tt(history[-1][i]))
            plt.savefig("images/"+ args.model + '_' + args.dataset + '/' + "optim_" +args.optim + "_mode_" +args.mode + "_iter_" + str(args.attack_iters) + "_" +str(i)+"_whole.png")
            plt.close()
        # plt.figure(figsize=(12, np.sqrt(args.bs) * 8))
        # for j in range(args.bs):
        #     for i in range(int(config['epochs']/config['interval'])):
        #         plt.subplot(int(config['epochs']/config['interval'] * args.bs  / 10), 10,
        #                     int(config['epochs']/config['interval']) * j + i + 1)
        #         plt.imshow(tt(history[i][j]))
        #         plt.axis('off')
        # plt.title("rec_MSE-%.4f" % (rec_mse))
        # plt.savefig('images/' + args.model + '_' + args.dataset + '/' + args.avg_type + '_bs_' + str(
        #     args.bs) + '_iter' + str(
        #     args.attack_iters) + 'start_' + str(args.start_attack_iters) + "_" +args.optim + "_mode_" +args.mode + "_" +args.cost_fn + '.png')
        # plt.close()

        if config['normalized'] == True:
            gt_data  = torch.clamp(gt_data * ds + dm, 0, 1)

        test_mse, test_psnr, test_ssim, imgs = [], [], [], []

        for i in range(args.bs):
            imgs.append(copy.deepcopy(history[-1][i].cpu()))
        imgs = torch.stack(imgs)
       
        gt_data = gt_data.cpu()

        test_mse = MSE(imgs.numpy(), gt_data.numpy())

        for i in range(args.bs):
            #mse = (history[i].cpu() - gt_data[i].cpu()).pow(2).mean().item()
            test_psnr.append(10 * math.log(1/test_mse[i], 10))
          
            # calculate ssim
            a = []
            for j in range(args.bs):
                if args.dataset != "mnist":
                    a.append(ssim(imgs[i].cpu().numpy(), gt_data[j].cpu().numpy(), win_size=3))
                else:  
                    a.append(ssim(imgs[i].squeeze(0).cpu().numpy(), gt_data[j].squeeze(0).cpu().numpy()))

            if args.dataset != "mnist":
                test_ssim.append(ssim(imgs[i].cpu().numpy(), gt_data[np.argsort(a)[0]].cpu().numpy(), win_size=3))
            else:  
                test_ssim.append(ssim(imgs[i].squeeze(0).cpu().numpy(), gt_data[np.argsort(a)[0]].squeeze(0).cpu().numpy()))
        
        # for i in range(args.bs):
        #     mse = (history[-1][i].cpu() - gt_data[i].cpu()).pow(2).mean().item()
        #     test_mse.append(mse)
        #     test_psnr.append(10 * math.log(1/mse, 10))
        #     # test_ssim = calculate_ssim(history[i], gt_data[i])
        #     if args.dataset != "mnist":
        #         test_ssim.append(ssim(history[-1][i].cpu().numpy(), gt_data[i].cpu().numpy(), win_size=3))
        #     else:  
        #         test_ssim.append(ssim(history[-1][i].squeeze(0).cpu().numpy(), gt_data[i].squeeze(0).cpu().numpy()))

    # save to table 

    mses, psnrs, ssims = np.nan_to_num(np.array(test_mse)), np.nan_to_num(np.array(test_psnr)), np.nan_to_num(np.array(test_ssim))
    mse_mean, psnr_mean, ssim_mean = mses.mean(), psnrs.mean(), ssims.mean()
    mse_std, psnr_std, ssim_std = np.std(mses), np.std(psnrs), np.std(ssims)
    mse_max, psnr_max, ssim_max = mses.max(), psnrs.max(), ssims.max()
    mse_min, psnr_min, ssim_min = mses.min(), psnrs.min(), ssims.min()
    mse_median, psnr_median, ssim_median = np.median(mses), np.median(psnrs), np.median(ssims)

    inversefed.utils.save_to_table(
        'tables/', 
        name=f"exp_{args.model}",
        model=args.model,
        dataset=args.dataset,
        trained=args.mode,
        restarts=args.attack_iters, # number of epochs
        bs = args.bs, 
        target_id = args.index,
        optim=args.optim,
        cost_fn=args.cost_fn,
        #weights=args.weights,
        tv=args.tv,
        psnr_mean=psnr_mean,
        mse_mean=mse_mean,
        ssim_mean=ssim_mean,
        psnr_std=psnr_std,
        mse_std=mse_std,
        ssim_std=ssim_std,
    )










