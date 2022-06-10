import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

from utils.sampling import label_to_onehot, cross_entropy_for_onehot



def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def fed_avg(self, local_ws, client_weights, lr_outer):

    w_avg = copy.deepcopy(local_ws[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * client_weights[0]

        for i in range(1, len(local_ws)):
            w_avg[k] += local_ws[i][k] * client_weights[i]

        self.w_t[k] = w_avg[k]

def local_update(train_ldr, model, lr):
    
    if torch.cuda.is_available():
        device = "cuda"
    optimizer = torch.optim.SGD(model.parameters(), lr, 
                            momentum=0.9,
                            weight_decay=0.0005) 
                                
    model.to(device)
    model.train()
    loss_meter = 0

    for batchid, (x, y) in enumerate(train_ldr):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()                
        loss = torch.tensor(0.).to(device)

        pred = model(x)
        loss += F.cross_entropy(pred, y)
        
        loss.backward()

        
        optimizer.step()                
        loss_meter += loss.item()         
    loss_meter = loss_meter/len(train_ldr)     
    # if dp:
    #     for param in self.model.parameters():
    #         param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)
    return model.state_dict(), loss_meter, 

def test(self, dataloader):

    self.model.to(self.device)
    self.model.eval()

    loss_meter = 0
    acc_meter = 0
    runcount = 0

    with torch.no_grad():
        for load in dataloader:
            data, target = load[:2]
            data = data.to(self.device)
            target = target.to(self.device)
    
            pred = self.model(data)  # test = 4
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
            pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
            runcount += data.size(0) 

    loss_meter /= runcount
    acc_meter /= runcount

    return  loss_meter, acc_meter 