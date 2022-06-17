from email.mime import image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from models.layers.conv2d import ConvBlock



def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class LeNet(nn.Module):
    def __init__(self, input_size = 32):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(input_size*input_size*3/4), 10) # 768 for cifar
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)

        # print(out.shape)
        # exit()
        out = self.fc(out)
        return out



# class AlexNet(nn.Module):
#     def __init__(self,num_classes=3, input_size = 32):
#         super(AlexNet, self).__init__()
#         self.features=nn.Sequential(
#             nn.Conv2d(3,48, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3,stride=2),
#             nn.Conv2d(48,128, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3,stride=2),
#             nn.Conv2d(128,192,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192,128,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3,stride=2),
#         )

#         self.classifier=nn.Sequential(
            
#             nn.Linear(128 * 6 * 6, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(2048, num_classes),
#         )

#     def forward(self,x):
#         #print(x.shape)
#         x=self.features(x)
#         x=torch.flatten(x,start_dim=1)
       
#         x=self.classifier(x)
       
#         return x



class AlexNet(nn.Module):

    def __init__(self, num_classes, input_size = 32):
        super().__init__()
        maxpoolidx = [1, 3, 7]
        layers = []
        inp = 3
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }
        for layeridx in range(8):
            if layeridx in maxpoolidx:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layeridx][0]
                p = kp[layeridx][1]
                layers.append(ConvBlock(inp, oups[layeridx], k, 1, p))
                inp = oups[layeridx]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(4 * input_size * input_size, num_classes)

    def forward(self, x):
        for m in self.features:
            x = m(x)
            # if isinstance(m, PassportPrivateBlock):
            #     x = m(x)
            # else:
            #     x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def get_convblock():
    #print(passport_kwargs)

    def convblock_(*args, **kwargs):
        return ConvBlock(*args, **kwargs)

    return convblock_


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):#(512, 512, 2) (512, 512, 1)
        super(BasicBlock, self).__init__()

        self.convbnrelu_1 = get_convblock()(in_planes, planes, 3, stride, 1)
        self.convbn_2 = get_convblock()(planes, planes, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = get_convblock()(in_planes, self.expansion * planes, 1, stride, 0) # input, output, kernel_size=1

    def forward(self, x):
        
        out = self.convbnrelu_1(x)
        out = self.convbn_2(out)

        if not isinstance(self.shortcut, nn.Sequential):
            
            out = out + self.shortcut(x)

        else: # if self.shortcut == nn.Sequential 
            out = out + x
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100): #BasicPrivateBlock, [2, 2, 2, 2], **model_kwargs
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks

        self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride): #BasicPrivateBlock, planes = 512, numblocks = 2, stride =2, **model_kwargs
        strides = [stride] + [1] * (num_blocks - 1) # [2] + [1]*1 = [2, 1]
        layers = []
        for i, stride in enumerate(strides): #stride = 2 & 1
            layers.append(block(self.in_planes, planes, stride)) # (512, 512, 2)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
       
        out = self.convbnrelu_1(x)

        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)
        for block in self.layer4:
            out = block(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def ResNet18(**model_kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])



