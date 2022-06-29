from email.mime import image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from models.layers.conv2d import ConvBlock

from torchvision.models import alexnet


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
        out = self.fc(out)
        return out

class LeNet_Imagenet(nn.Module):
    def __init__(self, input_size = 224):
        super(LeNet_Imagenet, self).__init__()
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
            nn.Linear(int(input_size*input_size*3/4), 20) # 768 for cifar
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        return out




class AlexNet_Imagenet(nn.Module):
    def __init__(self,num_classes=3, input_size = 32):
        super(AlexNet_Imagenet, self).__init__()

        params = []
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier=nn.Sequential(

            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                params.append(layer.weight)
                params.append(layer.bias)

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                params.append(layer.weight)
                params.append(layer.bias)
        
        self._load_pretrained_from_torch(params)

    def _load_pretrained_from_torch(self, params):
        # load a pretrained alexnet from torchvision
        torchmodel = alexnet(True)
        torchparams = []
        for layer in torchmodel.features:
            if isinstance(layer, nn.Conv2d):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for layer in torchmodel.classifier:
            if isinstance(layer, nn.Linear):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for torchparam, param in zip(torchparams, params):
            # print(torchparam.size())
            # print(param.size())
            if torchparam.size() == param.size():
                param.data.copy_(torchparam.data)
            # assert torchparam.size() == param.size(), 'size not match'
            # param.data.copy_(torchparam.data)

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet_Cifar(nn.Module):

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


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        return out


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
    def __init__(self, block, num_blocks, num_classes=100, imagenet = True): #BasicPrivateBlock, [2, 2, 2, 2], **model_kwargs
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks

        self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if imagenet == True:
            self.linear = nn.Linear(25088, num_classes)
        else:
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
        # print(out.shape)
        # exit()
        out = self.linear(out)

        return out

def ResNet18(**model_kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)




