from email.mime import image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from models.layers.conv2d import ConvBlock
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional, Dict, Mapping, cast


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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlock(nn.Module):
#     expansion: int = 1

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

#     expansion: int = 4

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.0)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):
#     def __init__(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         layers: List[int],
#         num_classes: int = 1000,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         #_log_api_usage_once(self)
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
#             )
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

#     def _make_layer(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         dilate: bool = False,
#     ) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                 )
#             )

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x: Tensor) -> Tensor:
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)


# @dataclass
# class Weights:
#     """
#     This class is used to group important attributes associated with the pre-trained weights.
#     Args:
#         url (str): The location where we find the weights.
#         transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
#             needed to use the model. The reason we attach a constructor method rather than an already constructed
#             object is because the specific object might have memory and thus we want to delay initialization until
#             needed.
#         meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
#             informative attributes (for example the number of parameters/flops, recipe link/methods used in training
#             etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
#             meta-data (for example the `classes` of a classification model) needed to use the model.
#     """

#     url: str
#     transforms: Callable
#     meta: Dict[str, Any]


# class WeightsEnum(StrEnum):
#     """
#     This class is the parent class of all model weights. Each model building method receives an optional `weights`
#     parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
#     `Weights`.
#     Args:
#         value (Weights): The data class entry with the weight information.
#     """

#     def __init__(self, value: Weights):
#         self._value_ = value

#     @classmethod
#     def verify(cls, obj: Any) -> Any:
#         if obj is not None:
#             if type(obj) is str:
#                 obj = cls.from_str(obj.replace(cls.__name__ + ".", ""))
#             elif not isinstance(obj, cls):
#                 raise TypeError(
#                     f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
#                 )
#         return obj

#     def get_state_dict(self, progress: bool) -> Mapping[str, Any]:
#         return load_state_dict_from_url(self.url, progress=progress)

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}.{self._name_}"

#     def __getattr__(self, name):
#         # Be able to fetch Weights attributes directly
#         for f in fields(Weights):
#             if f.name == name:
#                 return object.__getattribute__(self.value, name)
#         return super().__getattr__(name)


# def _resnet(
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

#     model = ResNet(block, layers, **kwargs)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress))

#     return model


# _COMMON_META = {
#     "min_size": (1, 1),
#     "categories": _IMAGENET_CATEGORIES,
# }


# class ResNet18_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 11689512,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 69.758,
#                     "acc@5": 89.078,
#                 }
#             },
#             "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
#         },
#     )
#     DEFAULT = IMAGENET1K_V1

# def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#     Args:
#         weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet18_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#     .. autoclass:: torchvision.models.ResNet18_Weights
#         :members:
#     """
#     weights = ResNet18_Weights.verify(weights)

#     return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

















# def get_convblock():
#     #print(passport_kwargs)

#     def convblock_(*args, **kwargs):
#         return ConvBlock(*args, **kwargs)

#     return convblock_


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):#(512, 512, 2) (512, 512, 1)
#         super(BasicBlock, self).__init__()

#         self.convbnrelu_1 = get_convblock()(in_planes, planes, 3, stride, 1)
#         self.convbn_2 = get_convblock()(planes, planes, 3, 1, 1)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = get_convblock()(in_planes, self.expansion * planes, 1, stride, 0) # input, output, kernel_size=1

#     def forward(self, x):
        
#         out = self.convbnrelu_1(x)
#         out = self.convbn_2(out)

#         if not isinstance(self.shortcut, nn.Sequential):
            
#             out = out + self.shortcut(x)

#         else: # if self.shortcut == nn.Sequential 
#             out = out + x
#         out = F.relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=100, imagenet = True): #BasicPrivateBlock, [2, 2, 2, 2], **model_kwargs
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         self.num_blocks = num_blocks

#         self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         if imagenet == True:
#             self.linear = nn.Linear(25088, num_classes)
#         else:
#             self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride): #BasicPrivateBlock, planes = 512, numblocks = 2, stride =2, **model_kwargs
#         strides = [stride] + [1] * (num_blocks - 1) # [2] + [1]*1 = [2, 1]
#         layers = []
#         for i, stride in enumerate(strides): #stride = 2 & 1
#             layers.append(block(self.in_planes, planes, stride)) # (512, 512, 2)
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
       
#         out = self.convbnrelu_1(x)

#         for block in self.layer1:
#             out = block(out)
#         for block in self.layer2:
#             out = block(out)
#         for block in self.layer3:
#             out = block(out)
#         for block in self.layer4:
#             out = block(out)

#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         # print(out.shape)
#         # exit()
#         out = self.linear(out)

#         return out

# def ResNet18(**model_kwargs):
#     return ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)




