from locale import normalize
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os 
import torch
from torchvision import transforms, datasets
from torchvision import utils as vutils
import torchvision

dm = torch.as_tensor([0.4802, 0.4481, 0.3975])[:, None, None]
ds = torch.as_tensor([0.2302, 0.2265, 0.2262])[:, None, None]

dir = "/home/jjlin/datasets/ILSVRC2012/train/" + "n02391049"+ "/"

datanames = os.listdir(dir)
for i in range(20):
    if not os.path.exists("/home/lbw/Data/imagenet20/"+ str(i)):
        os.makedirs("/home/lbw/Data/imagenet20/"+ str(i)) 
gt_data = []
target_id_ = 0
while len(gt_data) < 500:
    img = torch.as_tensor(
        np.array(Image.open(dir + datanames[target_id_]).resize((224, 224), Image.BICUBIC)) / 255, dtype=torch.float
    )
    if len(img.shape) == 3:
        img = img.permute(2, 0, 1).contiguous()
        gt_data.append(img)
    target_id_ += 1

for i in range(500):
    vutils.save_image(gt_data[i], "/home/lbw/Data/imagenet20/"+ str(7) + "/" + str(i) +".jpeg")

# 0 n01443537 goldfish
# 1 n01514668 cock
# 2 n01914609 sea anemone
# 3 n02099601 golden dog
# 4 n02165456 ladybug
# 5 n02259212 leafhopper
# 6 n02281406 butterfly
# 7 n02391049 zimbra
# 8 n02510455 panda
# 9 n02443114 fox
# 10 n02892767 bra
# 11 n02992211 cello
# 12 n03100240 car
# 13 n03063599 coffee mug
# 14 n03249569 drum
# 15 n03485794 handkerchief
# 16 n03676483 lipstick
# 17 n03916031 perfume
# 18 n03982430 pool table
# 19 n11939491 daisy