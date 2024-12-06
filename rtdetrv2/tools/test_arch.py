import os 
import sys 
from torchinfo import summary
import torch
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.nn.backbone.presnet import PResNet
from src.nn.backbone.bhresnet import BHResNet
from src.nn.backbone.hiera import imgnet_dict,  Hiera
from src.nn.backbone.hiera_wrapper import Hiera

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



# Create input transformations
input_size = 640

dummy = torch.rand(1, 3, input_size, input_size )

bb_old = PResNet(depth=18, 
            num_stages=4, 
            return_idx=[0, 1, 2, 3], 
            act='relu',
            freeze_at=0, 
            freeze_norm=True, 
            pretrained=False)
#summary(bb_old)

"""
#print("NEW:")

bb_new = BHResNet(depth=18, 
            num_stages=4, 
            return_idx=[0, 1, 2, 3], 
            act='relu',
            freeze_at=0, 
            freeze_norm=True, 
            pretrained=False)
#summary(bb_new)"""


bb_new = Hiera()

start1= time.perf_counter()
out1 = bb_old(dummy)
print("Resnet18 took ", time.perf_counter() - start1, "s")

for stage in out1:
   print(stage.shape)

print("---------------")
start2= time.perf_counter()

out2 = bb_new(dummy)
print("BHResnet18 took ", time.perf_counter() - start1, "s")

for stage in out2:
   print(stage.shape)

