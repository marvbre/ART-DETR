import os 
import sys 
from torchinfo import summary
import torch
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.nn.backbone.presnet import PResNet
from src.nn.backbone.bhresnet import BHResNet
from src.nn.backbone.hiera import Hiera, HybridEncoderReplacement #, imgnet_dict
from src.nn.backbone.hiera_wrapper import PHiera
from src.nn.backbone.regnet import RegNet

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2

from src.nn.backbone import CSPDarkNet, CSPPAN, CSPDarkNetPAN



# Create input transformations
input_size = 1536

dummy = torch.rand(1, 3, input_size, input_size )

m = CSPDarkNet(3, width_multi=1.0, depth_multi=1.25,  return_idx = [1,2,3,4,-1])
#m = CSPDarkNetPAN(3, width_multi=width_multi, depth_multi=depth_multi,  return_idx = [3,4,5,6,7])
#[64, 128, 256, 512, 1024, 1024, 1024, 1024] 
#summary(m)


start2= time.perf_counter()
out2 = m(dummy)
print("CSP-Darknet-P7 took ", time.perf_counter() - start2, "s and puts out:")

for stage in out2:
   print(stage.shape)
n = CSPPAN(in_channels=[256, 512, 1024, 1024, 1024], depth_multi=1.25, act='silu')

start3= time.perf_counter()
out3 = n(out2)
print("PAN took ", time.perf_counter() - start2, "s and puts out:")
for stage in out3:
   print(stage.shape)

#problem is: CSP Backbone assumes every level is half resolution of previous level and half the channels...