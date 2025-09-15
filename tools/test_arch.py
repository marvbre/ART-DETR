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
from src.nn.backbone.vit import ViT_Backbone


# Create input transformations
input_size = 1280


dummy = torch.rand(1, 3, input_size, input_size )

m = ViT_Backbone()

start2= time.perf_counter()
out2 = m(dummy)
print("ViT took ", time.perf_counter() - start2, "s and puts out:")

for stage in out2:
   print(stage.shape)

lol = HybridEncoder(in_channels=[384, 384, 384],
                     feat_strides=[8, 16, 32],
                     hidden_dim=256,
                     nhead=8,
                     dim_feedforward = 1024,
                     dropout=0.0,
                     enc_act='gelu',
                     use_encoder_idx=[2],
                     num_encoder_layers=1,
                     pe_temperature=10000,
                     expansion=1.0,
                     depth_mult=1.0,
                     act='silu',
                     eval_spatial_size=None,
                     version='v2')
summary(lol)
lol.forward(out2)

#problem is: CSP Backbone assumes every level is half resolution of previous level and half the channels...