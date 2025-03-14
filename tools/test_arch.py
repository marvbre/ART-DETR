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

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2



# Create input transformations
input_size = 640

dummy = torch.rand(1, 3, input_size, input_size )

bb_old = PResNet(depth=18, 
            num_stages=4, 
            return_idx=[0, 1, 2, 3], 
            act='relu',
            freeze_at=-1, 
            freeze_norm=False, 
            pretrained=False)


#summary(bb_old)
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


bb_new = PHiera(no_head=True) #embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), input_size=(1280, 1280), patch_stride=(4,4))  # bhresnet 18 [4, 4, 2, 1] -> (4, 4, 2, 1)
#summary(bb_new)

for name, param in bb_new.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

start2= time.perf_counter()
intermediates = bb_new.forward(dummy, return_intermediates=True)

for stage in intermediates:
   print(stage.shape)


print("New took ", time.perf_counter() - start2, "s")

decoder = RTDETRTransformerv2(feat_channels= [96, 192, 384, 768], #[256, 256, 256, 256]
                              feat_strides= [4, 8, 16, 32], #[4, 8, 16, 32]
                              hidden_dim= 256,
                              num_levels= 4,

                              num_layers= 6,
                              num_queries= 300,

                              num_denoising= 100,
                              label_noise_ratio= 0.5,
                              box_noise_scale= 1.0, # 1.0 0.4

                              eval_idx= -1,

                              # NEW
                              num_points= [4, 4, 4, 4], # [3,3,3] [2,2,2]
                              cross_attn_method= 'default', # default, discrete
                              query_select_method= 'default', # default, agnostic)
)

decoder.forward(intermediates)

"""
start1= time.perf_counter()
out1 = bb_old(dummy)
print("Resnet18 took ", time.perf_counter() - start1, "s")

for stage in out1:
   print(stage.shape)

print("---------------")

out2 = bb_new(dummy)
print("New took ", time.perf_counter() - start2, "s")

for stage in out2:
   print(stage.shape)"""

