import os 
import sys 
from torchinfo import summary
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.nn.backbone.presnet import PResNet
from src.nn.backbone.bhresnet import BHResNet
from src.nn.backbone.hiera import Hiera

dummy = torch.rand(1, 3, 224, 224)


bb_old = PResNet(depth=18, 
            num_stages=4, 
            return_idx=[0, 1, 2, 3], 
            act='relu',
            freeze_at=0, 
            freeze_norm=True, 
            pretrained=False)
summary(bb_old)

start1= time.perf_counter()
out1 = bb_old(dummy)
print("Resnet18 took ", time.perf_counter() - start1, "s")
for stage in out1:
   print(stage.shape)


print("NEW:")
"""
bb_new = BHResNet(depth=18, 
            num_stages=4, 
            return_idx=[0, 1, 2, 3], 
            act='relu',
            freeze_at=0, 
            freeze_norm=True, 
            pretrained=False)
summary(bb_new)


out1 = bb_old(dummy)
for stage in out1:
   print(stage.shape)
print("---------------")
out2 = bb_new(dummy)
for stage in out2:
   print(stage.shape)"""

bb_new = Hiera(input_size=(640, 640)).from_pretrained("facebook/hiera_tiny_224.mae_in1k") 
summary(bb_new)

start2= time.perf_counter()
out = bb_new(dummy)
print("Hiera took ", time.perf_counter() - start2, "s")

#print(bb_new)
