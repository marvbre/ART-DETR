import torch 
import torch.nn as nn
from ...core import register
from typing import List, Tuple
import math
import torch.nn.functional as F

from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights

regnet = regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V2)

@register()
class RegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = regnet.stem
        self.stage1 = regnet.trunk_output.block1
        self.stage2 = regnet.trunk_output.block2
        self.stage3 = regnet.trunk_output.block3
        self.stage4 = regnet.trunk_output.block4

    def forward(self, x):
        x = self.stem(x)
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        return [feat2, feat3, feat4]

"""
@register()
class RegNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V2)
        self.stem = self.backbone.stem
        self.stages = create_feature_extractor(self.backbone.trunk_output, return_nodes={'block1': 'feat1','block2': 'feat2', 'block3': 'feat3',  'block4': 'feat4'})

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x.values()"""
        

  