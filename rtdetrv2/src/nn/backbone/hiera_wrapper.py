import torch 
import torch.nn as nn

from ...core import register
from typing import List, Tuple

@register()

class Hiera(nn.Module):
    
    def __init__(self, input_size: Tuple[int, ...] = (224, 224), num_classes: int = 1000, no_head: bool = False, pretrained: bool = True, model: str = "hiera_tiny_224", checkpoint:str = "mae_in1k_ft_in1k"):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/hiera", model=model, pretrained=pretrained, checkpoint=checkpoint, num_classes=num_classes, input_size=input_size)
        self.model.head.requires_grad_(False)
        self.model.norm.requires_grad_(False)

        if(no_head):
            del self.model.Hiera.head
            del self.model.Hiera.norm



    def forward(self, x: torch.Tensor, return_intermediates: bool = True) -> torch.Tensor:
        
        x = self.model(x, return_intermediates=return_intermediates)

        if(return_intermediates):
            _, intermediates = x
            return [x, intermediates[0].permute(0,3,2,1), intermediates[1].permute(0,3,2,1), intermediates[2].permute(0,3,2,1),  intermediates[3].permute(0,3,2,1)]
        else:
            return x