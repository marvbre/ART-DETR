import torch 
import torch.nn as nn

from ...core import register
from typing import List, Tuple

from src.nn.backbone.hiera import Unroll, Reroll
import math
import torch.nn.functional as F

@register()
class PHiera(nn.Module):
    
    def __init__(self, input_size: Tuple[int, ...] = (224, 224), num_classes: int = 1000, no_head: bool = False, pretrained: bool = True, model: str = "hiera_tiny_224", checkpoint:str = "mae_in1k_ft_in1k"):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/hiera", model=model, pretrained=pretrained, checkpoint=checkpoint, num_classes=num_classes, input_size=input_size)
        print("Pretrained weights from Meta loaded")
        self.model.head.requires_grad_(False)
        #self.model.norm.requires_grad_(False)

        if(no_head):
            self.model.head = nn.Identity()
            #self.model.norm = nn.Identity()

        #self.update_input_size(640)

    def forward(self, x: torch.Tensor, return_intermediates: bool = True) -> torch.Tensor:
        
        x = self.model(x, return_intermediates=return_intermediates)

        if(return_intermediates):
            _, intermediates = x #4 stages
            # Maybe apply another normalization here ?? 
            return [intermediates[0].permute(0,3,2,1), intermediates[1].permute(0,3,2,1), intermediates[2].permute(0,3,2,1),  intermediates[3].permute(0,3,2,1)]
        else:
            return x
        
    def update_input_size(self, new_size):
      patch_stride = self.model.patch_stride
      self.model.input_size = (new_size, new_size)
      self.model.tokens_spatial_shape = [i // s for i, s in zip(self.model.input_size, patch_stride)]
      self.model.num_tokens = math.prod(self.model.tokens_spatial_shape)
    
      # Reset roll and reroll modules
      self.model.unroll = Unroll((new_size, new_size), patch_stride, [self.model.q_stride] * len(self.model.stage_ends[:-1]))
      self.model.reroll = Reroll((new_size, new_size), patch_stride, [self.model.q_stride] * len(self.model.stage_ends[:-1]), self.model.stage_ends, self.model.q_pool)
      
      #self.model.pos_embed = nn.Parameter(torch.zeros(1, self.model.num_tokens, 96)) dont set pos embed to zero! interpolate.
      old_pos_embed = self.model.pos_embed  # Save old embeddings
      old_size = int(math.sqrt(old_pos_embed.shape[1]))  # Assuming square shape

      new_pos_embed = F.interpolate(
         old_pos_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2), 
         size=(self.model.tokens_spatial_shape[0], self.model.tokens_spatial_shape[1]), 
         mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(1, self.model.num_tokens, -1)

      self.model.pos_embed = nn.Parameter(new_pos_embed)
      print("Before:", old_pos_embed.shape, " afteR:", self.model.pos_embed.shape)




