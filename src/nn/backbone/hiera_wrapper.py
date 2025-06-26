import torch 
import torch.nn as nn
from ...core import register

#From https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py#L152
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

def build_hiera_backbone(config_file, ckpt_path=None, device="cuda", mode="train"):
    with initialize_config_dir(config_dir="/datatank/LRT13/repos/ART-DETR/configs/hiera/", job_name="sam2_init"):
      cfg = compose(config_name=config_file)
      OmegaConf.resolve(cfg)
      model = instantiate(cfg.model, _recursive_=True)
      _load_checkpoint(model, ckpt_path)
      model = model.to(device)
      if mode == "eval":
         model.eval()
      return model

@register()
class PHiera(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = build_hiera_backbone("hiera_t.yaml", "./checkpoints/hiera_sam2_t.pt" , device="cuda")
        for param in self.model.parameters():
         param.requires_grad = False

    def forward(self, x: torch.Tensor, return_intermediates: bool = True) -> torch.Tensor:
        out = self.model(x)
        if(return_intermediates):
            return out['backbone_fpn']
        else:
         return out