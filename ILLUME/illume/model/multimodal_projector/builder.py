from abc import ABC
from functools import partial

import torch
import torch.nn as nn
import re

from illume.registry_utils import build_from_cfg

from einops import rearrange, repeat
from . import MM_PROJECTOR


class BaseMMProjector(ABC, nn.Module):
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def tune(self):
        for p in self.parameters():
            p.requires_grad = True
    
    @property
    def downsample_rate(self):
        return 1
    
    @property
    def downsample_rate_per_side(self):
        return 1


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super(LambdaLayer, self).__init__()
        self.fn = fn
    
    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@MM_PROJECTOR.register_module()
class IdentityMap(BaseMMProjector):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x
    
    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


@MM_PROJECTOR.register_module()
class SimpleResBlock(BaseMMProjector):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


@MM_PROJECTOR.register_module()
class MLPProjector(nn.Sequential, BaseMMProjector):
    def __init__(self, mm_hidden_size, hidden_size, mlp_depth=2, pre_norm=False):
        modules = []
        if pre_norm:
            modules.append(nn.LayerNorm(mm_hidden_size))
        modules.append(nn.Linear(mm_hidden_size, hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        super(MLPProjector, self).__init__(*modules)


@MM_PROJECTOR.register_module()
class MixedProjector(nn.Module):
    def __init__(self, projector_cfg1, projector_cfg2, mm_hidden_size, hidden_size, **kwargs):
        super(MixedProjector, self).__init__()
        projector_cfg1.update(mm_hidden_size=mm_hidden_size[0], hidden_size=hidden_size)
        projector_cfg2.update(mm_hidden_size=mm_hidden_size[1], hidden_size=hidden_size)
        self.projector_1 = build_from_cfg(projector_cfg1, MM_PROJECTOR)
        self.projector_2 = build_from_cfg(projector_cfg2, MM_PROJECTOR)
    
    def tune(self):
        self.projector_1.tune()
        self.projector_2.tune()
    
    def freeze(self):
        self.projector_1.freeze()
        self.projector_2.freeze()
    
    def downsample_rate(self):
        return self.projector_1.downsample_rate
    
    @property
    def downsample_rate_per_side(self):
        return self.projector_1.downsample_rate_per_side
    
    def forward(self, image_features):
        image_feature_1, image_feature_2 = image_features
        image_feature_1 = self.projector_1(image_feature_1)
        image_feature_2 = self.projector_2(image_feature_2)
        image_features = torch.concat([image_feature_1, image_feature_2], dim=1)
        return image_features


def build_mm_projector(mm_projector_cfg, **kwargs):
    mm_projector_cfg.update(kwargs)
    trainable = mm_projector_cfg.pop('trainable', True)
    model = build_from_cfg(mm_projector_cfg, MM_PROJECTOR)
    
    if trainable:
        model.tune()
    else:
        model.freeze()
    return model


def build_vision_projector(vision_projector_cfg, **kwargs):
    return build_mm_projector(vision_projector_cfg, **kwargs)
