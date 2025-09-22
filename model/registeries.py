# /ETESVS/model/builder.py
from typing import Any, Dict, Optional, Union, List

from utils.builder import Registry, build

BACKBONES       = Registry('backbone')
NECKS           = Registry('neck')
HEADS           = Registry('head')
ARCHITECTURE    = Registry('architecture')
LOSSES          = Registry('loss')
POSTPROCESSING = Registry('postprocessing')

# Expose component builders so the ARCHITECTURE class can call them internally
def build_backbone(cfg: Optional[Dict[str, Any]]):
    return build(cfg, BACKBONES) if cfg else None

def build_neck(cfg: Optional[Dict[str, Any]]):
    return build(cfg, NECKS) if cfg else None

def build_head(cfg: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], **kw):
    return build(cfg, HEADS, **kw) if cfg else None

def build_loss(cfg: Optional[Dict[str, Any]]):
    return build(cfg, LOSSES) if cfg else None

def build_architecture(cfg: Dict[str, Any]):
    return build(cfg, ARCHITECTURE, key='architecture')

def build_postprocessing(cfg: Optional[Dict[str, Any]]):
    return build(cfg, POSTPROCESSING) if cfg else None  

def build_model(cfg: Dict[str, Any]):
    """
    Only architecture-driven builds:
      - If cfg contains 'MODEL', flatten and pass MODEL.* directly to ARCHITECTURE
      - Else expect flat {'architecture': '...'} dict
    The registered architecture is responsible for building its backbone/head/etc.
    """
    if 'MODEL' in cfg:
        arch_cfg = cfg['MODEL'].copy()
        if 'architecture' not in arch_cfg:
            raise KeyError("MODEL must contain 'architecture' when using architecture-driven builds.")
        return build_architecture(arch_cfg)

    if 'architecture' in cfg:
        return build_architecture(cfg)

    raise ValueError("Config must contain either 'MODEL' with 'architecture', or flat 'architecture'.")
