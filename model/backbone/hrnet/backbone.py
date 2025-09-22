import torch.nn as nn
from functools import partial
from model.backbone.hrnet.factory import hrnet18, hrnet32, hrnet48
from model.registeries import BACKBONES

def _maybe_freeze(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

class HRNetBackbone(nn.Module):
    """
    Registry-facing wrapper. Build with:
      HRNetBackbone(variant='hrnet32', freeze_backbone=False, pretrained=True, progress=True, **kwargs)

    Notes:
    - **variant** is fixed by registration (see backbones/__init__.py).
    - Keep kwargs open for future HRNet options (norm layer, etc.).
    """
    def __init__(self,
                 variant: str = 'hrnet32',
                 freeze_backbone: bool = False,
                 pretrained: bool = True,
                 progress: bool = True,
                 **kwargs):
        super().__init__()
        # build core
        if variant == 'hrnet18':
            self.backbone = hrnet18(pretrained=pretrained, progress=progress, **kwargs)
        elif variant == 'hrnet32':
            self.backbone = hrnet32(pretrained=pretrained, progress=progress, **kwargs)
        elif variant == 'hrnet48':
            self.backbone = hrnet48(pretrained=pretrained, progress=progress, **kwargs)
        else:
            raise ValueError(f"Unsupported HRNet variant: {variant}")

        if freeze_backbone:
            print(f"[HRNetBackbone] Freezing parameters for {variant}.")
            _maybe_freeze(self.backbone)

    def forward(self, x):
        return self.backbone(x)

def register_hrnet_variants():
    """
    Call once (e.g., in backbones/__init__.py) to expose:
      name: HRNET18 / HRNET32 / HRNET48
    which match your YAML `backbone.name`.
    """
    BACKBONES.register(name="HRNET18")(partial(HRNetBackbone, variant='hrnet18'))
    BACKBONES.register(name="HRNET32")(partial(HRNetBackbone, variant='hrnet32'))
    BACKBONES.register(name="HRNET48")(partial(HRNetBackbone, variant='hrnet48'))