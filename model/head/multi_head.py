import torch
import torch.nn as nn
from typing import Dict, Any, List, Union

from model.registeries import HEADS, build_head


def _normalize_head_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if "name" in item:             # new format
        return dict(item)
    # legacy: {"CTCHead": {...}}
    assert len(item) == 1, f"Invalid head_list entry: {item}"
    name = list(item.keys())[0]
    cfg = item[name]
    out = {"name": name}
    if isinstance(cfg, dict):
        out.update(cfg)
    return out


@HEADS.register(name="MultiHead")
class MultiHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_list: Dict[str, int],
        head_list: List[Dict[str, Any]],
        use_pool: bool = False,
        pool_kernel: tuple = (3, 2),
        pool_stride: tuple = (3, 2),
        pool_padding: tuple = (0, 0),
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_list = out_channels_list
        self.use_pool = use_pool

        if self.use_pool:
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        self.subheads = nn.ModuleDict()   # {"CTCHead": module, "NRTRHead": module}
        self._ctc_key = None
        self._nrtr_key = None

        for raw in head_list:
            cfg = _normalize_head_item(raw)
            name = cfg.pop("name")
            decoder_key = cfg.pop("decoder_key", None)
            if decoder_key is None:
                raise KeyError(f"[MultiHead] '{name}' requires 'decoder_key'")
            if decoder_key not in self.out_channels_list:
                raise KeyError(f"[MultiHead] decoder_key '{decoder_key}' not in out_channels_list={list(self.out_channels_list.keys())}")

            sub_cfg = {
                "name": name,
                "in_channels": in_channels,
                "out_channels": self.out_channels_list[decoder_key],
                "decoder_key": decoder_key,
                **cfg,
            }
            sub_head = build_head(sub_cfg)
            self.subheads[name] = sub_head

            if name == "CTCHead":
                self._ctc_key = decoder_key
            elif name == "NRTRHead":
                self._nrtr_key = decoder_key
            else:
                raise NotImplementedError(f"[MultiHead] Head '{name}' not supported (SAR removed).")

        if self._ctc_key is None:
            raise AssertionError("[MultiHead] CTCHead is required in head_list.")

    def _maybe_pool(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) if self.use_pool else x

    def _maybe(self, key: str):
        # SAFE accessor for ModuleDict
        return self.subheads[key] if key in self.subheads else None

    def _split_ctc_outputs(self, ctc_out: Any):
        if isinstance(ctc_out, tuple) and len(ctc_out) == 2:
            feats, logits = ctc_out
            return feats, logits
        return None, ctc_out

    def _select_targets(self, targets, decoder_key: str):
        if isinstance(targets, dict):
            return targets.get(decoder_key, None)
        return targets

    def forward(self, x: torch.Tensor, targets: Union[None, Dict[str, Any], Any] = None):
        x = self._maybe_pool(x)

        # --- CTC path (required) ---
        ctc_head = self._maybe("CTCHead")  # << no .get()
        if ctc_head is None:
            raise RuntimeError("CTCHead is required but not found in subheads.")
        ctc_targets = self._select_targets(targets, self._ctc_key)
        ctc_out = ctc_head(x, targets=ctc_targets)
        ctc_feats, ctc_pred = self._split_ctc_outputs(ctc_out)

        if not self.training:
            return ctc_pred  # eval returns CTC only (backward compatible)

        outputs = {"ctc": ctc_pred}
        if ctc_feats is not None:
            outputs["ctc_neck"] = ctc_feats

        # --- NRTR path (optional) ---
        nrtr_head = self._maybe("NRTRHead")  # << no .get()
        if nrtr_head is not None:
            nrtr_targets = self._select_targets(targets, self._nrtr_key)
            gtc_out = nrtr_head(x, targets=nrtr_targets)
            outputs["gtc"] = gtc_out

        return outputs
