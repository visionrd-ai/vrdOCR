import torch
import torch.nn as nn
from typing import Dict, Any, List, Union

from model.registeries import HEADS, build_head


def _normalize_head_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts both legacy head_list entries like:
      - {"CTCHead": {...}} or {"NRTRHead": {...}}
    and the updated entries like:
      - {"name": "CTCHead", "decoder_key": "...", ...}

    Returns a unified dict:
      {"name": <str>, "decoder_key": <str or None>, ...rest }
    """
    if "name" in item:
        # new format
        name = item["name"]
        out = dict(item)
        return out
    else:
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
    """
    Registry-ready MultiHead that composes multiple heads (CTC + NRTR, no SAR).
    It constructs each sub-head via HEADS registry using the provided per-head config.

    Expected top-level config (matches your YAML):
      - in_channels: int
      - out_channels_list: dict mapping decoder_key -> vocab size
      - head_list: list of per-head dicts. Each entry supports:
          updated style:
            { name: "CTCHead",
              decoder_key: "CTCLabelDecode",
              neck: {...}, head: {...} }
            { name: "NRTRHead",
              decoder_key: "NRTRLabelDecode",
              nrtr_dim: 384, max_text_length: 150, ... }
          legacy style:
            { "CTCHead": { decoder_key: ..., neck: {...}, head:{...} } }
            { "NRTRHead": { decoder_key: ..., nrtr_dim: ..., ... } }

    Forward behavior:
      - train: returns a dict with:
          "ctc": logits (B, T, C) (or, if CTCHead was configured with return_feats=True,
                                   "ctc_neck" contains sequence features and "ctc" contains logits)
          and, if NRTR present:
          "gtc": logits from NRTR Transformer (B, T, V)
      - eval: returns CTC probabilities only (backward-compatible with your prior MultiHead)
    """
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

        # storage
        self.subheads = nn.ModuleDict()     # name -> module instance
        self._ctc_key = None                # decoder key for ctc
        self._nrtr_key = None               # decoder key for nrtr

        # build each sub-head from registry
        for raw in head_list:
            cfg = _normalize_head_item(raw)
            name = cfg.pop("name")
            decoder_key = cfg.pop("decoder_key", None)

            if decoder_key is None:
                raise KeyError(f"[MultiHead] '{name}' entry must include 'decoder_key'")

            if decoder_key not in self.out_channels_list:
                raise KeyError(f"[MultiHead] decoder_key '{decoder_key}' not found in out_channels_list={list(self.out_channels_list.keys())}")

            sub_cfg = {
                "name": name,
                "in_channels": in_channels,
                "out_channels": self.out_channels_list[decoder_key],
                "decoder_key": decoder_key,
                **cfg,  # pass through head-specific params (e.g., neck/head blocks for CTC; dims for NRTR)
            }

            # build via registry
            sub_head = build_head(sub_cfg)
            self.subheads[name] = sub_head

            # remember which types we have
            if name == "CTCHead":
                self._ctc_key = decoder_key
            elif name == "NRTRHead":
                self._nrtr_key = decoder_key
            else:
                # ignore any other heads silently or raise; since you asked to exclude SAR, we guard here
                raise NotImplementedError(f"[MultiHead] Head '{name}' is not supported (SAR removed).")

        # basic sanity
        if self._ctc_key is None:
            raise AssertionError("[MultiHead] CTCHead is required in head_list.")
        # NRTR is optional; we’ll branch on presence at forward-time

    def _maybe_pool(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_pool:
            return x
        return self.pool(x)

    def _split_ctc_outputs(self, ctc_out: Any):
        """
        CTCHead can be configured with head.return_feats=True, which returns (feats, logits) during training.
        Normalize to (seq_feats|None, logits_or_probs).
        """
        if isinstance(ctc_out, tuple) and len(ctc_out) == 2:
            feats, logits = ctc_out
            return feats, logits
        return None, ctc_out

    def _select_targets(self, targets, decoder_key: str):
        """
        Flexible target router:
          - if targets is a dict, use targets[decoder_key] (e.g., 'CTCLabelDecode' or 'NRTRLabelDecode')
          - else, pass through as-is (legacy behavior)
        """
        if isinstance(targets, dict):
            return targets.get(decoder_key, None)
        return targets

    def forward(self, x: torch.Tensor, targets: Union[None, Dict[str, Any], Any] = None):
        # optional pooling (kept for backward-compat with your original)
        x = self._maybe_pool(x)

        outputs = {}

        # --- CTC path (required) ---
        ctc_head = self.subheads.get("CTCHead", None)
        assert ctc_head is not None, "CTCHead must exist"
        ctc_targets = self._select_targets(targets, self._ctc_key)
        ctc_out = ctc_head(x, targets=ctc_targets)
        ctc_feats, ctc_pred = self._split_ctc_outputs(ctc_out)

        # eval behavior: return CTC only (matches your old MultiHead)
        if not self.training:
            return ctc_pred

        outputs["ctc"] = ctc_pred
        if ctc_feats is not None:
            outputs["ctc_neck"] = ctc_feats

        # --- NRTR path (optional) ---
        nrtr_head = self.subheads.get("NRTRHead", None)
        if nrtr_head is not None:
            nrtr_targets = self._select_targets(targets, self._nrtr_key)
            gtc_out = nrtr_head(x, targets=nrtr_targets)
            outputs["gtc"] = gtc_out

        return outputs


