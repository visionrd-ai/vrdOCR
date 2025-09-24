# helpers.py
import torch
from typing import List, Tuple

def ocr_collate(batch):
    """
    Batch is a list of samples where each sample = [img, label_ctc, label_gtc, length, weight]
      - img:       (H, W, C) uint8 or float; assumed same H,W across batch after transforms
      - label_ctc: (max_len,) Long
      - label_gtc: (max_len,) Long  (unused here)
      - length:    () Long  -- valid length for CTC targets
      - weight:    () Float -- optional sample weight
    Returns:
      images:        (B, 3, H, W) float32 in [0,1]
      label_ctc:     (B, max_len) Long
      lengths:       (B,) Long
      weights:       (B,) Float
    """
    imgs, lab_ctc, lab_gtc, lens, wts = zip(*batch)
    # images -> float32 CHW [0,1]
    imgs = [img.float() / 255.0 for img in imgs]
    imgs = [img.permute(2, 0, 1).contiguous() if img.ndim == 3 else img for img in imgs]
    images = torch.stack(imgs, dim=0)

    label_ctc = torch.stack(lab_ctc, dim=0).long()
    lengths   = torch.stack(lens, dim=0).long().view(-1)
    weights   = torch.stack(wts,  dim=0).float().view(-1)
    return images, label_ctc, lengths, weights

def pack_ctc_targets(label_ctc: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert (B, max_len) + (B,) -> flat targets for CTCLoss.
    Assumes label_ctc entries are in [1..N] (blank=0 is not used in targets).
    """
    parts = []
    B = label_ctc.size(0)
    for i in range(B):
        L = int(lengths[i].item())
        if L > 0:
            parts.append(label_ctc[i, :L])
    if len(parts) == 0:
        return torch.zeros((0,), dtype=torch.long, device=label_ctc.device)
    return torch.cat(parts, dim=0).long()

def decode_gt_batch_ctc(label_ctc: torch.Tensor, lengths: torch.Tensor, charset: List[str], blank_idx: int = 0) -> List[str]:
    """
    Map ground-truth CTC ids (no blanks, no collapse) to strings.
    If blank_idx==0, characters are in [1..N] -> charset[k-1].
    """
    texts = []
    B = label_ctc.size(0)
    N = len(charset)
    for i in range(B):
        L = int(lengths[i].item())
        ids = label_ctc[i, :L].tolist()
        chars = []
        for k in ids:
            if blank_idx == 0:
                ch_idx = k - 1
            else:
                ch_idx = k if k < blank_idx else (k - 1)
            if 0 <= ch_idx < N:
                chars.append(charset[ch_idx])
        texts.append("".join(chars))
    return texts
