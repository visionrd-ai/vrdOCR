# helpers_ctc.py
from typing import List
import torch

def toyo_collate(batch):
    """
    Each dataset item = [img, label_ctc, label_gtc, length, weight]
      - img is already CHW float tensor if you used ToTensorV2 in transforms
      - label_ctc is padded to max_text_length (LongTensor)
      - length is the valid length for CTC targets (LongTensor scalar)
    Returns:
      images:       (B, C, H, W) float
      label_ctc:    (B, max_len) long
      lengths:      (B,) long
      weights:      (B,) float
    """
    imgs, lab_ctc, lab_gtc, lens, wts = zip(*batch)
    images    = torch.stack(imgs, dim=0)                # already normalized by Albumentations
    label_ctc = torch.stack(lab_ctc, dim=0).long()
    lengths   = torch.stack(lens,    dim=0).long().view(-1)
    weights   = torch.stack(wts,     dim=0).float().view(-1)
    return images, label_ctc, lengths, weights

def pack_ctc_targets(label_ctc: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    (B, max_len) + (B,) -> flat targets for CTCLoss.
    Assumes label ids in [1..N] (blank=0 is not used in targets).
    """
    parts = []
    B = label_ctc.size(0)
    for i in range(B):
        L = int(lengths[i].item())
        if L > 0:
            parts.append(label_ctc[i, :L])
    if not parts:
        return torch.zeros((0,), dtype=torch.long, device=label_ctc.device)
    return torch.cat(parts, dim=0).long()

def decode_gt_batch_ctc(label_ctc: torch.Tensor, lengths: torch.Tensor, charset: List[str], blank_idx: int = 0) -> List[str]:
    """
    Convert GT ids (no blanks, no collapse) back to strings using the same charset as the decoder.
    If blank_idx == 0, map k -> charset[k-1].
    """
    texts = []
    N = len(charset)
    for i in range(label_ctc.size(0)):
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
