from typing import List, Dict, Any, Optional
import io
import os
import torch
import warnings

from model.registeries import POSTPROCESSING

def _load_charset_from_file(path: str, use_space_char: bool = True) -> List[str]:
    """
    Loads one character per line from `path` (UTF-8).
    IMPORTANT: we DO NOT strip() whitespace so that a single-space line is preserved.
    We only remove newline/carriage return characters.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"character_dict_path not found: {path}")

    charset: List[str] = []
    with io.open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            ch = raw.rstrip("\r\n")        # keep spaces; only drop newline chars
            if ch == "":                   # skip truly empty lines
                continue
            charset.append(ch)

    if use_space_char and " " not in charset:
        charset.append(" ")

    # De-duplicate while preserving order (just in case)
    seen = set()
    deduped = []
    for ch in charset:
        if ch not in seen:
            deduped.append(ch)
            seen.add(ch)
    return deduped

@POSTPROCESSING.register(name="CTCGreedy")
class CTCGreddyDecode:
    """
    Paddle-like CTC label decoder.

    Config:
      - character_dict_path: str (file with one character per line; may include a ' ' line)
      - use_space_char: bool (if True and ' ' missing, it's appended)
      - blank_idx: int = 0  (typical Paddle setup)
      - remove_repeats: bool = True (CTC collapse rule)

    Usage:
      decoder = build_postprocessing({
          "name": "CTCLabelDecode",
          "character_dict_path": "./data/en_dict.txt",
          "use_space_char": True,
          "blank_idx": 0
      })
      preds = decoder(logits)  # logits: (T, B, C)
    """
    def __init__(
        self,
        character_dict_path: str,
        use_space_char: bool = True,
        blank_idx: int = 0,
        remove_repeats: bool = True
    ):
        self.blank_idx = int(blank_idx)
        self.remove_repeats = bool(remove_repeats)
        self.charset: List[str] = _load_charset_from_file(character_dict_path, use_space_char)

    # ----- helpers -----
    def _ids_to_text(self, ids: List[int]) -> str:
        """
        Convert class indices (post-argmax) into string:
          - remove blanks
          - collapse repeats (if enabled)
          - map class k -> charset[k-1] if blank_idx == 0
        """
        text_chars: List[str] = []
        prev = None
        for k in ids:
            # skip blanks
            if k == self.blank_idx:
                prev = k
                continue
            # collapse repeats
            if self.remove_repeats and prev == k:
                prev = k
                continue

            # map to charset
            if self.blank_idx == 0:
                # logits classes: [blank] + charset -> indices 0..N
                ch_idx = k - 1
            else:
                # generic mapping when blank is not zero:
                # assume classes are charset with one extra blank at `blank_idx`
                # shift everything above blank_idx down by 1
                ch_idx = k if k < self.blank_idx else (k - 1)

            if 0 <= ch_idx < len(self.charset):
                text_chars.append(self.charset[ch_idx])
            # else: invalid index; skip silently
            prev = k
        return "".join(text_chars)

    # ----- main decode -----
    @torch.no_grad()
    def __call__(self, logits: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Greedy CTC decode.

        Args:
          logits: (T, B, C) unnormalized or log-probs OK (we softmax here)

        Returns:
          List[{"text": str, "score": float}] length B
            - text: decoded string
            - score: mean per-timestep max prob as a simple confidence
        """
        if logits.dim() != 3:
            raise ValueError(f"Expected logits of shape (T, B, C); got {tuple(logits.shape)}")

        T, B, C = logits.shape
        N = len(self.charset)

        # Sanity check on class dimension vs dict size + blank
        expected_C = N + 1  # [blank] + charset
        if C != expected_C:
            warnings.warn(
                f"[CTCLabelDecode] logits C={C} does not match expected N+1={expected_C} "
                f"(dict={N}, blank_idx={self.blank_idx}). "
                f"Double-check head.num_classes and dictionary length.",
                stacklevel=2
            )

        probs  = logits.softmax(dim=2)        # (T, B, C)
        max_k  = probs.argmax(dim=2)          # (T, B)
        max_p  = probs.max(dim=2).values      # (T, B)

        results: List[Dict[str, Any]] = []
        for b in range(B):
            ids = max_k[:, b].tolist()
            text = self._ids_to_text(ids)
            score = float(max_p[:, b].mean().item())
            results.append({"text": text, "score": score})
        return results

    # Optional utility (can help you inspect the loaded dict)
    def get_charset(self) -> List[str]:
        return list(self.charset)
