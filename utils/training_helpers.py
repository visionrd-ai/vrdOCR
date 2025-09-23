import csv
from typing import Dict, List, Tuple, Iterable, Any
import torch
import numpy as np

def init_histories(loss_names: Iterable[str]):
    """
    Build step-wise and epoch-wise history dicts with a generic 'other_losses' map.
    """
    loss_names = list(loss_names)

    history = {
        "steps": [],
        "total_loss": [],
        "other_losses": {name: [] for name in loss_names},  # per-step lists
        "train_acc": [],
        "eval_steps": [],
        "eval_acc": [],
    }

    epoch_history = {
        "epoch": [],
        "total_loss": [],
        "other_losses": {name: [] for name in loss_names},  # per-epoch lists
        "train_acc": [],
        "eval_acc": [],
        "lr": [],
    }
    return history, epoch_history


def init_epoch_accumulators(loss_names: Iterable[str]):
    """
    Accumulators for epoch means.
    """
    acc = {
        "total_loss": 0.0,
        "acc": 0.0,
        "count": 0,
        "loss_sums": {name: 0.0 for name in loss_names},
    }
    return acc


def _to_float(x) -> float:
    """Robustly convert tensors / numpy scalars / python scalars to float."""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, (np.floating,)):
        return float(x)
    # numpy array with 0-dim
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return float(x.item())
    return float(x)

def update_step_history(
    history: Dict[str, Any],
    global_step: int,
    total_loss,
    loss_dict: Dict[str, Any],
    train_acc: float,
):
    # ensure containers exist
    if "steps" not in history: history["steps"] = []
    if "total_loss" not in history: history["total_loss"] = []
    if "train_acc" not in history: history["train_acc"] = []
    if "other_losses" not in history or not isinstance(history["other_losses"], dict):
        history["other_losses"] = {}

    history["steps"].append(int(global_step))
    history["total_loss"].append(_to_float(total_loss))
    history["train_acc"].append(float(train_acc))

    for name, val in (loss_dict or {}).items():
        # create list if missing
        history["other_losses"].setdefault(name, [])
        history["other_losses"][name].append(_to_float(val))


def update_epoch_accumulators(
    ep_acc: Dict[str, Any],
    total_loss: torch.Tensor,
    loss_dict: Dict[str, torch.Tensor],
    train_acc: float,
):
    ep_acc["total_loss"] += float(_to_float(total_loss))
    ep_acc["acc"] += float(train_acc)
    for name, val in loss_dict.items():
        ep_acc["loss_sums"][name] += float(_to_float(val))
    ep_acc["count"] += 1


def finalize_epoch_means(
    ep_acc: Dict[str, Any]
) -> Tuple[float, Dict[str, float], float]:
    denom = max(1, ep_acc["count"])
    total_mean = ep_acc["total_loss"] / denom
    loss_means = {k: v / denom for k, v in ep_acc["loss_sums"].items()}
    acc_mean = ep_acc["acc"] / denom
    return total_mean, loss_means, acc_mean


def write_epoch_header(epoch_csv_path, loss_names: List[str]):
    """
    Create a CSV with dynamic columns for the additional losses.
    """
    cols = ["epoch", "train_acc", "eval_acc", "total_loss", *loss_names, "lr"]
    with open(epoch_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)
    return cols


def write_epoch_row(epoch_csv_path, cols: List[str], row_values: Dict[str, Any]):
    """
    Write one epoch line to CSV using the provided column order.
    """
    with open(epoch_csv_path, "a", newline="") as f:
        csv.writer(f).writerow([row_values.get(c, "") for c in cols])


def log_close_examples_csv(csv_path, rows):
    if not rows:
        return
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)


def format_loss_line(total_loss: torch.Tensor, loss_dict: Dict[str, torch.Tensor]) -> str:
    parts = [f"Total: {total_loss.item():.3f}"]
    for k, v in loss_dict.items():
        parts.append(f"{k}: {v:.3f}")
    return " | ".join(parts)
