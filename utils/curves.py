# utils/save_curves.py

import json
from pathlib import Path
from typing import Dict, List, Optional

# single shared live figure (2x2)
_LIVE_FIG = None
_AX_STEP_LOSS = None
_AX_STEP_ACC = None
_AX_EPOCH_LOSS = None
_AX_EPOCH_ACC = None


def _select_backend(vis: bool):
    import matplotlib
    if vis:
        # prefer interactive; fallback to Agg
        for cand in ("Qt5Agg", "TkAgg", "MacOSX"):
            try:
                matplotlib.use(cand, force=True)
                return
            except Exception:
                continue
        matplotlib.use("Agg", force=True)
    else:
        matplotlib.use("Agg", force=True)


def _truncate_pair(xs: List[float], ys: List[float]):
    n = min(len(xs), len(ys))
    return xs[:n], ys[:n]


def _plot_losses(ax, xs: List[float], total: List[float],
                 others: Optional[Dict[str, List[float]]] = None,
                 xlab="Step", title="Loss"):
    ax.clear()
    x, y = _truncate_pair(xs, total)
    if len(x) > 0:
        ax.plot(x, y, label="Total")
    for name, series in (others or {}).items():
        x2, y2 = _truncate_pair(xs, series)
        if len(x2) > 0:
            ax.plot(x2, y2, label=str(name))
    ax.set_xlabel(xlab); ax.set_ylabel("Loss"); ax.set_title(title); ax.legend()
    ax.grid(True, alpha=0.25)


def _plot_acc_steps(ax, steps: List[float], train_acc: List[float],
                    eval_steps: Optional[List[float]] = None,
                    eval_acc: Optional[List[float]] = None,
                    title="Accuracy (Step)"):
    ax.clear()
    x, y = _truncate_pair(steps, train_acc)
    if len(x) > 0:
        ax.plot(x, y, label="Train Acc")
    if eval_steps and eval_acc:
        xs, ys = _truncate_pair(eval_steps, eval_acc)
        if len(xs) > 0:
            ax.plot(xs, ys, label="Eval Acc")
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy (%)"); ax.set_title(title); ax.legend()
    ax.grid(True, alpha=0.25)


def _plot_acc_epochs(ax, epochs: List[int], train_acc: List[float],
                     eval_acc: Optional[List[Optional[float]]] = None,
                     title="Accuracy (Epoch)"):
    import math
    ax.clear()
    e, y = _truncate_pair(epochs, train_acc)
    if len(e) > 0:
        ax.plot(e, y, label="Train Acc")
    if eval_acc is not None:
        ys = [float("nan") if v is None else v for v in eval_acc]
        e2, y2 = _truncate_pair(epochs, ys)
        if len(e2) > 0:
            ax.plot(e2, y2, label="Eval Acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title(title); ax.legend()
    ax.grid(True, alpha=0.25)


def _ensure_live_canvas():
    """Create a single 2x2 live window if not already created."""
    global _LIVE_FIG, _AX_STEP_LOSS, _AX_STEP_ACC, _AX_EPOCH_LOSS, _AX_EPOCH_ACC
    if _LIVE_FIG is None:
        import matplotlib.pyplot as plt
        plt.ion()
        _LIVE_FIG, axs = plt.subplots(2, 2, figsize=(14, 9))
        _AX_STEP_LOSS, _AX_STEP_ACC = axs[0]
        _AX_EPOCH_LOSS, _AX_EPOCH_ACC = axs[1]
        # set a window title when supported
        try:
            _LIVE_FIG.canvas.manager.set_window_title("Training — Live Curves")
        except Exception:
            pass


def save_curves(run_dir: str, history: dict, vis: bool = False):
    """
    Step-level plots & snapshot.

    history:
      steps: List[int]
      total_loss: List[float]
      other_losses: Dict[str, List[float]]   (optional)
      train_acc: List[float]
      eval_steps: List[int]  (optional)
      eval_acc: List[float]  (optional)
    """
    _select_backend(vis)
    import matplotlib.pyplot as plt

    rd = Path(run_dir); rd.mkdir(parents=True, exist_ok=True)

    steps: List[float] = history.get("steps", [])
    total_loss: List[float] = history.get("total_loss", [])
    other_losses: Dict[str, List[float]] = history.get("other_losses", {}) or {}
    train_acc: List[float] = history.get("train_acc", [])
    eval_steps: List[float] = history.get("eval_steps", [])
    eval_acc: List[float] = history.get("eval_acc", [])

    # save static PNGs
    fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(1, 1, 1)
    _plot_losses(ax, steps, total_loss, other_losses, xlab="Step", title="Loss vs Step")
    fig.tight_layout(); fig.savefig(rd / "loss_curve.png", dpi=160); plt.close(fig)

    fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(1, 1, 1)
    _plot_acc_steps(ax, steps, train_acc, eval_steps, eval_acc, title="Accuracy vs Step")
    fig.tight_layout(); fig.savefig(rd / "accuracy_curve.png", dpi=160); plt.close(fig)

    with open(rd / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # live: update top row only (step loss + step acc)
    if vis:
        _ensure_live_canvas()
        _plot_losses(_AX_STEP_LOSS, steps, total_loss, other_losses,
                     xlab="Step", title="Loss (Step, Live)")
        _plot_acc_steps(_AX_STEP_ACC, steps, train_acc, eval_steps, eval_acc,
                        title="Accuracy (Step, Live)")

        _LIVE_FIG.tight_layout()
        _LIVE_FIG.canvas.draw_idle()
        plt.pause(0.001)  # non-blocking UI refresh


def save_epoch_curves(run_dir: str, epoch_history: dict, vis: bool = False):
    """
    Epoch-level plots & snapshot.

    epoch_history:
      epoch: List[int]
      total_loss: List[float]
      other_losses: Dict[str, List[float]]   (optional)
      train_acc: List[float]
      eval_acc: List[Optional[float]]        (optional; can contain None)
    """
    _select_backend(vis)
    import matplotlib.pyplot as plt

    rd = Path(run_dir); rd.mkdir(parents=True, exist_ok=True)

    epochs: List[int] = epoch_history.get("epoch", [])
    total_loss: List[float] = epoch_history.get("total_loss", [])
    other_losses: Dict[str, List[float]] = epoch_history.get("other_losses", {}) or {}
    train_acc: List[float] = epoch_history.get("train_acc", [])
    eval_acc: List[Optional[float]] = epoch_history.get("eval_acc", [])

    # save static PNGs
    fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(1, 1, 1)
    _plot_losses(ax, epochs, total_loss, other_losses, xlab="Epoch", title="Loss vs Epoch")
    fig.tight_layout(); fig.savefig(rd / "loss_curve_epoch.png", dpi=160); plt.close(fig)

    fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(1, 1, 1)
    _plot_acc_epochs(ax, epochs, train_acc, eval_acc, title="Accuracy vs Epoch")
    fig.tight_layout(); fig.savefig(rd / "accuracy_curve_epoch.png", dpi=160); plt.close(fig)

    with open(rd / "epoch_history.json", "w") as f:
        json.dump(epoch_history, f, indent=2)

    # live: update bottom row only (epoch loss + epoch acc)
    if vis:
        _ensure_live_canvas()
        _plot_losses(_AX_EPOCH_LOSS, epochs, total_loss, other_losses,
                     xlab="Epoch", title="Loss (Epoch, Live)")
        _plot_acc_epochs(_AX_EPOCH_ACC, epochs, train_acc, eval_acc,
                         title="Accuracy (Epoch, Live)")

        _LIVE_FIG.tight_layout()
        _LIVE_FIG.canvas.draw_idle()
        plt.pause(0.001)  # non-blocking
