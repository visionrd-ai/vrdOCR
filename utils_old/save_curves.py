import matplotlib
matplotlib.use("Agg")          # headless render
import matplotlib.pyplot as plt
import json
from pathlib import Path


# --- NEW: helper to save plots ---
def save_curves(run_dir: str, history: dict):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Loss curves
    plt.figure()
    plt.plot(history["steps"], history["total_loss"], label="Total")
    plt.plot(history["steps"], history["ctc_loss"],   label="CTC")
    plt.plot(history["steps"], history["nrtr_loss"],  label="NRTR")
    plt.xlabel("Train step"); plt.ylabel("Loss"); plt.title("Loss vs Step"); plt.legend()
    plt.tight_layout(); plt.savefig(run_dir / "loss_curve.png", dpi=160); plt.close()

    # Accuracy curves (train vs eval)
    plt.figure()
    plt.plot(history["steps"], history["train_acc"], label="Train Acc")
    if len(history["eval_steps"]) and len(history["eval_acc"]):
        plt.plot(history["eval_steps"], history["eval_acc"], label="Eval Acc")
    plt.xlabel("Step"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy vs Step"); plt.legend()
    plt.tight_layout(); plt.savefig(run_dir / "accuracy_curve.png", dpi=160); plt.close()

    # Persist history for later replotting
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
        
def save_epoch_curves(run_dir: str, epoch_history: dict):
    rd = Path(run_dir)
    # Epoch losses
    plt.figure()
    plt.plot(epoch_history["epoch"], epoch_history["total_loss"], label="Total")
    plt.plot(epoch_history["epoch"], epoch_history["ctc_loss"],   label="CTC")
    plt.plot(epoch_history["epoch"], epoch_history["nrtr_loss"],  label="NRTR")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(rd / "loss_curve_epoch.png", dpi=160); plt.close()

    # Epoch accuracies
    plt.figure()
    plt.plot(epoch_history["epoch"], epoch_history["train_acc"], label="Train Acc")
    if any(x is not None for x in epoch_history["eval_acc"]):
        # fill holes with NaN for a clean line
        xs = epoch_history["epoch"]
        ys = [y if y is not None else float("nan") for y in epoch_history["eval_acc"]]
        plt.plot(xs, ys, label="Eval Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy vs Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(rd / "accuracy_curve_epoch.png", dpi=160); plt.close()

    # Persist epoch history
    with open(rd / "epoch_history.json", "w") as f:
        json.dump(epoch_history, f, indent=2)