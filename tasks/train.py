import csv, os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rapidfuzz.distance import Levenshtein

import data_old
import data_old.toyota_dataset

from model.registeries import build_loss, build_postprocessing
from utils.curves import save_curves, save_epoch_curves
from utils.training_helpers import (
    init_histories, init_epoch_accumulators, update_step_history,
    update_epoch_accumulators, finalize_epoch_means,
    write_epoch_header, write_epoch_row, log_close_examples_csv,
    format_loss_line
)
from src.metric import RecMetric


def _yaml_get(cfg: Dict[str, Any], dotted: str, default=None):
    """
    Simple helper to fetch nested keys via a dotted path.
    """
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node

def _get_transforms():
    # ----------------------------
    # Transforms
    # ----------------------------
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.Affine(scale=(1.0, 1.0), rotate=(-0.005, -0.005),
                 translate_percent=(-0.005, 0.005), shear=(-0.005, -0.005), p=0.6),
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.4),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.Resize(width=320, height=32),
        ToTensorV2(),
    ])

    eval_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.Resize(width=640, height=64),
        ToTensorV2(),
    ])
    
    return transform, eval_transform

def _get_dataloader(cfg):
    DATASET = cfg.get("DATASET", {})
    batch_size = int(cfg.get("batch_size", 2))
    num_workers = int(cfg.get("num_workers", 8))
    transform, eval_transform = _get_transforms()
    # ----------------------------
    # Datasets / Dataloaders
    # ----------------------------
    dataset = data_old.toyota_dataset.ToyotaDataset(
        dir=DATASET.get("dir", ""), 
        file=DATASET.get("train", ""), 
        max_len=DATASET.get("max_text_length", 150), 
        dict_path=DATASET.get("character_dict_path", "./data/en_dict.txt"),
        split="train", transforms=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    eval_dataset = data_old.toyota_dataset.ToyotaDataset(
        dir=DATASET.get("dir", ""), 
        file=DATASET.get("val", ""), 
        max_len=DATASET.get("max_text_length", 150), 
        dict_path=DATASET.get("character_dict_path", "./data/en_dict.txt"),
        split="val", transforms=eval_transform
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return dataset, dataloader, eval_dataloader, batch_size

def _get_csvs(run_dir: Path):
    # ----------------------------
    # Prepare run dir + CSVs
    # ----------------------------
    train_close_csv = run_dir / "train_close_examples.csv"
    with open(train_close_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "similarity", "pred", "target"])
    eval_close_csv = run_dir / "eval_close_examples.csv"
    with open(eval_close_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "batch", "similarity", "pred", "target"])

    epoch_csv = run_dir / "epoch_metrics.csv"
    
    return train_close_csv, eval_close_csv, epoch_csv

@torch.no_grad()
def _evaluate_once(
    epc: int,
    eval_iter: int,
    model,
    eval_loader,
    decoder,
    metric,
    close_thresh: float,
    print_every_n_batches: int,
    device: torch.device,
    run_dir: Path,
    run_name: str,
    history: Dict[str, Any],
    best_accuracy: float,
    logger,
    eval_close_csv
):
    model.eval()
    eval_iter += 1
    eval_accs = []

    for eval_idx, eval_batch in enumerate(eval_loader):
        eval_images = eval_batch[0].to(device)
        eval_outs = model(eval_images, labels=None)
        eval_predictions, eval_labels_decoded = decoder(eval_outs, eval_batch[1])

        metr = metric([eval_predictions, eval_labels_decoded])
        eval_accs.append(metr["acc"])

        # close examples
        preds = [p[0] for p in eval_predictions]
        gts = [g[0] for g in eval_labels_decoded]
        rows = []
        for p, t in zip(preds, gts):
            sim = Levenshtein.normalized_similarity(p.strip(), t.strip())
            if sim >= close_thresh:
                rows.append([epc, eval_idx, f"{sim:.4f}", p, t])
        log_close_examples_csv(eval_close_csv, rows)

        if eval_idx % print_every_n_batches == 0:
            logger.info(
                f"Epoch {epc}, Batch {eval_idx}/{len(eval_loader)} | "
                f"Eval Acc: {metr['acc']:.3f}% | Norm Edit: {metr['norm_edit_dis']:.3f}"
            )
            logger.info(
                f"Example — Pred: '{eval_predictions[0][0]}'  Label: '{eval_labels_decoded[0][0]}'"
            )

    overall = sum(eval_accs) / max(1, len(eval_accs))
    logger.info(f"Epoch {epc} | Overall Eval#{eval_iter} Acc: {overall:.3f}%")

    history["eval_steps"].append(history.get("steps", [])[-1] if history.get("steps") else 0)
    history["eval_acc"].append(overall)

    # best model by eval acc
    if overall > best_accuracy:
        best_accuracy = overall
        best_model_path = run_dir / f"{run_name}_best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"New best (eval) model saved: {best_accuracy:.3f}% -> {best_model_path}")

    return overall, best_accuracy


def train(cfg, model, run_dir: Path, run_name: str, logger, vis: bool = False):
    """
    Train loop with dynamic losses (any number) and modular utilities.
    """
    # ----------------------------
    # Decoder via registry (YAML>DECODER)
    # ----------------------------
    DECODER = cfg.get("DECODER", {})
    decoder = build_postprocessing(DECODER)[0]
    if decoder is None:
        raise ValueError("DECODER missing in YAML. Please add your Decoder block.")

    # ----------------------------
    # Training knobs from YAML
    # ----------------------------
    TRAIN = cfg.get("TRAIN", {})
    num_epochs = int(TRAIN.get("num_epochs", 1500))
    start_epoch = int(TRAIN.get("resume_checkpoint", 0)) if TRAIN.get("resume_checkpoint") else 0
    eval_every_n_batches = TRAIN.get("eval_interval", None)  # computed if None
    save_every_n_batches = int(TRAIN.get("save_interval", 500))
    print_every_n_batches = int(TRAIN.get("print_interval", 10))
    CLOSE_SIM_THRESH = float(TRAIN.get("close_similarity_thresh", 0.85))
    lr = float(TRAIN.get("lr", 1e-4))
    step_size = int(TRAIN.get("step_size", 10))
    gamma = float(TRAIN.get("gamma", 0.1))
    device_str = str(TRAIN.get("device", "cuda:0"))

    # ----------------------------
    # Datasets / Dataloaders
    # ----------------------------
    dataset, dataloader, eval_dataloader, batch_size = _get_dataloader(cfg)


    if eval_every_n_batches is None:
        eval_every_n_batches = len(dataset) // batch_size - 1

    # ----------------------------
    # Loss from YAML via registry
    # ----------------------------
    loss_cfg = _yaml_get(cfg, "MODEL.loss", None)
    if loss_cfg is None:
        raise ValueError("MODEL.loss missing in YAML. Please add your MultiLoss block.")
    loss_fn = build_loss(loss_cfg)
    
    dynamic_loss_names = None

    # ----------------------------
    # Optim / Scheduler / Metric
    # ----------------------------z
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    metric = RecMetric(logger=logger)

    # ----------------------------
    # CSVs
    # ----------------------------
    train_close_csv, eval_close_csv, epoch_csv = _get_csvs(run_dir)

    epoch_cols = []

    # ----------------------------
    # Device & train mode
    # ----------------------------
    device = torch.device(device_str)
    model.to(device).train()

    best_accuracy = 0.0
    global_step = 0

    # Bootstrap histories once we know the dynamic loss names
    history = None
    epoch_history = None
    ep_acc = None

    # ----------------------------
    # Training loop
    # ----------------------------
    for epc in range(start_epoch, num_epochs):
        eval_iter = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            images = batch[0].cuda()
            labels = batch[1:]
            labels = [label.cuda() for label in labels]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outs = model(images, labels)
                losses_out = loss_fn(outs, batch)  # {'loss': total, 'CTCLoss':..., 'NRTRLoss':...}

            total_loss = losses_out["loss"]
            # dynamic: pick everything except 'loss'
            step_loss_dict = {k: v for k, v in losses_out.items() if k != "loss"}

            # Initialize histories the first time we learn the loss names
            if dynamic_loss_names is None:
                dynamic_loss_names = list(step_loss_dict.keys())
                history, epoch_history = init_histories(dynamic_loss_names)
                ep_acc = init_epoch_accumulators(dynamic_loss_names)
                # write epoch CSV header dynamically
                epoch_cols = write_epoch_header(epoch_csv, dynamic_loss_names)
                logger.info(f"[train] Dynamic losses detected: {dynamic_loss_names}")

            # decode + metric
            preds, labels_dec = decoder(outs['ctc'], batch[1])
            metr = metric([preds, labels_dec])

            # record batch
            update_step_history(history, global_step, total_loss, step_loss_dict, metr["acc"])

            if batch_idx % print_every_n_batches == 0:
                logger.info(
                    f"Epoch {epc}/{num_epochs}, Batch {batch_idx}/{len(dataloader)} | "
                    f"Train Acc: {metr['acc']:.3f}% | Norm Edit: {metr['norm_edit_dis']:.3f} | "
                    f"{format_loss_line(total_loss, step_loss_dict)}"
                )
                # log a few 'close' examples from metric if available
                if len(metr.get("close_examples", [])) > 0:
                    rows = []
                    for (p, t, sim) in metr["close_examples"][:20]:
                        rows.append([epc, global_step, f"{sim:.4f}", p, t])
                    log_close_examples_csv(train_close_csv, rows)

            # backward/update
            total_loss.backward()
            optimizer.step()
            global_step += 1

            # accumulate for epoch means
            update_epoch_accumulators(ep_acc, total_loss, step_loss_dict, metr["acc"])

            # eval mid-epoch
            if (batch_idx + 1) % eval_every_n_batches == 0:
                eval_acc, best_accuracy = _evaluate_once(
                    epc, eval_iter, model, eval_dataloader, decoder, metric,
                    CLOSE_SIM_THRESH, print_every_n_batches, device,
                    run_dir, run_name, history, best_accuracy, logger, eval_close_csv
                )
                model.train()

            # snapshot
            if (batch_idx + 1) % save_every_n_batches == 0:
                weight_filename = run_dir / f"{run_name}_e{epc}_b{batch_idx}.pth"
                logger.info(f"Saving model as {weight_filename}")
                torch.save(model.state_dict(), weight_filename)

        # ---- end of epoch ----
        # epoch means
        total_mean, loss_means, acc_mean = finalize_epoch_means(ep_acc)

        # eval at epoch end
        epoch_eval_acc, best_accuracy = _evaluate_once(
            epc, 0, model, eval_dataloader, decoder, metric,
            CLOSE_SIM_THRESH, print_every_n_batches, device,
            run_dir, run_name, history, best_accuracy, logger, eval_close_csv=eval_close_csv
        )
        model.train()

        # lr
        current_lr = next(iter(optimizer.param_groups))["lr"]

        # epoch history append
        epoch_history["epoch"].append(epc)
        epoch_history["train_acc"].append(acc_mean)
        epoch_history["eval_acc"].append(epoch_eval_acc)
        epoch_history["total_loss"].append(total_mean)
        for k in epoch_history["other_losses"].keys():
            epoch_history["other_losses"][k].append(loss_means.get(k, 0.0))
        epoch_history["lr"].append(current_lr)

        # epoch CSV row
        row = {
            "epoch": epc,
            "train_acc": f"{acc_mean:.6f}",
            "eval_acc": f"{epoch_eval_acc:.6f}",
            "total_loss": f"{total_mean:.6f}",
            "lr": f"{current_lr:.8f}",
        }
        for k in dynamic_loss_names:
            row[k] = f"{loss_means.get(k, 0.0):.6f}"
        write_epoch_row(epoch_csv, epoch_cols, row)

        # save live plots
        save_curves(run_dir, history, vis=vis)
        save_epoch_curves(run_dir, epoch_history, vis=vis)

        # reset epoch accumulators
        ep_acc = init_epoch_accumulators(dynamic_loss_names)

        # end-of-epoch log + scheduler
        logger.info(
            f"{100*'_'}\n"
            f"Epoch {epc}/{num_epochs} | TrainAcc: {acc_mean:.3f}% | EvalAcc: {epoch_eval_acc:.3f}% | "
            f"Total(mean): {total_mean:.3f} | "
            + " | ".join([f"{k}(mean): {loss_means[k]:.3f}" for k in dynamic_loss_names])
            + f"\n{100*'_'}"
        )
        scheduler.step()
