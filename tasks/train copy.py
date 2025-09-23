# train.py
import csv, datetime, logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

# keep your existing imports as-is
from rapidfuzz.distance import Levenshtein
import data_old.toyota_dataset

from model.registeries import build_postprocessing

# from utils_old.save_curves import save_curves, save_epoch_curves
from utils.curves import save_curves, save_epoch_curves
from src.metric import RecMetric


def yaml_get(dct, path, default=None):
    """Tiny helper to safely read nested YAML dicts."""
    cur = dct
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def train(cfg, model, run_dir, run_name, device, logger, vis= False):
    DECODER = cfg.get("DECODER", {})
    decoder = build_postprocessing(DECODER)
    if decoder is None:
        raise ValueError("DECODER missing in YAML. Please add your Decoder block.")

    # ----------------------------
    # Training knobs from YAML
    # ----------------------------
    # We’ll look under TRAIN: { num_epochs, eval_every_n_batches, save_every_n_batches, print_every_n_batches, close_similarity_thresh, lr, step_size, gamma }
    TRAIN = cfg.get("TRAIN", {})
    num_epochs = int(TRAIN.get("num_epochs", 1500))
    eval_every_n_batches = TRAIN.get("eval_every_n_batches", None)  # if None, we’ll compute below
    save_every_n_batches = int(TRAIN.get("save_every_n_batches", 500))
    print_every_n_batches = int(TRAIN.get("print_every_n_batches", 10))
    CLOSE_SIM_THRESH = float(TRAIN.get("close_similarity_thresh", 0.85))
    lr = float(TRAIN.get("lr", 1e-4))
    step_size = int(TRAIN.get("step_size", 10))
    gamma = float(TRAIN.get("gamma", 0.1))

    # ----------------------------
    # Transforms (kept as-is)
    # ----------------------------
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.Affine(scale=(1.0, 1.0), rotate=(-0.005, -0.005),
                 translate_percent=(-0.005, 0.005), shear=(-0.005, -0.005), p=0.6),
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.4),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.Resize(width=320, height=32),
        ToTensorV2()
    ])

    eval_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.Resize(width=640, height=64),
        ToTensorV2()
    ])

    # ----------------------------
    # Datasets / Dataloaders (as-is)
    # ----------------------------
    
    DATASET = cfg.get("DATASET", {})

    batch_size = int(cfg.get("batch_size", 2))

    dataset = data_old.toyota_dataset.ToyotaDataset(
        input_dir=DATASET.get("train", ""), split="train", transforms=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    eval_dataset = data_old.toyota_dataset.ToyotaDataset(
        input_dir=DATASET.get("val", ""), split="val", transforms=eval_transform
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    if eval_every_n_batches is None:
        eval_every_n_batches = len(dataset) // batch_size - 1

    # ----------------------------
    # Loss from YAML via registry
    # ----------------------------
    # We’ll pull loss config from MODEL.loss in YAML and build via LOSSES registry in main.
    # main.py passes loss_fn in model? We keep an external loss, like your original loop.
    from model.registeries import build_loss
    loss_cfg = yaml_get(cfg, "MODEL.loss", None)
    if loss_cfg is None:
        raise ValueError("MODEL.loss missing in YAML. Please add your MultiLoss block.")
    loss_fn = build_loss(loss_cfg)

    # ----------------------------
    # Optim / Scheduler
    # ----------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ----------------------------
    # Metrics / history / CSVs
    # ----------------------------
    metric = RecMetric(logger=logger)

    best_accuracy = 0.0
    history = {
    "steps": [],
    "total_loss": [],
    "other_losses": {},  # any names you like
    "train_acc": [],
    "eval_steps": [],
    "eval_acc": []
    }

    epoch_history = {
    "epoch": [],
    "total_loss": [],
    "other_losses": {},  # optional
    "train_acc": [],
    "eval_acc": []
    }
    epoch_csv = run_dir / "epoch_metrics.csv"
    with open(epoch_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_acc", "eval_acc", "total_loss", "ctc_loss", "nrtr_loss", "lr"])

    train_close_csv = run_dir / "train_close_examples.csv"
    eval_close_csv = run_dir / "eval_close_examples.csv"
    with open(train_close_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "similarity", "pred", "target"])
    with open(eval_close_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "batch", "similarity", "pred", "target"])

    device = torch.device(device)
    model.to(device)
    model.train()

    global_step = 0

    def evaluate(epc, eval_iter, model, eval_loader):
        nonlocal best_accuracy, history, global_step
        model.eval()
        eval_iter += 1
        eval_accs = []

        for eval_idx, eval_batch in enumerate(eval_loader):
            eval_images = eval_batch[0].to(device)
            eval_outs = model(eval_images, labels=None)
            eval_predictions, eval_labels_decoded = decoder(eval_outs, eval_batch[1])

            eval_metr = metric([eval_predictions, eval_labels_decoded])
            eval_accs.append(eval_metr["acc"])

            # Save "close" examples
            preds = [p[0] for p in eval_predictions]
            gts = [g[0] for g in eval_labels_decoded]
            rows = []
            for p, t in zip(preds, gts):
                sim = Levenshtein.normalized_similarity(p.strip(), t.strip())
                if sim >= CLOSE_SIM_THRESH:
                    rows.append([epc, eval_idx, f"{sim:.4f}", p, t])
            if rows:
                with open(eval_close_csv, "a", newline="") as f:
                    csv.writer(f).writerows(rows)

            if eval_idx % print_every_n_batches == 0:
                logging.info(
                    f"Epoch {epc}/{num_epochs}, Batch {eval_idx}/{len(eval_loader)} | "
                    f"Eval Accuracy: {eval_metr['acc']:.3f}% | Norm Edit Distance: {eval_metr['norm_edit_dis']:.3f}"
                )
                logging.info(
                    f"Epoch {epc}/{num_epochs}, Batch {eval_idx}/{len(eval_loader)} | "
                    f"Eval Pred: '{eval_predictions[0][0]}'\tLabel: '{eval_labels_decoded[0][0]}'"
                )

        overall_accuracy = sum(eval_accs) / max(1, len(eval_accs))
        logging.info(f"Epoch {epc}/{num_epochs} | Overall Eval Iter#{eval_iter} Accuracy: {overall_accuracy:.3f}%")

        # record eval point aligned with current global_step
        history["eval_steps"].append(global_step)
        history["eval_acc"].append(overall_accuracy)

        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
            best_model_path = run_dir / f"{run_name}_best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with accuracy: {best_accuracy}% at {best_model_path}")
        return overall_accuracy

    # ----------------------------
    # Training loop (kept same)
    # ----------------------------
    start_epoch, end_epoch = 0, num_epochs

    for epc in range(start_epoch, end_epoch):
        epoch_accuracies = []
        eval_iter = 0

        ep_sum_total = 0.0
        ep_sum_ctc = 0.0
        ep_sum_nrtr = 0.0
        ep_sum_acc = 0.0
        ep_count = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            images = batch[0].to(device)
            labels = [label.to(device) for label in batch[1:]]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                outs = model(images, labels)
                losses = loss_fn(outs, batch)

            ctc_loss = losses["CTCLoss"]
            nrtr_loss = losses["NRTRLoss"]
            total_loss = losses["loss"]

            preds, labels_dec = decoder(outs["ctc"], batch[1])
            accuracies = metric([preds, labels_dec])

            # record batch curves
            history["steps"].append(global_step)
            history["train_acc"].append(accuracies["acc"])
            history["ctc_loss"].append(float(ctc_loss.detach().item()))
            history["nrtr_loss"].append(float(nrtr_loss.detach().item()))
            history["total_loss"].append(float(total_loss.detach().item()))

            ep_sum_total += float(total_loss.detach().item())
            ep_sum_ctc += float(ctc_loss.detach().item())
            ep_sum_nrtr += float(nrtr_loss.detach().item())
            ep_sum_acc += float(accuracies["acc"])
            ep_count += 1

            if batch_idx % print_every_n_batches == 0:
                logging.info(
                    f"Epoch {epc}/{end_epoch}, Batch {batch_idx}/{len(dataloader)} | "
                    f"Train Accuracy: {accuracies['acc']:.3f}% | Norm Edit Distance: {accuracies['norm_edit_dis']:.3f} | "
                    f"Total Loss: {total_loss.item():.3f} | CTC Loss: {ctc_loss.item():.3f} | NRTR Loss: {nrtr_loss.item():.3f}"
                )
                # persist a few "close" examples during TRAIN
                if len(accuracies.get("close_examples", [])) > 0:
                    rows = []
                    for (p, t, sim) in accuracies["close_examples"][:20]:
                        rows.append([epc, global_step, f"{sim:.4f}", p, t])
                    with open(train_close_csv, "a", newline="") as f:
                        csv.writer(f).writerows(rows)
                    logging.info(f"Close Examples (pred, target, similarity): {accuracies['close_examples'][:5]}")

            total_loss.backward()
            optimizer.step()

            global_step += 1

            if (batch_idx + 1) % eval_every_n_batches == 0:
                evaluate(epc, eval_iter, model, eval_dataloader)
                model.train()

            if (batch_idx + 1) % save_every_n_batches == 0:
                weight_filename = run_dir / f"{run_name}_e{epc}_b{batch_idx}.pth"
                logging.info(f"Saving model as {weight_filename}")
                torch.save(model.state_dict(), weight_filename)

            epoch_accuracies.append(accuracies["acc"])

        # epoch means
        train_acc_epoch = (ep_sum_acc / max(1, ep_count))
        total_loss_epoch = (ep_sum_total / max(1, ep_count))
        ctc_loss_epoch = (ep_sum_ctc / max(1, ep_count))
        nrtr_loss_epoch = (ep_sum_nrtr / max(1, ep_count))

        # eval at epoch end
        epoch_eval_acc = evaluate(epc, 0, model, eval_dataloader)
        model.train()

        current_lr = next(iter(optimizer.param_groups))["lr"]
        epoch_history["epoch"].append(epc)
        epoch_history["train_acc"].append(train_acc_epoch)
        epoch_history["eval_acc"].append(epoch_eval_acc)
        epoch_history["total_loss"].append(total_loss_epoch)
        epoch_history["ctc_loss"].append(ctc_loss_epoch)
        epoch_history["nrtr_loss"].append(nrtr_loss_epoch)
        epoch_history["lr"].append(current_lr)

        with open(epoch_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epc, f"{train_acc_epoch:.6f}", f"{epoch_eval_acc:.6f}",
                f"{total_loss_epoch:.6f}", f"{ctc_loss_epoch:.6f}", f"{nrtr_loss_epoch:.6f}",
                f"{current_lr:.8f}"
            ])

        accuracy = sum(epoch_accuracies) / max(1, len(epoch_accuracies))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), run_dir / f"{run_name}_best_model.pth")

        logging.info(f"{100*'_'}\nEpoch {epc}/{end_epoch} | Accuracy: {accuracy:.3f}%\n{100*'_'}")
        logging.info(f"Epoch {epc} Accuracy: {accuracy:.3f}%")

        scheduler.step()
        save_curves(run_dir, history, vis=vis)
        save_epoch_curves(run_dir, epoch_history, vis=vis)

