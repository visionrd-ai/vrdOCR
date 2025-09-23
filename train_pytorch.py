import argparse
import logging
import json
import os
import datetime
import csv
import data_old.toyota_dataset
from src.vrd_ocr import vrdOCR
from utils_old.save_curves import save_curves, save_epoch_curves
from rapidfuzz.distance import Levenshtein
import data_old
import torch
# from paddle.io import BatchSampler, DataLoader
from src.multi_loss import MultiLoss
import paddle.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils_old.postprocess import CTCLabelDecode
from src.metric import RecMetric
import editdistance
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser(description="Train or evaluate the vrdOCR model.")
parser.add_argument("--model_path", type=str, help="Path to the pretrained model weights.")
parser.add_argument("--dataset_train", default="/home/multi-gpu/Talal/vrdOCR/data/toyota_dataset/labels_train.txt", type=str, help="Path to the dataset directory.")
parser.add_argument("--dataset_val",default="/home/multi-gpu/Talal/vrdOCR/data/toyota_dataset/labels_val.txt", type=str, help="Path to the validation dataset directory.")
parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    help="Custom name for the training run.")
parser.add_argument("--freeze_backbone", type=bool, help="Freeze backbone or not")

args = parser.parse_args()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.run_name:
    run_name = f"{args.run_name}_{timestamp}"
else:
    run_name = timestamp
run_dir = f"/home/multi-gpu/Talal/vrdOCR/runs/new_updated_{run_name}"
os.makedirs(run_dir, exist_ok=True)



config_path = r'/home/multi-gpu/Talal/vrdOCR/data_old/config.json'
config = json.load(open(config_path, 'r'))
run_config_path = os.path.join(run_dir, 'config.json')
with open(run_config_path, 'w') as config_file:
    json.dump(config, config_file, indent=4)
    
CLOSE_SIM_THRESH = float(config.get("close_similarity_thresh", 0.85))

train_close_csv = os.path.join(run_dir, "train_close_examples.csv")
eval_close_csv  = os.path.join(run_dir, "eval_close_examples.csv")

# Initialize CSVs with headers
with open(train_close_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch","step","similarity","pred","target"])
with open(eval_close_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch","batch","similarity","pred","target"])

log_filename = os.path.join(run_dir, f'training_log_{args.run_name}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

backbone_config = {"scale": 0.95, "conv_kxk_num": 4, "freeze_backbone":False}
head_config = {'name': 'MultiHead', 'head_list': [{'CTCHead': {'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True}, 'Head': {'fc_decay': 1e-05}}}, {'NRTRHead': {'nrtr_dim': 384, 'max_text_length': 150}}], 'out_channels_list': {'CTCLabelDecode': 97, 'NRTRLabelDecode': 100}, 'in_channels': 480}
model = vrdOCR(backbone_config=backbone_config, head_config=head_config).cuda()
model.load_state_dict(torch.load(r'/home/multi-gpu/Talal/vrdOCR/weights/HRNet_deploy_augd2_best_model.pth'))
logger.info("Compiling model...")
# model = torch.compile(model)
logger.info("Compilation complete!")

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.Affine(scale=(1.0, 1.0), rotate=(-0.005, -0.005), translate_percent=(-0.005, 0.005), shear=(-0.005, -0.005), p=0.6),
    A.ImageCompression(quality_lower=30, quality_upper=70, p=0.4),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.Resize(width=320, height=32),
    ToTensorV2()
])
#640->160, 320->80, pool to 25 

eval_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.Resize(width=320, height=32),
    ToTensorV2()
])

# model = load_weights(model, args.model_path)
start_epoch, end_epoch=0,1500
decoder = CTCLabelDecode(character_dict_path="./data/en_dict.txt", use_space_char=True)
# batch_size = 128
# dataset = data.ocr_dataset.OCRDataset(input_dir=args.dataset_train, split='train', transforms=transform)

batch_size = 2
dataset = data_old.toyota_dataset.ToyotaDataset(
    dir="/home/multi-gpu/Talal/vrdOCR/data/toyota_dataset", 
    file="labels_train.txt", 
    max_len=150, 
    dict_path="./data/en_dict.txt",
    split="train", transforms=transform
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=8
)

eval_dataset = data_old.toyota_dataset.ToyotaDataset(
    dir="/home/multi-gpu/Talal/vrdOCR/data/toyota_dataset", 
    file="labels_val.txt", 
    max_len=150, 
    dict_path="./data/en_dict.txt",
    split="train", transforms=eval_transform
)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8
)

loss_config = {'loss_config_list': [{'CTCLoss': None}, {'NRTRLoss': None}]}
loss_fn = MultiLoss(**loss_config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

device = "gpu:{}".format(dist.ParallelEnv().dev_id)
eval_every_n_batches = config.get('eval_every_n_batches', len(dataset) // batch_size - 1)
save_every_n_batches = config.get('save_every_n_batches', 500)
num_epochs = config.get('num_epochs', end_epoch)
print_every_n_batches = config.get('print_every_n_batches', 10)
metric = RecMetric(logger)
best_accuracy = 0.0  # Initialize the best accuracy variable

history = {
    "steps": [],        # train step index
    "train_acc": [],    # per-batch
    "ctc_loss": [],
    "nrtr_loss": [],
    "total_loss": [],
    "eval_steps": [],   # step index when eval ran
    "eval_acc": []      # overall eval accuracy recorded each eval
}

epoch_history = {
    "epoch": [],
    "train_acc": [],
    "eval_acc": [],
    "total_loss": [],
    "ctc_loss": [],
    "nrtr_loss": [],
    "lr": [],
}
epoch_csv = os.path.join(run_dir, "epoch_metrics.csv")
with open(epoch_csv, "w", newline="") as f:
    csv.writer(f).writerow(["epoch","train_acc","eval_acc","total_loss","ctc_loss","nrtr_loss","lr"])

global_step = 0


def evaluate_old(epc, eval_iter, model, eval_loader):
    global best_accuracy
    model.eval()
    eval_iter += 1
    eval_accs = []
    for eval_idx, eval_batch in enumerate(eval_loader):
        eval_images = eval_batch[0].cuda()
        eval_outs = model(eval_images, labels=None)
        eval_predictions, eval_labels_decoded = decoder(eval_outs, eval_batch[1])
        
        eval_metr = metric([eval_predictions, eval_labels_decoded])
        eval_accs.append(eval_metr['acc'])
        if eval_idx % print_every_n_batches == 0:
            logger.info(f"Epoch {epc}/{num_epochs}, Batch {eval_idx}/{len(eval_loader)} | Eval Accuracy: {eval_metr['acc']:.3f}% | Norm Edit Distance: {eval_metr['norm_edit_dis']:.3f}")
            logger.info(f"Epoch {epc}/{num_epochs}, Batch {eval_idx}/{len(eval_loader)} | Eval Pred: '{eval_predictions[0][0]}'\tLabel: '{eval_labels_decoded[0][0]}'")

    overall_accuracy = sum(eval_accs) / len(eval_accs)
    logger.info(f"Epoch {epc}/{num_epochs} | Overall Eval Iter#{eval_iter} Accuracy: {overall_accuracy:.3f}%")    

    if overall_accuracy > best_accuracy:
        best_accuracy = overall_accuracy
        best_model_path = os.path.join(run_dir, f'{args.run_name}_best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"New best model saved with accuracy: {best_accuracy}% at {best_model_path}")
        
def evaluate(epc, eval_iter, model, eval_loader):
    global best_accuracy, history, global_step
    model.eval()
    eval_iter += 1
    eval_accs = []

    for eval_idx, eval_batch in enumerate(eval_loader):
        eval_images = eval_batch[0].cuda()
        eval_outs = model(eval_images, labels=None)
        eval_predictions, eval_labels_decoded = decoder(eval_outs, eval_batch[1])

        # metric returns dict: {'acc', 'norm_edit_dis', ...}
        eval_metr = metric([eval_predictions, eval_labels_decoded])
        eval_accs.append(eval_metr['acc'])

        # --- NEW: save eval "close" predictions via similarity ---
        # eval_predictions / eval_labels_decoded are lists like [[str], [str], ...]
        preds = [p[0] for p in eval_predictions]
        gts   = [g[0] for g in eval_labels_decoded]
        rows  = []
        for p, t in zip(preds, gts):
            sim = Levenshtein.normalized_similarity(p.strip(), t.strip())
            if sim >= CLOSE_SIM_THRESH:
                rows.append([epc, eval_idx, f"{sim:.4f}", p, t])
        if rows:
            with open(eval_close_csv, "a", newline="") as f:
                w = csv.writer(f); w.writerows(rows)

        if eval_idx % print_every_n_batches == 0:
            logger.info(
                f"Epoch {epc}/{num_epochs}, Batch {eval_idx}/{len(eval_loader)} | "
                f"Eval Accuracy: {eval_metr['acc']:.3f}% | Norm Edit Distance: {eval_metr['norm_edit_dis']:.3f}"
            )
            logger.info(
                f"Epoch {epc}/{num_epochs}, Batch {eval_idx}/{len(eval_loader)} | "
                f"Eval Pred: '{eval_predictions[0][0]}'\tLabel: '{eval_labels_decoded[0][0]}'"
            )

    overall_accuracy = sum(eval_accs) / len(eval_accs)
    logger.info(f"Epoch {epc}/{num_epochs} | Overall Eval Iter#{eval_iter} Accuracy: {overall_accuracy:.3f}%")

    # --- NEW: record eval point aligned with current global_step ---
    history["eval_steps"].append(global_step)
    history["eval_acc"].append(overall_accuracy)

    if overall_accuracy > best_accuracy:
        best_accuracy = overall_accuracy
        best_model_path = os.path.join(run_dir, f'{args.run_name}_best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"New best model saved with accuracy: {best_accuracy}% at {best_model_path}")
    return overall_accuracy

best_accuracy = 0.0  # Reset best accuracy for the new run
for epc in range(start_epoch, end_epoch):
    epoch_accuracies = []
    eval_iter = 0
    
    # --- NEW: accumulators for epoch means ---
    ep_sum_total = 0.0
    ep_sum_ctc   = 0.0
    ep_sum_nrtr  = 0.0
    ep_sum_acc   = 0.0
    ep_count     = 0
    
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        images = batch[0].cuda()
        labels = batch[1:]
        labels = [label.cuda() for label in labels]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outs = model(images, labels)
            losses = loss_fn(outs, batch)
        ctc_loss = losses['CTCLoss']
        nrtr_loss = losses['NRTRLoss']

        total_loss = ctc_loss + nrtr_loss
        import pdb; pdb.set_trace()
        preds, labels_dec = decoder(outs['ctc'], batch[1])
        accuracies = metric([preds, labels_dec])

        # --- NEW: record per-batch curves
        history["steps"].append(global_step)
        history["train_acc"].append(accuracies["acc"])
        history["ctc_loss"].append(float(ctc_loss.detach().item()))
        history["nrtr_loss"].append(float(nrtr_loss.detach().item()))
        history["total_loss"].append(float(total_loss.detach().item()))
        
        # --- NEW: accumulate for epoch means ---
        ep_sum_total += float(total_loss.detach().item())
        ep_sum_ctc   += float(ctc_loss.detach().item())
        ep_sum_nrtr  += float(nrtr_loss.detach().item())
        ep_sum_acc   += float(accuracies["acc"])
        ep_count     += 1

        if batch_idx % print_every_n_batches == 0: 
            logger.info(
                f"Epoch {epc}/{end_epoch}, Batch {batch_idx}/{len(dataloader)} | "
                f"Train Accuracy: {accuracies['acc']:.3f}% | Norm Edit Distance: {accuracies['norm_edit_dis']:.3f} | "
                f"Total Loss: {total_loss.item():.3f} | CTC Loss: {ctc_loss.item():.3f} | NRTR Loss: {nrtr_loss.item():.3f}"
            )
            # --- NEW: persist a few "close" examples during TRAIN
            if len(accuracies.get('close_examples', [])) > 0:
                # close_examples: list of (pred, target, similarity)
                rows = []
                for (p, t, sim) in accuracies['close_examples'][:20]:
                    rows.append([epc, global_step, f"{sim:.4f}", p, t])
                with open(train_close_csv, "a", newline="") as f:
                    w = csv.writer(f); w.writerows(rows)
                logger.info(f"Close Examples (pred, target, similarity): {accuracies['close_examples'][:5]}")

        total_loss.backward()
        optimizer.step()

        global_step += 1  # --- NEW
        

        if (batch_idx + 1) % eval_every_n_batches == 0:
            evaluate(epc, eval_iter, model, eval_dataloader)
            model.train()

        if (batch_idx+1) % save_every_n_batches == 0:
            weight_filename = os.path.join(run_dir, f'{args.run_name}_e{epc}_b{batch_idx}.pth')
            logger.info(f"Saving model as {weight_filename}")
            # torch.save(model.state_dict(), weight_filename)
 
        epoch_accuracies.append(accuracies['acc'])

    train_acc_epoch  = (ep_sum_acc / max(1, ep_count))
    total_loss_epoch = (ep_sum_total / max(1, ep_count))
    ctc_loss_epoch   = (ep_sum_ctc / max(1, ep_count))
    nrtr_loss_epoch  = (ep_sum_nrtr / max(1, ep_count))

    # Optional: eval at epoch end to get epoch-level eval acc
    epoch_eval_acc = evaluate(epc, 0, model, eval_dataloader)  # returns overall eval %
    model.train()
    
    current_lr = next(iter(optimizer.param_groups))["lr"]
    epoch_history["epoch"].append(epc)
    epoch_history["train_acc"].append(train_acc_epoch)
    epoch_history["eval_acc"].append(epoch_eval_acc)
    epoch_history["total_loss"].append(total_loss_epoch)
    epoch_history["ctc_loss"].append(ctc_loss_epoch)
    epoch_history["nrtr_loss"].append(nrtr_loss_epoch)
    epoch_history["lr"].append(current_lr)

    # --- NEW: append CSV row ---
    with open(epoch_csv, "a", newline="") as f:
        csv.writer(f).writerow([epc, f"{train_acc_epoch:.6f}", f"{epoch_eval_acc:.6f}",
                                f"{total_loss_epoch:.6f}", f"{ctc_loss_epoch:.6f}", f"{nrtr_loss_epoch:.6f}",
                                f"{current_lr:.8f}"])

    accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
    
    if best_accuracy < accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(run_dir, f'{args.run_name}_best_model.pth'))

    logger.info(f"{100*'_'}\nEpoch {epc}/{end_epoch} | Accuracy: {accuracy:.3f}%\n{100*'_'}")
    logger.info(f"Epoch {epc} Accuracy: {accuracy:.3f}%")

    # --- NEW: update LR scheduler and refresh plots each epoch
    scheduler.step()
    save_curves(run_dir, history)   # writes loss_curve.png, accuracy_curve.png, history.json
    save_epoch_curves(run_dir, epoch_history)  # writes loss_curve_epoch.png, accuracy_curve_epoch.png, epoch_history.json