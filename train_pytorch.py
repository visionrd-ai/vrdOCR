import argparse
import logging
import json
import os
import datetime
import data.ocr_dataset
from src.vrd_ocr import vrdOCR
import data
import torch
# from paddle.io import BatchSampler, DataLoader
from src.multi_loss import MultiLoss
import paddle.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.postprocess import CTCLabelDecode
from src.metric import RecMetric
import editdistance
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser(description="Train or evaluate the vrdOCR model.")
parser.add_argument("--model_path", type=str, help="Path to the pretrained model weights.")
parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    help="Custom name for the training run.")
parser.add_argument("--freeze_backbone", type=bool, help="Freeze backbone or not")

args = parser.parse_args()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.run_name:
    run_name = f"{args.run_name}_{timestamp}"
else:
    run_name = timestamp
run_dir = f"runs/{run_name}"
os.makedirs(run_dir, exist_ok=True)

config_path = 'data/config.json'
config = json.load(open(config_path, 'r'))
run_config_path = os.path.join(run_dir, 'config.json')
with open(run_config_path, 'w') as config_file:
    json.dump(config, config_file, indent=4)

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

backbone_config = {"scale": 0.95, "conv_kxk_num": 4, "freeze_backbone":args.freeze_backbone}
head_config = {'name': 'MultiHead', 'head_list': [{'CTCHead': {'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True}, 'Head': {'fc_decay': 1e-05}}}, {'NRTRHead': {'nrtr_dim': 384, 'max_text_length': 150}}], 'out_channels_list': {'CTCLabelDecode': 97, 'NRTRLabelDecode': 100}, 'in_channels': 480}

model = vrdOCR(backbone_config=backbone_config, head_config=head_config).cuda()
logger.info("Compiling model...")
# model = torch.compile(model)
logger.info("Compilation complete!")

def load_weights(model, path):
    if path:
        def_dict = torch.load(args.model_path)
        try: 
            model.load_state_dict(def_dict, strict=False)
        except: 
            mod_dict = {}
            for key, value in def_dict.items():
                mod_dict['_orig_mod.'+key] = value 

            model.load_state_dict(mod_dict, strict=False)

        logger.info(f"Loaded model weights from {path}")
        # filename = os.path.basename(path)
        # parts = filename.replace(args.run_name, '').split('_')
        # epoch, batch = int(parts[0][1:]), int(parts[1][1:].split('.')[0])
        return model
    return model

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.Affine(scale=(1.0, 1.0), rotate=(-0.005, -0.005), translate_percent=(-0.005, 0.005), shear=(-0.005, -0.005), p=0.6),
    A.ImageCompression(quality_lower=30, quality_upper=70, p=0.4),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.Resize(width=640, height=64),
    ToTensorV2()
])
#640->160, 320->80, pool to 25 

eval_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.Resize(width=640, height=64),
    ToTensorV2()
])

model = load_weights(model, args.model_path)
start_epoch, end_epoch=0,0
decoder = CTCLabelDecode(character_dict_path='utils/en_dict.txt', use_space_char=True)
batch_size = 4
dataset = data.ocr_dataset.OCRDataset(input_dir='/home/amur/Amur/ForgeryDetectionV1.2/vrdOCR/dataset/mixed_new', split='train', transforms=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

eval_dataset = data.ocr_dataset.OCRDataset(input_dir='/home/amur/Amur/ForgeryDetectionV1.2/vrdOCR/dataset/mixed_new', split='val', transforms=eval_transform)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

loss_config = {'loss_config_list': [{'CTCLoss': None}, {'NRTRLoss': None}]}
loss_fn = MultiLoss(**loss_config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

device = "gpu:{}".format(dist.ParallelEnv().dev_id)
eval_every_n_batches = config.get('eval_every_n_batches', 296)
save_every_n_batches = config.get('save_every_n_batches', 500)
num_epochs = config.get('num_epochs', 200)
print_every_n_batches = config.get('print_every_n_batches', 10)
metric = RecMetric()
best_accuracy = 0.0  # Initialize the best accuracy variable

def evaluate(epc, eval_iter, model, eval_loader):
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

for epc in range(start_epoch, num_epochs):
    epoch_accuracies = []
    eval_iter = 0
    for batch_idx, batch in enumerate(dataloader):
        images = batch[0].cuda()
        labels = batch[1:]
        labels = [label.cuda() for label in labels]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outs = model(images, labels)
            losses = loss_fn(outs, batch)
        ctc_loss = losses['CTCLoss']
        nrtr_loss = losses['NRTRLoss']

        total_loss = ctc_loss + nrtr_loss

        preds, labels = decoder(outs['ctc'], batch[1])
        accuracies = metric([preds, labels])
        if batch_idx % print_every_n_batches == 0: 
            logger.info(f"Epoch {epc}/{num_epochs}, Batch {batch_idx}/{len(dataloader)} | Train Accuracy: {accuracies['acc']:.3f}% | Norm Edit Distance: {accuracies['norm_edit_dis']:.3f} | Total Loss: {total_loss.item():.3f} | CTC Loss: {ctc_loss.item():.3f} | NRTR Loss: {nrtr_loss.item():.3f}")#logger.info(f"Epoch {epc}/{num_epochs}, Batch {batch_idx}/{len(dataloader)} | Train Accuracy: {accuracies['acc']}% | Norm Edit Distance: {accuracies['norm_edit_dis']} | Total Loss: {total_loss.item()} | CTC Loss: {ctc_loss.item()} | NRTR Loss: {nrtr_loss.item()}")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % eval_every_n_batches == 0:
            evaluate(epc, eval_iter, model, eval_dataloader)
            model.train()

        if (batch_idx+1) % save_every_n_batches == 0:
            weight_filename = os.path.join(run_dir, f'{args.run_name}_e{epc}_b{batch_idx}.pth')
            logger.info(f"Saving model as {weight_filename}")
            # torch.save(model.state_dict(), weight_filename)
 
        epoch_accuracies.append(accuracies['acc'])
    
    # print(f"{100*'_'}\nEpoch {epc}/{num_epochs} | Accuracy: {sum(epoch_accuracies)/len(epoch_accuracies)}%\n{100*'_'}")
    logger.info(f"Epoch {epc} Accuracy: {sum(epoch_accuracies) / len(epoch_accuracies):.3f}%")
    scheduler.step()