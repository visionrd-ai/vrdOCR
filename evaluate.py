import argparse
import torch
import json
from src.vrd_ocr import vrdOCR
import data
from utils.postprocess import CTCLabelDecode
from src.metric import RecMetric
import logging
import os
from paddle.io import BatchSampler, DataLoader
import paddle.distributed as dist
import numpy as np 
import cv2 



# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate the vrdOCR model.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model weights.")
parser.add_argument("--config_path", type=str, default='data/config.json', help="Path to the configuration file.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation.")
args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = json.load(open(args.config_path, 'r'))

# Load model configuration
backbone_config = {"scale": 0.95, "conv_kxk_num": 4, "freeze_backbone": False}
head_config = {'name': 'MultiHead', 'head_list': [{'CTCHead': {'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True}, 'Head': {'fc_decay': 1e-05}}}, {'NRTRHead': {'nrtr_dim': 384, 'max_text_length': 150}}], 'out_channels_list': {'CTCLabelDecode': 97, 'NRTRLabelDecode': 100}, 'in_channels': 512}
device = "gpu:{}".format(dist.ParallelEnv().dev_id)

# Initialize model and load weights
model = vrdOCR(backbone_config=backbone_config, head_config=head_config).to(args.device)
def load_weights(model, path):
    if path:
        def_dict = torch.load(args.model_path)
        try: 

            model.load_state_dict(def_dict)
        except: 
            mod_dict = {}
            for key, value in def_dict.items():
                mod_dict[key.replace('_orig_mod.', '')] = value 
            model.load_state_dict(mod_dict)

        logger.info(f"Loaded model weights from {path}")
        # filename = os.path.basename(path)
        # parts = filename.replace(args.run_name, '').split('_')
        # epoch, batch = int(parts[0][1:]), int(parts[1][1:].split('.')[0])
        return model
    return model

model = load_weights(model, args.model_path)
model.eval()

# Initialize data loader for evaluation
eval_dataset = data.simple_dataset.SimpleDataSet(config=config, mode='Eval', logger=None, seed=None)
eval_sampler = BatchSampler(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
eval_data_loader = DataLoader(
    dataset=eval_dataset,
    batch_sampler=eval_sampler,
    num_workers=config['Eval']['loader']['num_workers'],
    collate_fn=None,
)

# Initialize decoder and metric
decoder = CTCLabelDecode(character_dict_path='utils/en_dict.txt', use_space_char=True)
metric = RecMetric()
model_name = args.model_path 
model_name = model_name.split('/')[-1].split('.')[0]
fail_dir = f'fail_cases/{model_name}'
os.makedirs(fail_dir, exist_ok=True)
fail_summary = open(os.path.join(fail_dir, 'fail_summary.txt'), 'w')

# Evaluation function
def evaluate_model(model, data_loader):
    logger.info("Starting evaluation...")
    total_accuracy = 0
    total_batches = 0
    all_accs = []
    pred_label = []
    for batch_idx, batch in enumerate(data_loader):
        with torch.no_grad():
            images =torch.tensor(batch['image'].numpy()).cuda()
            labels = batch['label_ctc']

            # Model predictions
            outputs = model(images, labels=None)
            predictions, labels_decoded = decoder(outputs, labels)
            pred_label.append([predictions,labels_decoded])
            batch_metrics = metric([predictions, labels_decoded], print_fail=True)
            if batch_metrics['fail_cases']:
                for fail_case in batch_metrics['fail_cases']:
                    fail_img = images[fail_case]
                    fail_str = predictions[fail_case][0]
                    label_str = labels_decoded[fail_case][0]
                    cv2.imwrite(os.path.join(fail_dir,f'{fail_str}.png'),(fail_img.cpu().numpy()*255).astype(np.uint8).transpose(1,2,0))
                    fail_summary.write(f'{fail_str}\n{label_str}\n\n')
            # Log accuracy for the batch
            accuracy = batch_metrics['acc']
            all_accs.append(accuracy)
            logger.info(f"Batch {batch_idx + 1}/{len(data_loader)} | Batch Accuracy: {accuracy}%")

        total_batches += 1

    overall_accuracy = sum(all_accs) / total_batches
    logger.info(f"Evaluation completed. Overall Accuracy: {overall_accuracy}%")
    fail_summary.close()
    return overall_accuracy

# Run evaluation
overall_accuracy = evaluate_model(model, eval_data_loader)
print(f"Overall Evaluation Accuracy: {overall_accuracy}%")
