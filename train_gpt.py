import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from src.swin_gpt import Swin_GPT2Decoder
from src.dataset import get_swingpt_dataloader
# from src.char_tokenizer import CharacterLevelTokenizer
from transformers import GPT2Tokenizer
import logging
import os
from datetime import datetime
from src.utils import * 

logging.getLogger("albumentations").setLevel(logging.ERROR)
logging.getLogger("albumentations").handlers.clear()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join('exp', timestamp)
os.makedirs(experiment_dir, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_file = os.path.join(experiment_dir, "training_log.txt")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    force=True
)

EPOCHS = 250
BATCH_SIZE = 16
LEARNING_RATE = 3e-5 
EVAL_EVERY_N_EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 2  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = os.path.join(experiment_dir, "best_model.pth")
LATEST_MODEL_PATH = os.path.join(experiment_dir, "latest_model.pth")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"bos_token": "<|beginoftext|>"})
tokenizer.add_special_tokens({"pad_token": "<pad>"})
vocab_size = len(tokenizer)

model = Swin_GPT2Decoder(vocab_size=vocab_size).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)

train_loader = get_swingpt_dataloader(
    'data/IIIT5K/train/annotations.txt', 
    'data/IIIT5K',
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    max_length=128
)
test_loader = get_swingpt_dataloader(
    'data/IIIT5K/test/annotations.txt', 
    'data/IIIT5K',
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    max_length=128
)

logging.basicConfig(filename="training_log.txt", level=logging.INFO, 
                    format="%(asctime)s - %(message)s")
logging.info("Training started...")

best_test_acc = float('-inf')

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

for epoch in range(EPOCHS):

    model.train()
    epoch_train_loss = 0.0
    epoch_train_metrics = {
        'batch_accs':[],
        'batch_cers':[],
    }
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]")):

        images, input_ids, attention_mask = batch
        
        images = images.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        decoder_input_ids = input_ids[:, :-1]
        decoder_target_ids = input_ids[:, 1:]
        
        outputs = model(images, decoder_input_ids)  
        loss = loss_fn(outputs.contiguous().view(-1, outputs.size(-1)), decoder_target_ids.contiguous().view(-1))
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        epoch_train_loss += loss.item()

        pred_ids = outputs.argmax(dim=-1)

        batch_metrics = get_batch_metrics(tokenizer, pred_ids, decoder_target_ids)
        epoch_train_metrics['batch_accs'].append(batch_metrics['acc'])
        epoch_train_metrics['batch_cers'].append(batch_metrics['cer'])

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
    if (i + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_epoch_train_acc = sum(epoch_train_metrics['batch_accs'])/len(epoch_train_metrics['batch_accs'])
    avg_epoch_train_cer = sum(epoch_train_metrics['batch_cers'])/len(epoch_train_metrics['batch_cers'])
    avg_epoch_train_loss = epoch_train_loss / len(train_loader)

    logging.info("-" * 40)
    logging.info(f"TRAIN | Epoch {epoch+1} - Training Avg Loss: {avg_epoch_train_loss:.3f}")
    logging.info(f"TRAIN | Epoch {epoch+1} - Training Avg Accuracy: {avg_epoch_train_acc:.3f}%")
    logging.info(f"TRAIN | Epoch {epoch+1} - Training Avg CER: {avg_epoch_train_cer:.3f}")
    logging.info("-" * 40)
    
    for j in range(min(2, pred_ids.size(0))):  
        pred_text = tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True)
        gt_text = tokenizer.decode(decoder_target_ids[j].cpu().tolist(), skip_special_tokens=True)
        logging.info(f"TRAIN | Epoch {epoch+1} - Sample {j}")
        logging.info(f"TRAIN | Epoch {epoch+1} - Predicted   : {pred_text}")
        logging.info(f"TRAIN | Epoch {epoch+1} - Ground Truth: {gt_text}")

    logging.info("-" * 40)
    logging.info("\n")

    if (epoch + 1) % EVAL_EVERY_N_EPOCHS == 0:
        val_accuracy, val_cer, samples = beam_search_evaluate(epoch+1, model, test_loader, tokenizer, 'cuda', beam_size=5)

        logging.info("-" * 40)
        logging.info(f"BEAM SEARCH | Epoch {epoch+1} | Eval Avg Accuracy: {val_accuracy:.3f}%")
        logging.info(f"BEAM SEARCH | Epoch {epoch+1} | Eval Avg CER: {val_cer:.3f}")
        logging.info("-" * 40)
        
        for j in range(min(2, samples['pred_ids'].size(0))):
            pred_text = tokenizer.decode(samples['pred_ids'][j].cpu().tolist(), skip_special_tokens=True)
            gt_text = tokenizer.decode(samples['target_ids'][j].cpu().tolist(), skip_special_tokens=True)
            logging.info(f"BEAM SEARCH | Epoch {epoch+1} - Sample {j}")
            logging.info(f"BEAM SEARCH | Epoch {epoch+1} - Predicted   : {pred_text}")
            logging.info(f"BEAM SEARCH | Epoch {epoch+1} - Ground Truth: {gt_text}")
        
        logging.info("-" * 40)
        logging.info("\n")

        scheduler.step(val_accuracy)
        torch.save(model.state_dict(), LATEST_MODEL_PATH)
        logging.info(f"INFO | Epoch {epoch+1} - LATEST model updated.")
        if val_accuracy > best_test_acc:
            best_test_acc = val_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logging.info(f"INFO | Epoch {epoch+1} - BEST model updated.")

logging.info("Training complete!")
