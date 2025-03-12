import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from src.model import ViT_TransformerDecoder
from src.dataset import get_dataloader
from src.char_tokenizer import CharacterLevelTokenizer
import Levenshtein 
import logging
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join('exp', timestamp)
os.makedirs(experiment_dir, exist_ok=True)
log_file = os.path.join(experiment_dir, "training_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format="%(asctime)s - %(message)s")

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
GRADIENT_ACCUMULATION_STEPS = 2  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = os.path.join(experiment_dir, "best_model.pth")

tokenizer = CharacterLevelTokenizer()
vocab_size = len(tokenizer)

model = ViT_TransformerDecoder(vocab_size=vocab_size).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

train_loader = get_dataloader(
    'data/IIIT5K/train/annotations.txt', 
    'data/IIIT5K',
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    max_length=128
)
test_loader = get_dataloader(
    'data/IIIT5K/test/annotations.txt', 
    'data/IIIT5K',
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    max_length=128
)

logging.basicConfig(filename="training_log.txt", level=logging.INFO, 
                    format="%(asctime)s - %(message)s")
logging.info("Training started...")

best_test_loss = None

def get_batch_metrics(pred_ids, decoder_target_ids):
    # if isinstance(pred_ids, list):

    #     [tokenizer.decode(pred_ids[j], skip_special_tokens=True) for j in range(len(pred_ids))]
    #     pred_texts = [tokenizer.decode(pred_ids[j], skip_special_tokens=True) for j in range(len(pred_ids))]
    # else:
    pred_texts = [tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True) for j in  range(pred_ids.shape[0])]
    
    gt_texts = [tokenizer.decode(decoder_target_ids[j].cpu().tolist(), skip_special_tokens=True) for j in range(pred_ids.shape[0])]

    correct = sum([1 for pred_text, gt_text in zip(pred_texts, gt_texts) if pred_text == gt_text])
    total = len(pred_texts)
    accuracy = correct / total * 100
    
    dists = [calculate_cer(pred_text, gt_text) for pred_text, gt_text in zip(pred_texts, gt_texts)]
    cer = sum(dists) / total

    return {'acc':accuracy, 'cer':cer}

def calculate_cer(pred_text, gt_text):
    return Levenshtein.distance(pred_text, gt_text) / len(gt_text) if len(gt_text) > 0 else 0


def beam_search_evaluate(epoch, model, test_loader, tokenizer, device, beam_size=5):
    """Evaluate the model using beam search decoding."""
    model.eval()
    epoch_test_metrics = {
        'batch_accs': [],
        'batch_cers': [],
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Epoch {epoch} [Beam Search Evaluation]")):
            images, input_ids, _ = batch
            images = images.to(device)
            pred_ids = model.generate(images, 
                                      start_token_id=tokenizer.convert_tokens_to_ids(tokenizer.bos_token), 
                                      end_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
                                      pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
                                      max_length=128,
                                      beam_size=beam_size)
            decoder_target_ids = input_ids[:, 1:].to(device)
            batch_metrics = get_batch_metrics(pred_ids, decoder_target_ids)
            epoch_test_metrics['batch_accs'].append(batch_metrics['acc'])
            epoch_test_metrics['batch_cers'].append(batch_metrics['cer'])
    
    avg_epoch_test_acc = sum(epoch_test_metrics['batch_accs']) / len(epoch_test_metrics['batch_accs'])
    avg_epoch_test_cer = sum(epoch_test_metrics['batch_cers']) / len(epoch_test_metrics['batch_cers'])
    
    logging.info("-" * 40)
    logging.info(f"BEAM SEARCH | Epoch {epoch+1} | Eval Avg Accuracy: {avg_epoch_test_acc:.3f}%")
    logging.info(f"BEAM SEARCH | Epoch {epoch+1} | Eval Avg CER: {avg_epoch_test_cer:.3f}")
    logging.info("-" * 40)
    
    for j in range(min(2, pred_ids.size(0))):
        pred_text = tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True)
        gt_text = tokenizer.decode(decoder_target_ids[j].cpu().tolist(), skip_special_tokens=True)
        logging.info(f"BEAM SEARCH | Epoch {epoch+1} - Sample {j}")
        logging.info(f"BEAM SEARCH | Epoch {epoch+1} - Predicted   : {pred_text}")
        logging.info(f"BEAM SEARCH | Epoch {epoch+1} - Ground Truth: {gt_text}")
    
    logging.info("-" * 40)
    logging.info("\n")
    
    return avg_epoch_test_acc, avg_epoch_test_cer

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
        loss.backward()
        epoch_train_loss += loss.item()

        pred_ids = outputs.argmax(dim=-1)

        batch_metrics = get_batch_metrics(pred_ids, decoder_target_ids)
        epoch_train_metrics['batch_accs'].append(batch_metrics['acc'])
        epoch_train_metrics['batch_cers'].append(batch_metrics['cer'])

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
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

    model.eval()
    epoch_test_loss = 0.0
    epoch_test_metrics = {
        'batch_accs': [],
        'batch_cers': [],
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Epoch {epoch+1} [Evaluation]")):
            images, input_ids, attention_mask = batch

            images = images.to(DEVICE)
            input_ids = input_ids.to(DEVICE)

            decoder_input_ids = input_ids[:, :-1]
            decoder_target_ids = input_ids[:, 1:]
            
            outputs = model(images, decoder_input_ids)
            loss = loss_fn(
                outputs.contiguous().view(-1, outputs.size(-1)),
                decoder_target_ids.contiguous().view(-1)
            )
            epoch_test_loss += loss.item()

            pred_ids = outputs.argmax(dim=-1)
            batch_metrics = get_batch_metrics(pred_ids, decoder_target_ids)
            epoch_test_metrics['batch_accs'].append(batch_metrics['acc'])
            epoch_test_metrics['batch_cers'].append(batch_metrics['cer'])

    avg_epoch_test_loss = epoch_test_loss / len(test_loader)
    avg_epoch_test_acc = sum(epoch_test_metrics['batch_accs']) / len(epoch_test_metrics['batch_accs'])
    avg_epoch_test_cer = sum(epoch_test_metrics['batch_cers']) / len(epoch_test_metrics['batch_cers'])
    
    logging.info("-" * 40)
    logging.info(f"TEST | Epoch {epoch+1} - Eval Avg Loss: {avg_epoch_test_loss:.3f}")
    logging.info(f"TEST | Epoch {epoch+1} - Eval Avg Accuracy: {avg_epoch_test_acc:.3f}%")
    logging.info(f"TEST | Epoch {epoch+1} - Eval Avg CER: {avg_epoch_test_cer:.3f}")
    logging.info("-" * 40)
    
    for j in range(min(2, pred_ids.size(0))):
        pred_text = tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True)
        gt_text = tokenizer.decode(decoder_target_ids[j].cpu().tolist(), skip_special_tokens=True)
        logging.info(f"TEST | Epoch {epoch+1} - Sample {j}")
        logging.info(f"TEST | Epoch {epoch+1} - Predicted   : {pred_text}")
        logging.info(f"TEST | Epoch {epoch+1} - Ground Truth: {gt_text}")
    
    logging.info("-" * 40)
    logging.info("\n")

    if best_test_loss is None or avg_epoch_test_loss < best_test_loss:
        best_test_loss = avg_epoch_test_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        logging.info(f"TEST | Epoch {epoch+1} - Best model updated.")

    beam_search_evaluate(epoch, model, test_loader, tokenizer, 'cuda', beam_size=5)
logging.info("Training complete!")
