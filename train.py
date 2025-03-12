import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import ViT_TransformerDecoder
from dataset import get_dataloader
from char_tokenizer import CharacterLevelTokenizer
import Levenshtein  # Install with `pip install python-Levenshtein`
import logging

# Hyperparameters and settings.
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
GRADIENT_ACCUMULATION_STEPS = 2  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_vit_t5.pth"

tokenizer = CharacterLevelTokenizer()
vocab_size = len(tokenizer)

model = ViT_TransformerDecoder(vocab_size=vocab_size).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

train_loader = get_dataloader(
    '/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/train/annotations.txt', 
    '/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K',
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer
)
test_loader = get_dataloader(
    '/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/test/annotations.txt', 
    '/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K',
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer
)

# Setup logging
logging.basicConfig(filename="training_log.txt", level=logging.INFO, 
                    format="%(asctime)s - %(message)s")
logging.info("Training started...")

best_test_loss = None

def get_batch_metrics(pred_ids, decoder_target_ids):

    pred_texts = [tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True) for j in range(pred_ids.shape[0])]
    gt_texts = [tokenizer.decode(decoder_target_ids[j].cpu().tolist(), skip_special_tokens=True) for j in range(pred_ids.shape[0])]

    correct = sum([1 for pred_text, gt_text in zip(pred_texts, gt_texts) if pred_text == gt_text])
    total = len(pred_texts)
    accuracy = correct / total * 100
    
    dists = [calculate_cer(pred_text, gt_text) for pred_text, gt_text in zip(pred_texts, gt_texts)]
    cer = sum(dists) / total

    return {'acc':accuracy, 'cer':cer}

def calculate_cer(pred_text, gt_text):
    return Levenshtein.distance(pred_text, gt_text) / len(gt_text) if len(gt_text) > 0 else 0

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


    # model.eval()
    # test_loss = 0.0
    # test_correct = 0
    # test_total = 0
    # test_cer = 0.0

    # with torch.no_grad():
    #     for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} [Evaluating]"):
    #         images, input_ids, attention_mask = batch
    #         images = images.to(DEVICE)
    #         input_ids = input_ids.to(DEVICE)
            
    #         decoder_input_ids = input_ids[:, :-1]
    #         decoder_target_ids = input_ids[:, 1:]
            
    #         outputs = model(images, decoder_input_ids)

    #         pred_ids = outputs.argmax(dim=-1)
            

    #         # Calculate CER and accuracy for the batch.
    #         for j in range(pred_ids.size(0)):
    #             pred_text = tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True)
    #             gt_text = tokenizer.decode(decoder_target_ids[j].cpu().tolist(), skip_special_tokens=True)
    #             test_cer += calculate_cer(pred_text, gt_text)
                
    #             if pred_text == gt_text:
    #                 test_correct += 1
    #             test_total += 1

    #         loss = loss_fn(outputs.contiguous().view(-1, outputs.size(-1)), decoder_target_ids.contiguous().view(-1))

    #         test_loss += loss.item()
            
    # for j in range(min(2, pred_ids.size(0))):  # Show first two samples.
    #     pred_text = tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True)
    #     gt_text = tokenizer.decode(decoder_target_ids[j].cpu().tolist(), skip_special_tokens=True)
    #     logging.info(f"Epoch {epoch+1} - Sample {j}:")
    #     logging.info(f"  Predicted   : {pred_text}")
    #     logging.info(f"  Ground Truth: {gt_text}")
    #     logging.info("-" * 40)
    # avg_test_loss = test_loss / len(test_loader)
    # test_accuracy = test_correct / test_total * 100
    # avg_test_cer = test_cer / test_total

    # # Log validation stats
    # logging.info(f"Epoch {epoch+1} - Test Loss: {avg_test_loss:.4f}, "
    #              f"Test Accuracy: {test_accuracy:.2f}%, "
    #              f"Test CER: {avg_test_cer:.4f}")

    # if best_test_loss is None or avg_test_loss < best_test_loss:
    #     best_test_loss = avg_test_loss
    #     torch.save(model.state_dict(), BEST_MODEL_PATH)
    #     logging.info(f"Saved best model to {BEST_MODEL_PATH}")

logging.info("Training complete!")
