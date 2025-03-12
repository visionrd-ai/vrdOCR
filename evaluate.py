import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from model import ViT_TransformerDecoder
from dataset import get_dataloader
from transformers import T5Tokenizer
import Levenshtein  # Install with `pip install python-Levenshtein`
from tqdm import tqdm 

# Hyperparameters and settings.
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_vit_t5.pth"

tokenizer = T5Tokenizer.from_pretrained("t5-small")
vocab_size = len(tokenizer)

# Load the pre-trained model
model = ViT_TransformerDecoder(vocab_size=vocab_size).to(DEVICE)

# Load the best model weights
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print(f"Loaded best model from {BEST_MODEL_PATH}")
else:
    print(f"Model file not found at {BEST_MODEL_PATH}, make sure the path is correct.")
    exit()

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Data loader for the test set
test_loader = get_dataloader(
    '/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/val_converted.txt', 
    '/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K',
    batch_size=BATCH_SIZE
)

# Evaluate the model
model.eval()
test_loss = 0.0
correct = 0
total = 0
total_cer = 0  # To accumulate CER for all batches

with torch.no_grad():
    for batch in tqdm(test_loader):
        images, input_ids, attention_mask = batch
        images = images.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        
        # Forward pass
        outputs = model(images, input_ids)

        # Get predicted ids
        pred_ids = outputs.argmax(dim=-1) 

        # for j in range(min(2, pred_ids.size(0))):  # Visualizing first 2 samples in the batch
        #     pred_text = tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True)
        #     gt_text = tokenizer.decode(input_ids[j].cpu().tolist(), skip_special_tokens=True)

        #     # Visualize the image and display predicted and ground truth texts
        #     img = images[j].cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy (H, W, C)
        #     img = (img * 255).astype("uint8")  # Assuming image was normalized
            
        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.title(f"Pred: {pred_text}\nGT: {gt_text}")
        #     plt.show()

        for j in range(pred_ids.size(0)):
            pred_text = tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True)
            gt_text = tokenizer.decode(input_ids[j].cpu().tolist(), skip_special_tokens=True)
            
            cer = Levenshtein.distance(pred_text, gt_text) / len(gt_text) if len(gt_text) > 0 else 0
            total_cer += cer

            if pred_text == gt_text:
                correct += 1
            total += 1
        
        # Calculate loss
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
        test_loss += loss.item()

# Average test loss, accuracy, and CER
avg_test_loss = test_loss / len(test_loader)
accuracy = correct / total * 100  # Calculate accuracy percentage
avg_cer = total_cer / total  # Average CER

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Average Character Error Rate (CER): {avg_cer:.4f}")
