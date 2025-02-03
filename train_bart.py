import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    ViTModel,
    ViTFeatureExtractor,
    RobertaTokenizer,  # for tokenization (assumes same vocab as BART, or you may switch to BartTokenizer)
    BartForConditionalGeneration,
)
# Assuming OCRDataset is defined in src/dataset.py and works as expected
from src.dataset import OCRDataset

########################################
# Transformer OCR Model using BART Decoder
########################################

class TransformerOCR(nn.Module):
    def __init__(self, 
                 vit_model="google/vit-base-patch16-224", 
                 bart_model="facebook/bart-base", 
                 max_length=32):
        """
        Args:
            vit_model (str): Pretrained ViT model identifier.
            bart_model (str): Pretrained BART model identifier to be used as the decoder.
            max_length (int): Maximum text sequence length.
        """
        super(TransformerOCR, self).__init__()
        # Encoder: Vision Transformer (ViT)
        self.encoder = ViTModel.from_pretrained(vit_model)
        
        # Decoder: Use BART's decoder and language modeling head from BartForConditionalGeneration.
        # We are discarding BART's encoder and only using its decoder.
        bart = BartForConditionalGeneration.from_pretrained(bart_model)
        # Extract the decoder (which includes cross-attention layers) and LM head.
        self.decoder = bart.model.decoder
        self.lm_head = bart.lm_head
        # Bart’s config may be useful (e.g., for max_length, d_model, etc.)
        self.config = bart.config
        self.max_length = max_length

    def forward(self, images, input_ids, attention_mask=None):
        """
        Args:
            images (torch.FloatTensor): Input images tensor.
            input_ids (torch.LongTensor): Target token IDs for the decoder.
            attention_mask: (Optional) attention mask for decoder inputs.
        Returns:
            logits (torch.FloatTensor): Predicted logits over the vocabulary.
        """
        # Encode images with ViT.
        encoder_outputs = self.encoder(images).last_hidden_state  # shape: (batch, source_seq_len, hidden_size)
        # Pass target tokens along with encoder outputs to BART's decoder.
        # BART's decoder expects:
        #    input_ids: (batch, target_seq_len)
        #    encoder_hidden_states: (batch, source_seq_len, d_model)
        decoder_outputs = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_outputs,encoder_attention_mask=None)#,  # adjust if needed)
        # decoder_outputs.last_hidden_state shape: (batch, target_seq_len, d_model)
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits

########################################
# Training and Validation Functions
########################################

def train(model, dataloader, tokenizer, device, val_dataloader, epochs=100, lr=5e-5):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    print("Starting training")
    for epoch in range(epochs):
        total_loss = 0
        for images, input_ids, attention_mask in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            # Note: In this example, we ignore the attention_mask from the dataloader.
            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            # Reshape logits and targets for computing loss:
            # logits: (batch * seq_len, vocab_size), input_ids: (batch * seq_len)
            loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        validate(model, val_dataloader, tokenizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

def validate(model, dataloader, tokenizer, device):
    """
    Validation function that computes the loss and decodes the model's outputs
    into text to display predictions alongside ground truth labels.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, input_ids, attention_mask in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            total_loss += loss.item()
            # Get predictions by taking the argmax along the vocabulary dimension
            predictions = torch.argmax(logits, dim=-1)
            # For each sample in the batch, decode prediction and target text
            for pred, target in zip(predictions, input_ids):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                target_text = tokenizer.decode(target, skip_special_tokens=True)
                all_predictions.append(pred_text)
                all_targets.append(target_text)
                print(f"Predicted: {pred_text} | Target: {target_text}")
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss, all_predictions, all_targets

########################################
# Main Execution
########################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize feature extractor and tokenizer.
    # You can continue to use the RobertaTokenizer if its vocabulary is compatible,
    # or switch to BartTokenizer (from transformers) for consistency.
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Create the training dataset. Adjust file paths as needed.
    dataset = OCRDataset(
        text_file="/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/train_converted.txt", 
        img_dir="/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K", 
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create the validation dataset.
    val_dataset = OCRDataset(
        text_file="/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/val_converted.txt", 
        img_dir="/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K", 
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    
    # Initialize the Transformer OCR model (using ViT encoder and BART decoder)
    model = TransformerOCR()
    
    train(model, dataloader, tokenizer, device, val_dataloader=val_dataloader)

if __name__ == "__main__":
    main()
