import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, ViTImageProcessor
from PIL import Image
import os
from transformers import T5TokenizerFast

class OCRDataset(Dataset):
    def __init__(self, text_file, image_root, vit_model="google/vit-base-patch16-224", tokenizer=None, max_length=128):
        self.image_root = image_root 
        self.max_length = max_length  

        with open(text_file, "r") as f:
            lines = f.readlines()
        self.data = [line.strip().split("\t") for line in lines]

        self.image_processor = ViTImageProcessor.from_pretrained(vit_model)
        if not tokenizer:
            raise ValueError("A tokenizer is required.")
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_path = os.path.join(img_path)

        # Process the image.
        image = Image.open(img_path).convert("RGB")
        image = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        tokenized = self.tokenizer(
            label, 
            add_special_tokens=False,  
            truncation=True, 
            max_length=self.max_length - 2,  # reserve space for the start and end tokens.
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        
        decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)#self.tokenizer.bos_token#self.tokenizer.pad_token_id#self.tokenizer.decoder_start_token_id or self.tokenizer.eos_token_id
        decoder_end_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        input_ids = torch.cat([torch.tensor([decoder_start_token_id]), input_ids, torch.tensor([decoder_end_token_id])], dim=0)
        
        if input_ids.shape[0] < self.max_length:
            pad_length = self.max_length - input_ids.shape[0]
            input_ids = torch.cat(
                [input_ids, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)],
                dim=0
            )
        
        # zero out attention mask for padding tokens
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return image, input_ids, attention_mask


def get_dataloader(text_file, image_root, tokenizer, batch_size=4, shuffle=True):
    return DataLoader(OCRDataset(text_file, image_root, tokenizer=tokenizer), batch_size=batch_size, shuffle=shuffle, num_workers=8)
