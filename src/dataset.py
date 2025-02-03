import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTFeatureExtractor, BartTokenizer
# Dataset
class OCRDataset(Dataset):
    def __init__(self, text_file, img_dir, feature_extractor, tokenizer, max_length=32):
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load image paths and labels from text file
        with open(text_file, "r") as f:
            lines = f.readlines()

        self.data = [line.strip().split("\t") for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(os.path.join(self.img_dir, img_path)).convert("RGB")
        img = self.feature_extractor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        
        label_tokens = self.tokenizer(label, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        input_ids = label_tokens["input_ids"].squeeze(0)
        attention_mask = label_tokens["attention_mask"].squeeze(0)
        
        return img, input_ids, attention_mask