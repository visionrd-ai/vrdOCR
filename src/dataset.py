import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor
from transformers import AutoImageProcessor  
from PIL import Image
import os
import albumentations as A
import numpy as np 

PAD_TOKEN = 3 

class ViTOCR(Dataset):
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

        self.transforms =  A.Compose([
                                        A.RandomBrightnessContrast(p=0.4),  # Random brightness/contrast adjustment
                                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.4),  # Random shift, scale, rotate
                                        A.ToFloat(),  # Normalize to 0-1 range
                                    ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_path = os.path.join(img_path)

        # Process the image.
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
    
        augmented = self.transforms(image=image_np)
        image_np = augmented["image"]
        
        image = Image.fromarray((image_np * 255).astype(np.uint8))
        image = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        # tokenized = self.tokenizer(
        #     label, 
        #     add_special_tokens=False,  
        #     truncation=True, 
        #     max_length=self.max_length - 2,  # reserve space for the start and end tokens.
        #     return_tensors="pt"
        # )
        tokenized = self.tokenizer(
                                    label, 
                                    add_special_tokens=False,  
                                    truncation=False, 
                                    return_tensors="pt"
                                )
        input_ids = tokenized["input_ids"].squeeze(0)
        
        decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)#self.tokenizer.bos_token#self.tokenizer.pad_token_id#self.tokenizer.decoder_start_token_id or self.tokenizer.eos_token_id
        decoder_end_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        input_ids = torch.cat([torch.tensor([decoder_start_token_id]), input_ids, torch.tensor([decoder_end_token_id])], dim=0)
        # if input_ids.shape[0] < self.max_length:
        #     pad_length = self.max_length - input_ids.shape[0]
        #     input_ids = torch.cat(
        #         [input_ids, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)],
        #         dim=0
        #     )
        
        # zero out attention mask for padding tokens
        # attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        assert decoder_end_token_id in input_ids and decoder_start_token_id in input_ids, f"decoder_start_token_id: {decoder_start_token_id}, decoder_end_token_id: {decoder_end_token_id}, input_ids: {input_ids}"
        return image, input_ids, attention_mask


from torch.nn.utils.rnn import pad_sequence

def dynamic_collate_fn(batch):
    # Unpack the batch: each item is a tuple (image, input_ids, attention_mask)
    images, input_ids_list, attention_masks_list = zip(*batch)
    
    # Stack images (assuming they are all the same size)
    images = torch.stack(images)
    
    # Pad the sequences dynamically (using pad_sequence from PyTorch)
    # pad_sequence expects a list of tensors and pads them to the max length in the batch.
    # We use batch_first=True to get output shape (batch_size, max_seq_length)
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=PAD_TOKEN)
    padded_attention_masks = pad_sequence(attention_masks_list, batch_first=True, padding_value=0)
    
    return images, padded_input_ids, padded_attention_masks

def get_vit_dataloader(text_file, image_root, tokenizer, batch_size=4, shuffle=True, max_length=128):
    return DataLoader(ViTOCR(text_file, image_root, tokenizer=tokenizer, max_length=max_length), 
                      batch_size=batch_size, shuffle=shuffle, num_workers=8, collate_fn=dynamic_collate_fn)


class SwinOCR(Dataset):
    def __init__(self, text_file, image_root, swin_model="microsoft/swin-tiny-patch4-window7-224", tokenizer=None, max_length=128):
        """
        Args:
            text_file: a text file with lines formatted as "img_path<TAB>label".
            image_root: root directory for images.
            swin_model: model identifier for the Swin image processor.
            tokenizer: a tokenizer (e.g. from a pretrained BART model).
            max_length: maximum target sequence length.
        """
        self.image_root = image_root 
        self.max_length = max_length  

        with open(text_file, "r") as f:
            lines = f.readlines()
        # Expecting each line to contain two fields: image_path and corresponding label.
        self.data = [line.strip().split("\t") for line in lines]

        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        if tokenizer is None:
            raise ValueError("A tokenizer is required.")
        self.tokenizer = tokenizer

        self.transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.4),              # Random brightness/contrast adjustment
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),       # Add random noise
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.4),  # Random shift/scale/rotate
            A.ToFloat(),                                      # Normalize to 0-1 range
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve image path and label.
        relative_img_path, label = self.data[idx]
        img_path = os.path.join(relative_img_path)

        # Open image and convert to RGB.
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
    
        # Apply augmentations.
        augmented = self.transforms(image=image_np)
        image_np = augmented["image"]
        
        # Convert back to PIL Image for the processor.
        image = Image.fromarray((image_np * 255).astype(np.uint8))
        image = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Tokenize label without adding special tokens (we do that manually).
        tokenized = self.tokenizer(
            label, 
            add_special_tokens=False,  
            truncation=False, 
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        
        # Manually add the special tokens expected by BART.
        decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        decoder_end_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        input_ids = torch.cat([
            torch.tensor([decoder_start_token_id]),
            input_ids,
            torch.tensor([decoder_end_token_id])
        ], dim=0)

        # Create an attention mask (here we use 1s for all tokens).
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        return image, input_ids, attention_mask

def get_dynamic_collate_fn(tokenizer):
    """
    Returns a collate function that pads input_ids and attention_masks using the tokenizer's pad token.
    """
    def dynamic_collate_fn(batch):
        images, input_ids_list, attention_masks_list = zip(*batch)
        images = torch.stack(images)
        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks_list, batch_first=True, padding_value=0)
        return images, padded_input_ids, padded_attention_masks
    return dynamic_collate_fn

def get_swin_dataloader(text_file, image_root, tokenizer, batch_size=4, shuffle=True, max_length=128):
    collate_fn = get_dynamic_collate_fn(tokenizer)
    dataset = SwinOCR(text_file, image_root, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, collate_fn=collate_fn)