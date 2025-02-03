import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTModel, ViTFeatureExtractor, RobertaTokenizer, RobertaConfig, RobertaModel
# Assuming OCRDataset is defined in src/dataset.py and works as expected
from src.dataset import OCRDataset

########################################
# Custom Decoder Components
########################################

class CustomDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(CustomDecoderLayer, self).__init__()
        # Self-attention for target tokens
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        # Cross-attention: attend over encoder outputs (e.g., from ViT)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        # Feed-forward network
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        # Layer normalization for each sub-layer
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # --- Self-Attention Sub-layer --- #
        residual = tgt
        self_attn_output, _ = self.self_attn(query=tgt, key=tgt, value=tgt,
                                               attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        tgt = residual + self.dropout1(self_attn_output)
        tgt = self.norm1(tgt)

        # --- Cross-Attention Sub-layer --- #
        residual = tgt
        cross_attn_output, _ = self.cross_attn(query=tgt, key=memory, value=memory,
                                                 attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)
        tgt = residual + self.dropout2(cross_attn_output)
        tgt = self.norm2(tgt)

        # --- Feed-Forward Sub-layer --- #
        residual = tgt
        ffn_output = self.linear2(F.relu(self.linear1(tgt)))
        tgt = residual + self.dropout3(ffn_output)
        tgt = self.norm3(tgt)

        return tgt

class CustomDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_length, dropout=0.1):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Hidden dimension of the model.
            num_layers (int): Number of decoder layers.
            num_heads (int): Number of attention heads.
            max_length (int): Maximum sequence length (for positional embeddings).
            dropout (float): Dropout probability.
        """
        super(CustomDecoder, self).__init__()
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            CustomDecoderLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)
        ])
        # Final linear projection to vocabulary dimension
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_length

    def forward(self, input_ids, encoder_outputs, 
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            input_ids (torch.LongTensor): Target token IDs, shape (batch, target_seq_len).
            encoder_outputs (torch.FloatTensor): Encoder outputs, shape (batch, source_seq_len, hidden_size).
        Returns:
            logits (torch.FloatTensor): Logits over the vocabulary, shape (batch, target_seq_len, vocab_size).
        """
        batch_size, target_seq_len = input_ids.size()

        # Create position indices and compute embeddings
        positions = torch.arange(0, target_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, target_seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        # Transpose for multi-head attention: (target_seq_len, batch, hidden_size)
        x = x.transpose(0, 1)

        # Prepare encoder outputs: (source_seq_len, batch, hidden_size)
        memory = encoder_outputs.transpose(0, 1)

        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, memory,
                      tgt_mask=tgt_mask,
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)

        # Transpose back to (batch, target_seq_len, hidden_size)
        x = x.transpose(0, 1)
        logits = self.fc_out(x)
        return logits

########################################
# Transformer OCR Model
########################################

class TransformerOCR(nn.Module):
    def __init__(self, 
                 vit_model="google/vit-base-patch16-224", 
                 roberta_model="roberta-base", 
                 max_length=32,
                 num_decoder_layers=6,
                 dropout=0.1):
        """
        Args:
            vit_model (str): Pretrained ViT model identifier.
            roberta_model (str): Pretrained RoBERTa model identifier (for configuration and tokenizer/vocab size).
            max_length (int): Maximum text sequence length.
            num_decoder_layers (int): Number of layers in the custom decoder.
            dropout (float): Dropout probability.
        """
        super(TransformerOCR, self).__init__()
        # Encoder: Vision Transformer (ViT)
        self.encoder = ViTModel.from_pretrained(vit_model)
        
        # Get configuration from a RoBERTa model to match decoder dimensions
        self.config = RobertaConfig.from_pretrained(roberta_model)
        vocab_size = self.config.vocab_size
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        
        # Decoder: our custom decoder with cross-attention layers
        self.decoder = CustomDecoder(vocab_size=vocab_size, 
                                     hidden_size=hidden_size, 
                                     num_layers=num_decoder_layers, 
                                     num_heads=num_heads, 
                                     max_length=max_length, 
                                     dropout=dropout)
        self.max_length = max_length

    def forward(self, images, input_ids, attention_mask=None):
        """
        Args:
            images (torch.FloatTensor): Input images tensor.
            input_ids (torch.LongTensor): Target token IDs for the decoder.
            attention_mask: (Optional) attention mask for decoder inputs (not used in this example).
        Returns:
            logits (torch.FloatTensor): Predicted logits over the vocabulary.
        """
        # Encode the images with ViT
        encoder_outputs = self.encoder(images).last_hidden_state  # shape: (batch, source_seq_len, hidden_size)
        logits = self.decoder(input_ids, encoder_outputs)
        return logits

########################################
# Training Script
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
            # Note: In this example, we ignore the attention_mask passed from the dataloader,
            # but you can incorporate it into your decoder if needed.
            
            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            # Reshape logits and targets for computing loss:
            # logits: (batch * seq_len, vocab_size), input_ids: (batch * seq_len)
            loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        validate(model, val_dataloader, tokenizer, device)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

########################################
# Main Execution
########################################

def load_roberta_weights_into_decoder(custom_decoder, roberta_model):
    """
    Loads RoBERTa's weights into the custom decoder.
    
    Args:
        custom_decoder (CustomDecoder): Instance of the custom decoder.
        roberta_model (RobertaModel): Pretrained RoBERTa model.
    """
    roberta_state_dict = roberta_model.state_dict()
    
    # Load token and position embeddings
    if hasattr(roberta_model, 'embeddings'):
        # Word embeddings
        custom_decoder.token_embedding.weight.data.copy_(roberta_model.embeddings.word_embeddings.weight.data)
        # Positional embeddings (if dimensions match)
        if custom_decoder.position_embedding.weight.shape == roberta_model.embeddings.position_embeddings.weight.shape:
            custom_decoder.position_embedding.weight.data.copy_(roberta_model.embeddings.position_embeddings.weight.data)
    
    num_layers = len(custom_decoder.layers)
    
    for i in range(num_layers):
        layer = custom_decoder.layers[i]
        prefix = f'encoder.layer.{i}.'  # RoBERTa uses this naming for each transformer block
        
        # --- Self-Attention Mapping --- #
        # Get RoBERTa's self-attention weights for query, key, value
        q_weight = roberta_state_dict[f'{prefix}attention.self.query.weight']
        k_weight = roberta_state_dict[f'{prefix}attention.self.key.weight']
        v_weight = roberta_state_dict[f'{prefix}attention.self.value.weight']
        # Concatenate weights as in nn.MultiheadAttention expects: [q; k; v]
        in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        layer.self_attn.in_proj_weight.data.copy_(in_proj_weight)
        
        # Similarly for biases
        q_bias = roberta_state_dict[f'{prefix}attention.self.query.bias']
        k_bias = roberta_state_dict[f'{prefix}attention.self.key.bias']
        v_bias = roberta_state_dict[f'{prefix}attention.self.value.bias']
        in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        layer.self_attn.in_proj_bias.data.copy_(in_proj_bias)
        
        # Output projection for self-attention
        layer.self_attn.out_proj.weight.data.copy_(roberta_state_dict[f'{prefix}attention.output.dense.weight'])
        layer.self_attn.out_proj.bias.data.copy_(roberta_state_dict[f'{prefix}attention.output.dense.bias'])
        
        # --- Feed-Forward (FFN) Mapping --- #
        # First linear layer (intermediate dense in RoBERTa)
        layer.linear1.weight.data.copy_(roberta_state_dict[f'{prefix}intermediate.dense.weight'])
        layer.linear1.bias.data.copy_(roberta_state_dict[f'{prefix}intermediate.dense.bias'])
        
        # Second linear layer (output dense in RoBERTa)
        layer.linear2.weight.data.copy_(roberta_state_dict[f'{prefix}output.dense.weight'])
        layer.linear2.bias.data.copy_(roberta_state_dict[f'{prefix}output.dense.bias'])
        
        # --- LayerNorm Mapping --- #
        # Map the first layer norm to RoBERTa's attention output LayerNorm
        layer.norm1.weight.data.copy_(roberta_state_dict[f'{prefix}attention.output.LayerNorm.weight'])
        layer.norm1.bias.data.copy_(roberta_state_dict[f'{prefix}attention.output.LayerNorm.bias'])
        
        # Map the third layer norm to RoBERTa's output LayerNorm
        layer.norm3.weight.data.copy_(roberta_state_dict[f'{prefix}output.LayerNorm.weight'])
        layer.norm3.bias.data.copy_(roberta_state_dict[f'{prefix}output.LayerNorm.bias'])
        # The middle norm (norm2) is for cross-attention and remains randomly initialized.

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize feature extractor and tokenizer (from the respective pretrained models)
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Create the dataset. Adjust the file paths as needed.
    dataset = OCRDataset(
        text_file="/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/train_converted.txt", 
        img_dir="/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K", 
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    val_dataset = OCRDataset(
        text_file="/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/val_converted.txt", 
        img_dir="/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K", 
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    model = TransformerOCR()
    roberta_pretrained = RobertaModel.from_pretrained("roberta-base")
    load_roberta_weights_into_decoder(model.decoder, roberta_pretrained)
    
    train(model, dataloader, tokenizer, device, val_dataloader=val_dataloader)


if __name__ == "__main__":
    main()
