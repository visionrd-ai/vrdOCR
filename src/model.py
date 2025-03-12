import math
import torch
import torch.nn as nn
from transformers import ViTModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Arguments:
            d_model: the feature dimension of the model.
            dropout: dropout probability.
            max_len: maximum length of the input sequences.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ViT_TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,       
        d_model=512,       
        num_decoder_layers=6,
        nhead=8,
        dropout=0.1,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.projection = nn.Linear(768, d_model)
        
        self.encoder_pos = PositionalEncoding(d_model, dropout)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_pos = PositionalEncoding(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. The masked positions are filled with -inf.
        This ensures that the model only attends to previous positions (autoregressive behavior).
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, images, tgt_input_ids):
        """
        Arguments:
            images: Tensor of shape (batch_size, channels, height, width)
            tgt_input_ids: Tensor of shape (batch_size, tgt_seq_len) with token indices.
        Returns:
            logits: Tensor of shape (batch_size, tgt_seq_len, vocab_size)
        """
        encoder_outputs = self.vit(images).last_hidden_state
        encoder_outputs = self.projection(encoder_outputs)
        encoder_outputs = self.encoder_pos(encoder_outputs)
        
        tgt_embeddings = self.token_embedding(tgt_input_ids)  # (batch, tgt_seq_len, d_model)
        tgt_embeddings = self.decoder_pos(tgt_embeddings)
        
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (src_seq_len, batch, d_model)
        tgt_embeddings = tgt_embeddings.transpose(0, 1)        # (tgt_seq_len, batch, d_model)
        
        tgt_seq_len = tgt_embeddings.size(0)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt_embeddings.device)
        
        decoder_output = self.transformer_decoder(
            tgt=tgt_embeddings,
            memory=encoder_outputs,
            tgt_mask=tgt_mask
        )
        decoder_output = decoder_output.transpose(0, 1)
        
        logits = self.fc_out(decoder_output)
        return logits