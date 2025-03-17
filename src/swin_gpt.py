import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import SwinModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Swin_GPT2Decoder(nn.Module):
    def __init__(self, vocab_size, dropout=0.1):
        super().__init__()
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.projection = nn.Linear(768, 768)
        self.encoder_pos = PositionalEncoding(768, dropout)

        config = GPT2Config.from_pretrained("gpt2")
        config.add_cross_attention = True  # This adds cross attention layers in each block.
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        self.gpt2.resize_token_embeddings(vocab_size)

    def forward(self, images, tgt_input_ids, attention_mask=None):
        """
        During training, tgt_input_ids are passed in (e.g. teacher forcing) and
        the image encoder outputs are provided as cross-attention memory.
        """
        # Get image features from swin: (batch, src_seq_len, 768)

        encoder_outputs = self.swin(images).last_hidden_state
        encoder_outputs = self.projection(encoder_outputs)
        encoder_outputs = self.encoder_pos(encoder_outputs)
        # GPT2 expects encoder_hidden_states as (batch, src_seq_len, hidden_size)
        outputs = self.gpt2(input_ids=tgt_input_ids, encoder_hidden_states=encoder_outputs, encoder_attention_mask=None)
        logits = outputs.logits  # (batch, tgt_seq_len, vocab_size)
        return logits

    def generate(self, images, beam_size=3, max_length=50,
                 start_token_id=None, end_token_id=None, pad_token_id=None):
        """
        A simple beam search decoding loop. In practice you might also consider using
        the HuggingFace generate() method after adapting it to handle cross attention.
        """
        self.eval()
        device = images.device
        with torch.no_grad():
            encoder_outputs = self.swin(images).last_hidden_state
            encoder_outputs = self.projection(encoder_outputs)
            encoder_outputs = self.encoder_pos(encoder_outputs)
            
            batch_size = images.size(0)
            generated_sequences = []
            
            for i in range(batch_size):
                # Select encoder outputs for the current sample: shape (1, src_seq_len, hidden_size)
                memory = encoder_outputs[i:i+1, :, :]
                beam = [([start_token_id], 0.0)]
                
                for _ in range(max_length):
                    new_beam = []
                    for seq, score in beam:
                        # If the sequence is already finished, keep it.
                        if seq[-1] == end_token_id:
                            new_beam.append((seq, score))
                            continue
                        # Prepare decoder input
                        decoder_input_ids = torch.tensor(seq, device=device).unsqueeze(0)  # (1, seq_len)
                        outputs = self.gpt2(
                            input_ids=decoder_input_ids,
                            encoder_hidden_states=memory,
                            encoder_attention_mask=None
                        )
                        # Get logits for the last time step
                        logits = outputs.logits[:, -1, :]  # (1, vocab_size)
                        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size)
                        topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                        for k in range(beam_size):
                            next_token = topk_indices[k].item()
                            next_score = score + topk_log_probs[k].item()
                            new_seq = seq + [next_token]
                            new_beam.append((new_seq, next_score))
                    # Keep best beam_size sequences
                    beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
                    if all(seq[-1] == end_token_id for seq, _ in beam):
                        break
                best_sequence = beam[0][0]
                # Pad the sequence if needed
                if len(best_sequence) < max_length:
                    best_sequence = best_sequence + [pad_token_id] * (max_length - len(best_sequence))
                generated_sequences.append(torch.tensor(best_sequence, device=device))
            return torch.stack(generated_sequences)
