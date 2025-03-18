import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartConfig, SwinModel
from transformers.modeling_outputs import BaseModelOutput
from transformers import BartModel

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

class Swin_BARTDecoder(nn.Module):
    def __init__(self, vocab_size, dropout=0.1):
        super().__init__()
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.projection = nn.Linear(768, 768)
        self.encoder_pos = PositionalEncoding(768, dropout)
        
        # self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        config = BartConfig(
            d_model=768, 
            decoder_layers=2, 
            decoder_ffn_dim=1024, 
            decoder_attention_heads=2,
        )
        self.bart = BartForConditionalGeneration(config)
        self.bart.resize_token_embeddings(vocab_size)

    def forward_teacher_forcing(self, images, tgt_input_ids, attention_mask=None):
        """
        During training, tgt_input_ids (with bos token) are provided
        and the image encoder outputs are passed as cross-attention memory.
        """
        # Get image features from Swin: (batch, src_seq_len, 768)
        encoder_outputs = self.swin(images).last_hidden_state
        encoder_outputs = self.projection(encoder_outputs)
        encoder_outputs = self.encoder_pos(encoder_outputs)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)
        
        outputs = self.bart(
            input_ids=tgt_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask
        )
        logits = outputs.logits  # (batch, tgt_seq_len, vocab_size)
        return logits
    
    def forward(self, images, tgt_input_ids, attention_mask=None, sampling_probability=0.0):
        """
        Args:
            images: Input images.
            tgt_input_ids: Tensor of shape (batch_size, seq_len) with ground truth token ids.
            sampling_probability: Probability of using the model's prediction instead of the ground truth at each time step.
                                This value should be scheduled (increased) over training epochs.
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        encoder_outputs = self.swin(images).last_hidden_state
        encoder_outputs = self.projection(encoder_outputs)
        encoder_outputs = self.encoder_pos(encoder_outputs)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)
        
        batch_size, seq_length = tgt_input_ids.size()  # e.g., seq_length might be 6 (BOS + 5 tokens)
        device = tgt_input_ids.device

        decoder_input_ids = tgt_input_ids[:, 0].unsqueeze(1)  # shape: (batch_size, 1)
        all_logits = []
        
        for t in range(seq_length - 1):
            outputs = self.bart(
                input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
            )
            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            all_logits.append(logits.unsqueeze(1))
            
            probs = torch.softmax(logits, dim=-1)
            predicted_tokens = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            use_pred = (torch.rand(batch_size, device=device) < sampling_probability).unsqueeze(1)
            ground_truth_tokens = tgt_input_ids[:, t+1].unsqueeze(1)
            next_tokens = torch.where(use_pred, predicted_tokens, ground_truth_tokens)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)
        
        logits = torch.cat(all_logits, dim=1)  # Shape: (batch_size, seq_length - 1, vocab_size)
        return logits

    def generate(self, images, beam_size=3, max_length=50,
                 start_token_id=None, end_token_id=None, pad_token_id=None):
        """
        A simple beam search decoding loop. You could also use self.bart.generate()
        after preparing the encoder outputs.
        """
        self.eval()
        device = images.device
        with torch.no_grad():
            encoder_outputs = self.swin(images).last_hidden_state
            encoder_outputs = self.projection(encoder_outputs)
            encoder_outputs = self.encoder_pos(encoder_outputs)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)
            
            batch_size = images.size(0)
            generated_sequences = []
            
            for i in range(batch_size):
                # Get the encoder outputs for one sample: (1, src_seq_len, hidden_size)
                memory = encoder_outputs.last_hidden_state[i:i+1, :, :]
                beam = [([start_token_id], 0.0)]
                
                for _ in range(max_length):
                    new_beam = []
                    for seq, score in beam:
                        if seq[-1] == end_token_id:
                            new_beam.append((seq, score))
                            continue
                        # Prepare decoder input (shape: (1, seq_len))
                        decoder_input_ids = torch.tensor(seq, device=device).unsqueeze(0)
                        # Call BART with the current sequence and the image encoder's memory.
                        outputs = self.bart(
                            input_ids=decoder_input_ids,
                            encoder_outputs=BaseModelOutput(last_hidden_state=memory)
                        )
                        logits = outputs.logits[:, -1, :]  # (1, vocab_size)
                        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                        topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                        for k in range(beam_size):
                            next_token = topk_indices[k].item()
                            next_score = score + topk_log_probs[k].item()
                            new_seq = seq + [next_token]
                            new_beam.append((new_seq, next_score))
                    beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
                    if all(seq[-1] == end_token_id for seq, _ in beam):
                        break
                
                best_sequence = beam[0][0]
                best_sequence = best_sequence[:max_length]
                if len(best_sequence) < max_length:
                    best_sequence += [pad_token_id] * (max_length - len(best_sequence))
                generated_sequences.append(torch.tensor(best_sequence, device=device))
            return torch.stack(generated_sequences)
