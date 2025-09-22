import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.self_attn = self_attn

        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)

        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        B, qN, C = query.shape

        if self.self_attn:
            qkv = self.qkv(query).view(B, qN, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # (B, heads, N, head_dim)
        else:
            assert key is not None, "cross-attn requires key"
            kN = key.size(1)
            q = self.q(query).view(B, qN, self.num_heads, self.head_dim).permute(0,2,1,3)
            kv = self.kv(key).view(B, kN, 2, self.num_heads, self.head_dim).permute(2,0,3,1,4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).contiguous().view(B, qN, C)
        x = self.out_proj(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.1, act=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        with_self_attn=True,
        with_cross_attn=False,
        epsilon=1e-5,
    ):
        super().__init__()
        self.with_self_attn = with_self_attn
        self.with_cross_attn = with_cross_attn

        if with_self_attn:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate, self_attn=True)
            self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
            self.drop1 = nn.Dropout(residual_dropout_rate)

        if with_cross_attn:
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
            self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
            self.drop2 = nn.Dropout(residual_dropout_rate)

        self.ffn = FFN(d_model, dim_feedforward, drop=residual_dropout_rate, act=nn.ReLU)
        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)
        self.drop3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            y = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.drop1(y))
        if self.with_cross_attn:
            y = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.drop2(y))
        y = self.ffn(tgt)
        tgt = self.norm3(tgt + self.drop3(y))
        return tgt


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx=0, scale_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.scale_embedding = scale_embedding
        self.d_model = d_model
        # init like transformer
        with torch.no_grad():
            self.embedding.weight.normal_(mean=0.0, std=d_model ** -0.5)

    def forward(self, x):
        e = self.embedding(x)
        if self.scale_embedding:
            e = e * math.sqrt(self.d_model)
        return e


class Transformer(nn.Module):
    """
    NRTR-style decoder-only transformer with optional (disabled here) encoder blocks.
    Inputs:
      - src: (B, S, C) sequence features (from before_gtc)
      - targets: Optional training targets (tuple or tensor). If training, teacher-forced.
    """
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=-1,
        beam_size=-1,
        num_decoder_layers=6,
        max_len=25,
        dim_feedforward=2048,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        out_channels=0,
        scale_embedding=True,
    ):
        super().__init__()
        self.out_channels = out_channels + 1  # +1 for pad/blank per your original code
        self.max_len = max_len
        self.beam_size = beam_size

        self.embedding = Embeddings(d_model, self.out_channels, padding_idx=0, scale_embedding=scale_embedding)

        # (Optional) encoder; disabled by default (num_encoder_layers <= 0)
        self.encoder = None
        if num_encoder_layers and num_encoder_layers > 0:
            self.encoder = nn.ModuleList([
                TransformerBlock(d_model, nhead, dim_feedforward, attention_dropout_rate, residual_dropout_rate,
                                 with_self_attn=True, with_cross_attn=False)
                for _ in range(num_encoder_layers)
            ])

        self.decoder = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, attention_dropout_rate, residual_dropout_rate,
                             with_self_attn=True, with_cross_attn=True)
            for _ in range(num_decoder_layers)
        ])

        # projection
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels, bias=False)
        # xavier init
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

    def _subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device='cpu'), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf')).unsqueeze(0).unsqueeze(0)
        return mask  # (1,1,sz,sz) to broadcast over (B,heads,...)

    def forward_train(self, src: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        # teacher-forcing: use input tokens shifted (remove last)
        tgt_in = tgt_tokens[:, :-1]                   # (B, T_in)
        tgt = self.embedding(tgt_in)                  # (B, T_in, C)
        tgt_mask = self._subsequent_mask(tgt.size(1)).to(tgt.device)

        # encoder (optional)
        memory = src
        if self.encoder is not None:
            for enc in self.encoder:
                memory = enc(memory)

        # decoder
        for dec in self.decoder:
            tgt = dec(tgt, memory, self_mask=tgt_mask)

        logits = self.tgt_word_prj(tgt)              # (B, T_in, vocab)
        return logits

    def forward_test(self, src: torch.Tensor):
        B = src.size(0)
        memory = src
        if self.encoder is not None:
            for enc in self.encoder:
                memory = enc(memory)

        # start token 2, end token 3 (as in your original)
        dec_seq = torch.full((B, 1), 2, dtype=torch.long, device=src.device)
        for _ in range(1, self.max_len):
            dec_embed = self.embedding(dec_seq)                # (B, t, C)
            tgt_mask = self._subsequent_mask(dec_embed.size(1)).to(src.device)
            tgt = dec_embed
            for dec in self.decoder:
                tgt = dec(tgt, memory, self_mask=tgt_mask)
            last = tgt[:, -1, :]                               # (B, C)
            word_prob = F.softmax(self.tgt_word_prj(last), -1) # (B, vocab)
            next_id = word_prob.argmax(-1)                     # (B,)
            dec_seq = torch.cat([dec_seq, next_id[:, None]], dim=1)
            # stop early if all EOS (3)
            if torch.all(next_id.eq(3)):
                break
        return dec_seq

    def forward(self, src: torch.Tensor, targets=None):
        if self.training and targets is not None:
            # allow (tokens, lengths) or just tokens
            if isinstance(targets, (list, tuple)):
                tgt_tokens = targets[0]
            else:
                tgt_tokens = targets
            return self.forward_train(src, tgt_tokens)
        else:
            return self.forward_test(src)
