import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm
from src.svtrnet_backbone import Mlp, zeros_

class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
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
        qN = query.size(1)

        if self.self_attn:
            qkv = self.qkv(query).view(query.size(0), qN, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            kN = key.size(1)
            q = self.q(query).view(query.size(0), qN, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(key).view(key.size(0), kN, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn += attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 1, 3).contiguous().view(query.size(0), qN, self.embed_dim)
        x = self.out_proj(x)

        return x


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
        super(TransformerBlock, self).__init__()
        self.with_self_attn = with_self_attn
        
        if with_self_attn:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate, self_attn=True)
            self.norm1 = LayerNorm(d_model, eps=epsilon)
            self.dropout1 = Dropout(residual_dropout_rate)
        
        self.with_cross_attn = with_cross_attn
        
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
            self.norm2 = LayerNorm(d_model, eps=epsilon)
            self.dropout2 = Dropout(residual_dropout_rate)

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            act_layer=nn.ReLU,
            drop=residual_dropout_rate,
        )

        self.norm3 = LayerNorm(d_model, eps=epsilon)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))

        if self.with_cross_attn:
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence."""

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).transpose(0, 1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(dim, dim)
        self.linear1.weight.data.fill_(1.0)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear2 = nn.Linear(dim, dim)
        self.linear2.weight.data.fill_(1.0)

    def forward(self, x):
        w_pe = self.pe[:x.size(-1), :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = w_pe.permute(1, 2, 0).unsqueeze(2)

        h_pe = self.pe[:x.size(-2), :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = h_pe.permute(1, 2, 0).unsqueeze(3)

        x = x + w_pe + h_pe
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3)).permute(2, 0, 1)

        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        w0 = torch.randn(vocab, d_model).mul_(d_model ** -0.5)
        self.embedding.weight.data.copy_(w0)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)


class Beam:
    """Beam search"""

    def __init__(self, size, device=False):
        self.size = size
        self._done = False
        self.scores = torch.zeros((size,), dtype=torch.float32)
        self.all_scores = []
        self.prev_ks = []
        self.next_ys = [torch.full((size,), 0, dtype=torch.int64)]
        self.next_ys[0][0] = 2

    def get_current_state(self):
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done
    def advance(self, word_prob):
        # If this is the first time step
        if len(self.prev_ks) == 0:
            beam_scores = word_prob[0]
        else:
            # Add the current scores to the cumulative beam scores
            beam_scores = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)

        # Flatten the beam scores to find the top scores
        flat_beam_scores = beam_scores.view(-1)

        # Get the top scores and their corresponding indices
        top_scores, top_indices = flat_beam_scores.topk(self.size, dim=0, largest=True, sorted=True)

        # Update scores
        self.scores = top_scores
        self.all_scores.append(self.scores)

        # Get beam and word indices
        prev_k = top_indices // word_prob.size(1)
        next_y = top_indices % word_prob.size(1)

        # Append previous indices and the next word indices to the beam
        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)

        # Check if any sequence is done (e.g., reached end-of-sentence token)
        if next_y[0].item() == 3:  # Assuming 3 is the <EOS> token index
            self._done = True

    def get_hypothesis(self, k):
        """Retrieve the hypothesis (sequence of words) for a given beam index."""
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k].item())
            k = self.prev_ks[j][k]
        return hyp[::-1]

    def get_tentative_hypothesis(self):
        """Retrieve all hypotheses at the current time step."""
        hyps = []
        for i in range(self.size):
            hyps.append(self.get_hypothesis(i))
        return hyps
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.postprocess import CTCLabelDecode, NRTRLabelDecode

class Transformer(nn.Module):
    """A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". 

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multihead attention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        beam_size=0,
        num_decoder_layers=6,
        max_len=25,
        dim_feedforward=1024,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        in_channels=0,
        out_channels=0,
        scale_embedding=True,
    ):
        super(Transformer, self).__init__()
        self.text_dec = NRTRLabelDecode(character_dict_path='utils/en_dict.txt', use_space_char=True)
        self.out_channels = out_channels + 1
        self.max_len = max_len
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model
        )

        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model,
                        nhead,
                        dim_feedforward,
                        attention_dropout_rate,
                        residual_dropout_rate,
                        with_self_attn=True,
                        with_cross_attn=False,
                    )
                    for i in range(num_encoder_layers)
                ]
            )
        else:
            self.encoder = None

        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    attention_dropout_rate,
                    residual_dropout_rate,
                    with_self_attn=True,
                    with_cross_attn=True,
                )
                for i in range(num_decoder_layers)
            ]
        )

        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        
        self.tgt_word_prj = nn.Linear(self.out_channels, d_model , bias=False)
        w0 = np.random.normal(
            0.0, d_model**-0.5, (self.out_channels, d_model)
        ).astype(np.float32)
        self.tgt_word_prj.weight.data = torch.tensor(w0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)

        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src  # B N C
        else:
            memory = src  # B N C
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
        output = tgt
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, targets=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        Shape:
            - src: :math:`(B, sN, C)`.
            - tgt: :math:`(B, tN, C)`.
        Examples:
            >>> output = transformer_model(src, tgt)
        """
        # import pdb; pdb.set_trace()
        # self.forward_beam(src)
        # out = self.forward_test(src)
        # self.text_dec(out[0])
        if self.training:
            max_len = targets[1].max().item()
            tgt = targets[0][:, : 2 + max_len]
            return self.forward_train(src, tgt)
        else:
            if self.beam_size > 0:
                return self.forward_beam(src)
            else:
                return self.forward_test(src)

    def forward_test(self, src):
        bs = src.shape[0]
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src  # B N C
        else:
            memory = src
        dec_seq = torch.full((bs, 1), 2, dtype=torch.int64).cuda()
        dec_prob = torch.full((bs, 1), 1.0, dtype=torch.float32).cuda()
        for len_dec_seq in range(1, self.max_len):
            dec_seq_embed = self.embedding(dec_seq)
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.shape[1]).to(dec_seq.device)
            tgt = dec_seq_embed
            for decoder_layer in self.decoder:
                tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
            dec_output = tgt
            dec_output = dec_output[:, -1, :]
            word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=-1)
            preds_idx = word_prob.argmax(dim=-1)
            if torch.equal(preds_idx, torch.full(preds_idx.shape, 3, dtype=torch.int64).cuda()):
                break
            preds_prob = word_prob.max(dim=-1).values
            dec_seq = torch.cat([dec_seq, preds_idx.view(-1, 1)], dim=1)
            dec_prob = torch.cat([dec_prob, preds_prob.view(-1, 1)], dim=1)
        return [dec_seq, dec_prob]

    def forward_beam(self, images):
        """Translation work in one batch"""

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """Indicate the position of an instance in a tensor."""
            return {
                inst_idx: tensor_position
                for tensor_position, inst_idx in enumerate(inst_idx_list)
            }

        def collect_active_part(
            beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm
        ):
            """Collect tensor parts associated to active instances."""

            beamed_tensor_shape = beamed_tensor.shape
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (
                n_curr_active_inst * n_bm,
                beamed_tensor_shape[1],
                beamed_tensor_shape[2],
            )

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor[torch.tensor(curr_active_inst_idx)]
            beamed_tensor = beamed_tensor.view(new_shape)

            return beamed_tensor

        def collate_active_info(
            src_enc, inst_idx_to_position_map, active_inst_idx_list
        ):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.

            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [
                inst_idx_to_position_map[k] for k in active_inst_idx_list
            ]
            active_inst_idx = torch.tensor(active_inst_idx, dtype=torch.int64)
            active_src_enc = collect_active_part(
                src_enc.permute(1, 0, 2), active_inst_idx, n_prev_active_inst, self.beam_size
            ).permute(1, 0, 2)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )
            return active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
            inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm
        ):
            """Decode and update beam status, and then return active beam idx"""

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [
                    b.get_current_state() for b in inst_dec_beams if not b.done
                ]
                dec_partial_seq = torch.stack(dec_partial_seq)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
                dec_seq = self.embedding(dec_seq.long().int().cuda())
                dec_seq = self.positional_encoding(dec_seq)
                tgt_mask = self.generate_square_subsequent_mask(dec_seq.shape[1]).to(dec_seq.device)
                tgt = dec_seq
                for decoder_layer in self.decoder:
                    tgt = decoder_layer(tgt, enc_output, self_mask=tgt_mask)
                dec_output = tgt
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            active_inst_idx = []
            n_active_inst = 0
            dec_partial_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)

            word_prob = predict_word(dec_partial_seq, enc_output, len(dec_partial_seq), self.beam_size)

            # Process the probability distribution and create new beams
            for i in range(len(inst_dec_beams)):
                if not inst_dec_beams[i].done:
                    inst_idx = inst_dec_beams[i].inst_idx
                    word_prob_i = word_prob[n_active_inst]

                    # Gather beam predictions
                    top_k_prob, top_k_idx = torch.topk(word_prob_i, self.beam_size)
                    for k in range(self.beam_size):
                        new_inst_dec_beam = inst_dec_beams[i].clone()
                        new_inst_dec_beam.add_word_and_prob(top_k_idx[k].item(), top_k_prob[k].item())
                        active_inst_idx.append(new_inst_dec_beam)

                    n_active_inst += 1

            return active_inst_idx

        # Prepare for beam search
        bs = images.shape[0]
        if self.encoder is not None:
            images = self.positional_encoding(images)
            for encoder_layer in self.encoder:
                images = encoder_layer(images)
            memory = images
        else:
            memory = images

        # Initialize beams
        inst_dec_beams = [BeamSearchInstance() for _ in range(bs)]
        dec_seq_len = 1

        # Decode until we reach max length or all beams are done
        while dec_seq_len < self.max_len and any(not b.done for b in inst_dec_beams):
            inst_dec_beams = beam_decode_step(inst_dec_beams, dec_seq_len, memory, get_inst_idx_to_tensor_position_map([b.inst_idx for b in inst_dec_beams]), self.beam_size)
            dec_seq_len += 1

        # Collect the results
        results = [b.get_best_sequence() for b in inst_dec_beams]
        return results

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class Embeddings(nn.Module):
    """Embedding layer for the transformer model."""
    def __init__(self, d_model, vocab, padding_idx=0, scale_embedding=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.scale_embedding = scale_embedding

    def forward(self, x):
        embed = self.embedding(x)
        if self.scale_embedding:
            return embed * (self.embedding.embedding_dim ** 0.5)
        return embed

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""
    def __init__(self, dropout=0.1, dim=512, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension
        self.encoding = self.encoding.cuda()
    def forward(self, x):
        x = x + self.encoding[:, :x.size(1)]
        return self.dropout(x)


class BeamSearchInstance:
    """A class to handle the state of each beam in beam search."""
    def __init__(self):
        self.inst_idx = None
        self.done = False
        self.dec_seq = []
        self.prob = []

    def add_word_and_prob(self, word, prob):
        self.dec_seq.append(word)
        self.prob.append(prob)

    def get_current_state(self):
        return torch.tensor(self.dec_seq)

    def get_best_sequence(self):
        return self.dec_seq

    def clone(self):
        clone = BeamSearchInstance()
        clone.inst_idx = self.inst_idx
        clone.done = self.done
        clone.dec_seq = self.dec_seq.copy()
        clone.prob = self.prob.copy()
        return clone