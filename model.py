#  -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math

base_dict = {1: "A", 2: "C", 3: "G", 4: "U"}


def decode_rna_sequences(one_hot_vectors):
    sequences = []
    for row in one_hot_vectors:
        sequence = "".join([base_dict[int(num)] for num in row.cpu().numpy()])
        sequences.append(sequence)
    return sequences


def rna_seq_embbding(OriginSeq, batch_converter, EmbeddingModel, device):
    EmbeddingModel = EmbeddingModel.to(device)
    batch_labels, batch_strs, batch_tokens = batch_converter(OriginSeq)
    batch_tokens = batch_tokens.to(device)

    tmp = []

    with torch.no_grad():
        results = EmbeddingModel(batch_tokens, repr_layers=[12])
    token_embeddings = results["representations"][12]

    return token_embeddings


def get_attention_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(-1).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attention_pad_mask_v2(seq_q):
    batch_size, len_q = seq_q.size()
    len_k = 1
    pad_attn_mask = torch.zeros((batch_size, len_q, len_k)).byte().cuda()
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    subsequence_mask = torch.triu(torch.ones(seq.size(0), seq.size(1), seq.size(1)), 1).byte().cuda()  # 生成一个上三角矩阵
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask.to(torch.bool), -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attention_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attention = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attention_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)

        return self.layer_norm(output + residual), attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)


class Adapter(nn.Module):
    def __init__(self, embed_dim, model_dim, dropout):
        super(Adapter, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(embed_dim, model_dim),
            nn.ReLU(),
            # nn.Linear(model_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        outputs = self.dense(inputs)
        return outputs


class AdapterStack(nn.Module):
    def __init__(self, num_layers, embed_dim, bottleneck_dim, dropout):
        super(AdapterStack, self).__init__()
        layers = []

        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else bottleneck_dim
            layers.append(Adapter(input_dim, bottleneck_dim, dropout))

        self.adapters = nn.Sequential(*layers)

    def forward(self, x):
        return self.adapters(x)


class Adapter_layernorm(nn.Module):
    def __init__(self, embed_dim, model_dim, dropout):
        super(Adapter_layernorm, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(embed_dim, model_dim),
            nn.ReLU(),
            # nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.size(0), 20, -1)
        outputs = self.dense(inputs)
        return outputs


class Predictor(nn.Module):
    def __init__(self, hidd_feat_dim, model_dim, dropout):
        super(Predictor, self).__init__()

        self.clf = nn.Sequential(
            nn.Dropout(2.0 * dropout),
            nn.Linear(hidd_feat_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(2.0 * dropout),
            nn.Linear(model_dim, 1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.clf(inputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, tgt_size, n_layers, d_model, d_ff, d_k, d_v, n_heads, dropout):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        dec_self_attention_pad_mask = get_attention_pad_mask(dec_inputs, dec_inputs)

        dec_self_attention_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attention_mask = torch.gt((dec_self_attention_subsequent_mask + dec_self_attention_pad_mask), 0)

        enc_outputs = enc_outputs.unsqueeze(1)

        dec_enc_attention_mask = get_attention_pad_mask_v2(dec_inputs)

        dec_self_attentions, dec_enc_attentions = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attention, dec_enc_attention = layer(
                dec_outputs,
                enc_outputs,
                dec_self_attention_mask,
                dec_enc_attention_mask,
            )
            dec_self_attentions.append(dec_self_attention)
            dec_enc_attentions.append(dec_enc_attention)
        return dec_outputs, dec_self_attentions, dec_enc_attentions


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.dec_enc_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attention_mask, dec_enc_attention_mask):
        dec_outputs, dec_self_attention = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs,
                                                                  dec_self_attention_mask)

        dec_outputs, dec_enc_attention = self.dec_enc_attention(dec_outputs, enc_outputs, enc_outputs,
                                                                dec_enc_attention_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attention, dec_enc_attention


class Generator(nn.Module):
    def __init__(self, d_model, vocab=5):
        super(Generator, self).__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, vocab))

    def forward(self, x):
        return self.proj(x)


class FullModel_guidance(nn.Module):
    def __init__(self, input_dim, model_dim, tgt_size, n_declayers, d_ff, d_k_v, n_heads, dropout):
        super().__init__()

        self.adapter = Adapter(embed_dim=input_dim, model_dim=model_dim, dropout=dropout)

        self.predictor = Predictor(hidd_feat_dim=model_dim, model_dim=model_dim, dropout=dropout)

        self.decoder = Decoder(
            tgt_size=tgt_size,
            n_layers=n_declayers,
            d_model=model_dim,
            d_ff=d_ff,
            d_k=d_k_v,
            d_v=d_k_v,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.generator = Generator(model_dim, vocab=tgt_size)

    def forward(self, rna_emds, rna_seq):
        hidd_feats = self.adapter(rna_emds)
        bind_scores = self.predictor(hidd_feats)
        dec_outputs, _, _ = self.decoder(rna_seq, hidd_feats)
        pred_seq = self.generator(dec_outputs)

        return bind_scores, pred_seq


class FullModel_guidance_stack(nn.Module):
    def __init__(self, num_layers, input_dim, model_dim, tgt_size, n_declayers, d_ff, d_k_v, n_heads, dropout):
        super().__init__()

        self.adapter = AdapterStack(num_layers=num_layers, embed_dim=input_dim, bottleneck_dim=model_dim,
                                    dropout=dropout)

        self.predictor = Predictor(hidd_feat_dim=model_dim, model_dim=model_dim, dropout=dropout)

        self.decoder = Decoder(
            tgt_size=tgt_size,
            n_layers=n_declayers,
            d_model=model_dim,
            d_ff=d_ff,
            d_k=d_k_v,
            d_v=d_k_v,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.generator = Generator(model_dim, vocab=tgt_size)

    def forward(self, rna_emds, rna_seq):
        hidd_feats = self.adapter(rna_emds)
        bind_scores = self.predictor(hidd_feats)
        dec_outputs, _, _ = self.decoder(rna_seq, hidd_feats)
        pred_seq = self.generator(dec_outputs)

        return bind_scores, pred_seq


class FullModelGru(nn.Module):
    def __init__(self, input_dim, model_dim, vocab_size, num_gru_layers, dropout):
        super().__init__()

        self.adapter = Adapter_layernorm(embed_dim=input_dim, model_dim=model_dim, dropout=dropout)

        self.predictor = Predictor(hidd_feat_dim=int(model_dim * 20), model_dim=model_dim, dropout=dropout)

        self.embed = nn.Embedding(vocab_size, model_dim)

        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )

        self.generator = Generator(model_dim, vocab=vocab_size)

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=device)

    def forward(self, seq_rep, seq):
        hidd_feats = self.adapter(seq_rep)

        flat_feats = hidd_feats.reshape(hidd_feats.size(0), -1)
        bind_scores = self.predictor(flat_feats)

        hidden = self.init_hidden(seq_rep.size(0), seq_rep.device)
        _, hidden = self.gru(hidd_feats, hidden)
        x = self.embed(seq)
        x, _ = self.gru(x, hidden)

        pred_seq = self.generator(x)

        return bind_scores, pred_seq


class FullModelCNN(nn.Module):
    """
    No LLM
    feats: one-hot
    with guidance
    """

    def __init__(self, input_dim, model_dim, tgt_size, n_declayers, d_ff, d_k_v, n_heads, dropout):
        super(FullModelCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.2),
        )

        self.predictor = Predictor(hidd_feat_dim=int(model_dim), model_dim=model_dim, dropout=dropout)

        self.decoder = Decoder(
            tgt_size=tgt_size,
            n_layers=n_declayers,
            d_model=model_dim,
            d_ff=d_ff,
            d_k=d_k_v,
            d_v=d_k_v,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.generator = Generator(model_dim, vocab=tgt_size)

    def forward(self, x, rna_seq):
        # x: [batch_size, sequence_length, 4]
        # permute to [batch_size, 4, sequence_length]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.squeeze(-1)
        bind_scores = self.predictor(x)

        # x: [batch_size,model_dim]
        dec_outputs, _, _ = self.decoder(rna_seq, x)
        pred_seq = self.generator(dec_outputs)

        return bind_scores, pred_seq


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=128, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        context_vector = self.linear(last_hidden)

        return context_vector


class FullModel_guidance_LSTM(nn.Module):
    def __init__(self, input_dim, model_dim, tgt_size, n_declayers, d_ff, d_k_v, n_heads, dropout):
        super(FullModel_guidance_LSTM, self).__init__()

        self.lstm = LSTMEncoder(input_size=input_dim, hidden_size=256, output_size=model_dim, num_layers=1)

        self.predictor = Predictor(hidd_feat_dim=model_dim, model_dim=model_dim, dropout=dropout)

        self.decoder = Decoder(tgt_size=tgt_size, n_layers=n_declayers, d_model=model_dim, d_ff=d_ff, d_k=d_k_v,
                               d_v=d_k_v, n_heads=n_heads, dropout=dropout)
        self.generator = Generator(model_dim, vocab=tgt_size)

    def forward(self, rna_emds, rna_seq):
        hidd_feats = self.lstm(rna_emds)
        bind_scores = self.predictor(hidd_feats)
        dec_outputs, _, _ = self.decoder(rna_seq, hidd_feats)
        pred_seq = self.generator(dec_outputs)

        return bind_scores, pred_seq
