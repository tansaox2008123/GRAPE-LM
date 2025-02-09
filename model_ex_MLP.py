# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math


def get_attention_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(-1) is PAD token
    pad_attn_mask = seq_k.data.eq(-1).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    # 最终得到的应该是一个最后n列为1的矩阵，即K的最后n个token为PAD。
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


def get_attention_pad_mask_v2(seq_q):
    batch_size, len_q = seq_q.size()
    # batch_size, len_k = seq_k.size()
    len_k = 1
    # pad_attn_mask = torch.from_numpy(np.zeros((batch_size, len_q, len_k), dtype=bool)).byte()#.cuda()
    pad_attn_mask = torch.zeros((batch_size, len_q, len_k)).byte().cuda()
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    # 最终得到的应该是一个最后n列为1的矩阵，即K的最后n个token为PAD。
    return pad_attn_mask  # .expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


# Sequence Mask 屏蔽未来词，构成上三角矩阵
def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    # attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    # np.triu()返回一个上三角矩阵，自对角线k以下元素全部置为0，k=0为主对角线
    # subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.triu(torch.ones(seq.size(0), seq.size(1), seq.size(1)), 1).byte().cuda()  # 生成一个上三角矩阵
    # 如果没转成byte，这里默认是Double(float64)，占据的内存空间大，浪费，用byte就够了
    # subsequence_mask = torch.from_numpy(subsequence_mask).byte()#.cuda()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# Attention 计算
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q:batch_size, n_heads, len_q, d_k
        # K:batch_size, n_heads, len_k, d_k
        # V:batch_size, n_heads, len_v, d_v
        # attn_mask:batch_size, n_heads, seq_len, seq_len

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # scores:batch_size, n_heads, len_q, len_k
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
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
        # attention_mask:batch_size,len_q,len_k
        # Q:batch_size,len_q,d_model
        # Q:batch_size,len_k,d_model
        # Q:batch_size,len_k,d_model
        residual, batch_size = Q, Q.size(0)

        # q_s:batch_size,n_heads,len_q,d_k
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k_s:batch_size,n_heads,len_k,d_k
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v_s:batch_size,n_heads,len_k,d_v
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # attention_mask:batch_size,len_q,len_k ----> batch_size,n_heads,len_q,len_k
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attention = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attention_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)

        return self.layer_norm(output + residual), attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # 生成一个形状为[max_len,d_model]的全为0的tensor
        pe = torch.zeros(max_len, d_model)
        # position:[max_len,1]，即[5000,1]，这里插入一个维度是为了后面能够进行广播机制然后和div_term直接相乘
        # 注意，要理解一下这里position的维度。每个pos都需要512个编码。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 共有项，利用指数函数e和对数函数log取下来，方便计算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 这里position * div_term有广播机制，因为div_term的形状为[d_model/2],即[256],符合广播条件，广播后两个tensor经过复制，形状都会变成[5000,256]，*表示两个tensor对应位置处的两个元素相乘
        # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置赋值给pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # 同理，这里是奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)
        # 上面代码获取之后得到的pe:[max_len*d_model]

        # 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 定一个缓冲区，其实简单理解为这个参数不更新就可以，但是参数仍然作为模型的参数保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # 这里的self.pe是从缓冲区里拿的
        # 切片操作，把pe第一维的前seq_len个tensor和x相加，其他维度不变
        # 这里其实也有广播机制，pe:[max_len,1,d_model]，第二维大小为1，会自动扩张到batch_size大小。
        # 实现词嵌入和位置编码的线性相加
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)


class Encoder(nn.Module):
    def __init__(self, embed_dim, model_dim, dropout):
        super(Encoder, self).__init__()
        self.dense = nn.Sequential(nn.Linear(embed_dim, model_dim),
                                   nn.ReLU(),
                                   # nn.Linear(model_dim, model_dim),
                                   nn.BatchNorm1d(model_dim),
                                   nn.Dropout(dropout))

    def forward(self, inputs):
        '''
        inputs: [batch_size, embed_dim]
        '''
        outputs = self.dense(inputs)
        return outputs


class Predictor(nn.Module):
    def __init__(self, hidd_feat_dim, model_dim, dropout):
        super(Predictor, self).__init__()

        self.clf = nn.Sequential(nn.Dropout(2.0 * dropout),
                                 nn.Linear(hidd_feat_dim, model_dim),
                                 nn.ReLU(),
                                 nn.Dropout(2.0 * dropout),
                                 nn.Linear(model_dim, 1))

    def forward(self, inputs):
        '''
        inputs: [batch_size, feature_dim]
        '''
        outputs = self.clf(inputs)
        return outputs


class SimpleDecoder(nn.Module):
    def __init__(self, hidd_feat_dim, model_dim):
        super(SimpleDecoder, self).__init__()
        self.dense = nn.Sequential(nn.Linear(hidd_feat_dim, model_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidd_feat_dim, model_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidd_feat_dim, 100))

    def forward(self, inputs):
        '''
        inputs: [batch_size, src_len]
        '''
        output = self.dense(inputs)
        output = output.view(-1, 20, 5)

        return output


class Decoder(nn.Module):
    def __init__(self, tgt_size, n_layers, d_model, d_ff, d_k, d_v, n_heads, dropout):
        super(Decoder, self).__init__()
        # self.tgt_emb = nn.Embedding(tgt_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])

    # dec_inputs:batch_size,target_len
    def forward(self, dec_inputs, enc_outputs):
        # 得到的dec_outputs维度为  dec_outputs:batch_size,tgt_len,d_model
        dec_outputs = self.tgt_emb(dec_inputs)
        # 得到的dec_outputs维度为  dec_outputs:batch_size,tgt_len,d_model
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)
        # print(dec_outputs)

        # 获取自注意力的pad_mask，1表示被mask
        dec_self_attention_pad_mask = get_attention_pad_mask(dec_inputs, dec_inputs)
        # print(dec_self_attention_pad_mask)

        # 获取上三角矩阵，即让注意力机制看不到未来的单词，1表示被mask
        dec_self_attention_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # print(dec_self_attention_subsequent_mask)

        # 两个mask矩阵相加，大于0的为1，不大于0的为0，即屏蔽了pad的信息，也屏蔽了未来时刻的信息，为1的在之后就会被fill到无限小
        dec_self_attention_mask = torch.gt((dec_self_attention_subsequent_mask + dec_self_attention_pad_mask), 0)

        # print(enc_outputs.shape)
        # print(enc_outputs.unsqueeze(1).shape)
        enc_outputs = enc_outputs.unsqueeze(1)

        dec_enc_attention_mask = get_attention_pad_mask_v2(dec_inputs)
        # print(dec_enc_attention_mask)

        dec_self_attentions, dec_enc_attentions = [], []
        for layer in self.layers:
            # print(enc_outputs.shape)
            dec_outputs, dec_self_attention, dec_enc_attention = layer(dec_outputs, enc_outputs,
                                                                       dec_self_attention_mask, dec_enc_attention_mask)
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
        # 自注意力机制Q,K,V都是dec_inputs
        dec_outputs, dec_self_attention = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs,
                                                                  dec_self_attention_mask)
        # 这里用dec_outputs作为Q，enc_outputs作为K和V
        dec_outputs, dec_enc_attention = self.dec_enc_attention(dec_outputs, enc_outputs, enc_outputs,
                                                                dec_enc_attention_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attention, dec_enc_attention


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab=5):
        super(Generator, self).__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.ReLU(),
                                  nn.Linear(d_model, vocab))

    def forward(self, x):
        return self.proj(x)


class FullModel(nn.Module):
    def __init__(self, input_dim, model_dim, tgt_size, n_declayers, d_ff, d_k_v, n_heads, dropout):
        super(FullModel, self).__init__()
        self.encoder = Encoder(embed_dim=input_dim,
                               model_dim=model_dim,
                               dropout=dropout)

        self.predictor = Predictor(hidd_feat_dim=model_dim,
                                   model_dim=model_dim,
                                   dropout=dropout)

        self.decoder = Decoder(tgt_size=tgt_size,
                               n_layers=n_declayers,
                               d_model=model_dim,
                               d_ff=d_ff,
                               d_k=d_k_v,
                               d_v=d_k_v,
                               n_heads=n_heads,
                               dropout=dropout)

        # self.projection = nn.Linear(d_model, tgt_size, bias=False)

        self.generator = Generator(model_dim, vocab=tgt_size)

    def forward(self, rna_emds, rna_seq):
        '''
        rna_emds: [batch_size, feat_dim]
        rna_seq: [batch_size, tgt_len]
        '''
        # hidd_feats = self.encoder(rna_emds)  # [batch_size, model_dim]
        # bind_scores = self.predictor(hidd_feats)  # [batch_size, 1]
        dec_outputs, _, _ = self.decoder(rna_seq, rna_emds)
        pred_seq = self.generator(dec_outputs)

        return pred_seq


class SimpleModel(nn.Module):
    def __init__(self, input_dim, model_dim, dropout):
        super(SimpleModel, self).__init__()
        self.encoder = Encoder(embed_dim=input_dim,
                               model_dim=model_dim,
                               dropout=dropout)

        self.predictor = Predictor(hidd_feat_dim=model_dim,
                                   model_dim=model_dim,
                                   dropout=dropout)

        self.decoder = SimpleDecoder(hidd_feat_dim=model_dim,
                                     model_dim=model_dim)

    def forward(self, rna_emds):
        '''
        rna_emds: [batch_size, feat_dim]
        rna_seq: [batch_size, tgt_len]
        '''
        hidd_feats = self.encoder(rna_emds)  # [batch_size, model_dim]
        bind_scores = self.predictor(hidd_feats)  # [batch_size, 1]
        pred_seq = self.decoder(hidd_feats)

        return bind_scores, pred_seq
