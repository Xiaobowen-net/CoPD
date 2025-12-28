import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
from einops import rearrange, reduce, repeat
from .embed import RotaryEncoding

def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    """
    计算斜率
    """
    # _n_heads是与n_heads接近的2的次数，例如：n_heads为5/6/7时，_n_heads为8
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    # m是alibi_bias_max/_n_heads到alibi_bias_max的等差数列
    m = m.mul(alibi_bias_max / _n_heads)
    # 计算斜率
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)

def build_alibi_bias(n_heads, seq_len, full=False, alibi_bias_max=1, device=None):
    """
    构建alibi注意力偏差
    """
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, 1, seq_len)
    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias

class DAC_structure(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05,
                 output_attention=False):
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries, keys, values, patch_index, tau=None, delta=None):

        B, L, H, E = queries.shape  # batch_size*channel, patch_size, n_head, d_model/n_head
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        scores = torch.einsum("blhe,bshe->bhls", queries,keys)  # batch*ch, nheads, p_size, p_size
        # scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
        attn = scale * scores
        series = self.dropout(torch.softmax(attn, dim=-1))  # B*N h pn pn

        V = torch.einsum("bhls,bshd->blhd", series, values)  # B*N pn h d//h

        if patch_index == -1:
            pass
        else:
            # Upsampling # B*N pn pn # B*N ps ps
            if series.shape[-1] == self.patch_size[patch_index]:
                series = repeat(series, 'b l m n -> b l (m repeat_m) (n repeat_n)',
                                        repeat_m=self.window_size // self.patch_size[patch_index],
                                        repeat_n=self.window_size // self.patch_size[patch_index])
                series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean',
                                reduce_b=self.window_size // self.patch_size[patch_index])
            else:
                series = series.repeat(1, 1, self.patch_size[patch_index], self.patch_size[patch_index])
                series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.patch_size[patch_index])

        if self.output_attention:
            # print(series)
            # print(series.shape)
            return (V.contiguous(), series)
            # return v_patch_size, v_patch_num, series_patch_size, series_patch_num
        else:
            return (None)

class ALiBiAttention(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05,
                 output_attention=False):
        super(ALiBiAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries, keys, values, patch_index, tau=None, delta=None):

        B, L, H, E = queries.shape  # batch_size*channel, patch_size, n_head, d_model/n_head
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        scores = torch.einsum("blhe,bshe->bhls", queries,keys)  # batch*ch, nheads, p_size, p_size
        # scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
        attn = scale * scores
        bias = build_alibi_bias(H, L).cuda()
        attn = attn + bias

        series = self.dropout(torch.softmax(attn, dim=-1))  # B*N h pn pn

        V = torch.einsum("bhls,bshd->blhd", series, values)  # B*N pn h d//h

        if patch_index == -1:
            pass
        else:
            # Upsampling # B*N pn pn # B*N ps ps
            if series.shape[-1] == self.patch_size[patch_index]:
                series = repeat(series, 'b l m n -> b l (m repeat_m) (n repeat_n)',
                                        repeat_m=self.window_size // self.patch_size[patch_index],
                                        repeat_n=self.window_size // self.patch_size[patch_index])
                series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean',
                                reduce_b=self.window_size // self.patch_size[patch_index])

                # series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)


            else:
                series = series.repeat(1, 1, self.patch_size[patch_index], self.patch_size[patch_index])
                series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.patch_size[patch_index])

                # series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)

        if self.output_attention:
            return (V.contiguous(), series)
            # return v_patch_size, v_patch_num, series_patch_size, series_patch_num
        else:
            return (None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.window_size = win_size
        self.n_heads = n_heads
        self.channel = channel

        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x, patch_index, tau=None, delta=None):
        # patch_num
        B, L, M = x.shape # B*N pn d
        H = self.n_heads
        queries, keys, values = x, x, x
        queries = self.patch_query_projection(queries).view(B, L, H, -1)
        keys = self.patch_key_projection(keys).view(B, L, H, -1)
        values = self.patch_value_projection(values).view(B, L, H, -1)

        V, series = self.inner_attention(queries, keys, values,
                                         patch_index, tau=tau, delta=delta)
        V = self.out_projection(V.reshape(B, L, -1)) # B*N pn d

        return V, series

class RelativePositionSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, patch_size, win_size):
        super(RelativePositionSelfAttention, self).__init__()
        self.d_model = d_model  # 输入序列的维度
        self.nhead = nhead  # 注意力头的数量
        self.window_size = win_size # 输入序列的长度
        self.patch_size = patch_size
        self.d_k = d_model // nhead  # 每个注意力头的维度


        # 定义线性层用于计算 Q, K, V
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        # 定义线性层用于输出
        self.fc = nn.Linear(d_model, d_model)

        # 定义相对位置编码的 Embedding 层
        self.relative_pos_emb = nn.Embedding(2 * win_size - 1, self.d_k)

    def forward(self, x, patch_index, tau=None, delta=None):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        # 计算 Q, K, V
        Q = self.WQ(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.WK(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.WV(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # 计算相对位置编码
        pos_indices = torch.arange(0, 2 * seq_len - 1).cuda()
        pos_embeddings = self.relative_pos_emb(pos_indices)

        # 计算注意力得分
        S = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)

        # 计算相对位置得分
        S_rel = torch.matmul(Q.unsqueeze(-2).cuda(), pos_embeddings.unsqueeze(0).transpose(-1, -2).cuda())
        S_rel = S_rel.view(batch_size, self.nhead, seq_len, 2 * seq_len - 1)

        # 将相对位置得分平移以对齐序列
        S_rel_shift = torch.zeros_like(S).cuda()
        for i in range(seq_len):
            for j in range(seq_len):
                S_rel_shift[..., i, j] = S_rel[..., i, j - i + seq_len - 1]

        # 计算注意力权重
        A = torch.softmax(S + S_rel_shift, dim=-1)

        # 应用注意力权重
        x = torch.matmul(A, V)

        # 拼接多个注意力头
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        series = A
        if patch_index == -1:
            pass
        else:
            # Upsampling # B*N pn pn # B*N ps ps
            if series.shape[-1] == self.patch_size[patch_index]:
                series = repeat(series, 'b l m n -> b l (m repeat_m) (n repeat_n)',
                                        repeat_m=self.window_size // self.patch_size[patch_index],
                                        repeat_n=self.window_size // self.patch_size[patch_index]).cuda()
                series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean',
                                reduce_b=self.window_size // self.patch_size[patch_index])
            else:
                series = series.repeat(1, 1, self.patch_size[patch_index], self.patch_size[patch_index]).cuda()
                series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.patch_size[patch_index])

        # 返回最终结果
        return self.fc(x), series

class RoAttentionLayer(nn.Module):
    def __init__(self, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        super(RoAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.patch_size = patch_size
        self.window_size = win_size
        self.n_heads = n_heads
        self.channel = channel

        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        # self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim, device):
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)

        # (output_dim//2)
        # 即公式里的i, i的范围是 [0,d/2]
        ids = torch.arange(0, output_dim // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / output_dim)

        # (max_len, output_dim//2)
        # 即公式里的：pos / (10000^(2i/d))
        embeddings = position * theta

        # (max_len, output_dim//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # (bs, head, max_len, output_dim//2, 2)
        # 在bs维度重复，其他维度都是1不重复
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

        # (bs, head, max_len, output_dim)
        # reshape后就是：偶数sin, 奇数cos了
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def RoPE(self, q, k):
        # q,k: (bs, head, max_len, output_dim)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        output_dim = q.shape[-1]

        # (bs, head, max_len, output_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

        # cos_pos,sin_pos: (bs, head, max_len, output_dim)
        # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

        # q,k: (bs, head, max_len, output_dim)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)  # reshape后就是正负交替了
        # 更新qw, *对应位置相乘
        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)
        # 更新kw, *对应位置相乘
        k = k * cos_pos + k2 * sin_pos

        return q, k

    def attention(self, q, k, v, dropout=None, use_RoPE=True):
        # q.shape: (bs, head, seq_len, dk)
        # k.shape: (bs, head, seq_len, dk)
        # v.shape: (bs, head, seq_len, dk)

        # if use_RoPE:
            # 使用RoPE进行位置编码
        q, k = self.RoPE(q, k)

        d_k = k.size()[-1]

        # 计算注意力权重
        # (bs, head, seq_len, seq_len)
        att_logits = torch.matmul(q, k.transpose(-2, -1))
        att_logits /= math.sqrt(d_k)

        # (bs, head, seq_len, seq_len)
        att_scores = F.softmax(att_logits, dim=-1)
        if dropout is not None:
            # 对权重进行dropout
            att_scores = dropout(att_scores)

        # 注意力权重与值的加权求和
        # (bs, head, seq_len, seq_len) * (bs, head, seq_len, dk) = (bs, head, seq_len, dk)
        return torch.matmul(att_scores, v), att_scores

    def forward(self, x, patch_index, tau=None, delta=None):
        # patch_num
        B, L, M = x.shape # B*N pn d
        H = self.n_heads
        queries, keys, values = x, x, x
        queries = self.patch_query_projection(queries).view(B, L, H, -1).permute(0,2,1,3)
        keys = self.patch_key_projection(keys).view(B, L, H, -1).permute(0,2,1,3)
        values = self.patch_value_projection(values).view(B, L, H, -1).permute(0,2,1,3)

        res, series = self.attention(queries, keys, values, dropout=None, use_RoPE=True)
        res = res.permute(0,2,1,3)
        res = self.out_projection(res.reshape(B, L, -1)) # B*N pn d

        if patch_index == -1:
            pass
        else:
            # Upsampling # B*N pn pn # B*N ps ps
            if series.shape[-1] == self.patch_size[patch_index]:
                series = repeat(series, 'b l m n -> b l (m repeat_m) (n repeat_n)',
                                repeat_m=self.window_size // self.patch_size[patch_index],
                                repeat_n=self.window_size // self.patch_size[patch_index]).cuda()
                series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean',
                                reduce_b=self.window_size // self.patch_size[patch_index])
            else:
                series = series.repeat(1, 1, self.patch_size[patch_index], self.patch_size[patch_index]).cuda()
                series = reduce(series, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.patch_size[patch_index])

        return res, series