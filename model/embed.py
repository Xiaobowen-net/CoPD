import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.pe[:, : x.size(1), :]
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = self.value_embedding(x) + self.position_embedding(x)
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()

        self.value_embedding = nn.Linear(c_in, d_model)
        # self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        x = self.value_embedding(x)
        return self.dropout(x)

class RotaryEncoding(torch.nn.Module):
    def __init__(self, dim, *, max_length=1024, base=10000):
        super().__init__()

        sin, cos = self._compute_buffers(dim, base, max_length)

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

        self.sin: torch.Tensor
        self.cos: torch.Tensor

    @staticmethod
    def _compute_buffers(dim, base, length):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(length, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = freqs.repeat(1, 2)  # repeat each frequency twice
        freqs = freqs[:, None, :]  # add batch dim

        return freqs.sin(), freqs.cos()

    def forward(self, queries, keys):
        return apply_rotary(queries, keys, self.sin, self.cos)


def rotate_half(x):
    middle = x.size(-1) // 2

    x1 = x[..., :middle]
    x2 = x[..., middle:]

    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x, sin, cos):
    return x * cos[: x.size(0)] + rotate_half(x) * sin[: x.size(0)]


@torch.jit.script
def apply_rotary(queries, keys, sin, cos):
    return _apply_rotary(queries, sin, cos), _apply_rotary(keys, sin, cos)

    

