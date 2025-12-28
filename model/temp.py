import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import DAC_structure, AttentionLayer, RelativePositionSelfAttention, RoAttentionLayer, ALiBiAttention
from .embed import DataEmbedding, TokenEmbedding, DataEmbedding_inverted
from .RevIN import RevIN
from .encdec import series_decomp, series_decomp_multi, Projector
from tkinter import _flatten


def D_matrix(N):
    D = torch.zeros(N - 1, N)
    D[:, 1:] = torch.eye(N - 1)
    D[:, :-1] -= torch.eye(N - 1)
    return D


class hp_filter(nn.Module):
    """
        Hodrick Prescott Filter to decompose the series
    """

    def __init__(self, lamb):
        super(hp_filter, self).__init__()
        self.lamb = lamb

    def forward(self, x):
        x = x.permute(0, 2, 1)
        N = x.shape[1]
        D1 = D_matrix(N)
        D2 = D_matrix(N - 1)
        D = torch.mm(D2, D1).to(device='cuda')

        g = torch.matmul(torch.inverse(torch.eye(N).to(device='cuda') + self.lamb * torch.mm(D.T, D)), x)
        res = x - g
        # g = g.permute(0, 2, 1)
        # res = res.permute(0, 2, 1)
        return res, g


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, patch_index, tau=None, delta=None):
        new_x, series = self.attention(
            x, patch_index, tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), series


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, patch_index, tau=None, delta=None):
        series_list = []
        for attn_layer in self.attn_layers:
            # V1, V2, series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            V, series = attn_layer(x, patch_index, tau=tau, delta=delta)
            series_list.append(series)

        if self.norm is not None:
            V = self.norm(V)
        return V, series_list


class DCdetector(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3, 5, 7], channel=55,
                 d_ff=512, dropout=0.2, moving_avg=20, p_hidden_dims=[256, 256], p_hidden_layers=2,
                 activation='gelu', output_attention=True):
        super(DCdetector, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size

        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(moving_avg)

        self.embedding_channel_size = DataEmbedding_inverted(win_size, d_model, dropout)
        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)

        # Dual Attention Encoder

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout,
                                      output_attention=output_attention),
                        # win_size, patch_size, channel, n_heads, win_size),
                        d_model, patch_size, channel, n_heads, win_size),
                    d_model,
                    # win_size,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.encoder_List = nn.ModuleList()
        for patch_index, _ in enumerate(patch_size):
            self.encoder_List.append(
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout,
                                              output_attention=output_attention),
                                # win_size, patch_size, channel, n_heads, win_size),
                                d_model, patch_size, channel, n_heads, win_size),
                            d_model,
                            # win_size,
                            d_ff,
                            dropout=dropout,
                            activation=activation
                        ) for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model)
                )
            )

        self.encoder_ALi = nn.ModuleList()
        for patch_index, _ in enumerate(patch_size):
            self.encoder_ALi.append(
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                ALiBiAttention(win_size, patch_size, channel, False, attention_dropout=dropout,
                                               output_attention=output_attention),
                                d_model, patch_size, channel, n_heads, win_size),
                            d_model,
                            d_ff,
                            dropout=dropout,
                            activation=activation
                        ) for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model)
                )
            )

        self.encoder_ALi1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ALiBiAttention(win_size, patch_size, channel, False, attention_dropout=dropout,
                                       output_attention=output_attention),
                        d_model, patch_size, channel, n_heads, win_size),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.encoder_Re = Encoder(
            [
                EncoderLayer(
                    RelativePositionSelfAttention(
                        d_model, n_heads, patch_size, win_size
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.encoder_Ro = Encoder(
            [
                EncoderLayer(
                    RoAttentionLayer(d_model, patch_size, channel, n_heads, win_size),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # self.dropout_series = nn.Dropout(dropout)
        self.dropout_series1 = nn.Dropout(dropout)
        self.dropout_series2 = nn.Dropout(dropout)
        self.lin_res = nn.Linear(d_model, c_out)
        self.dropout_res = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.projection_c = nn.Linear(d_model, win_size, bias=True)

        # self.trend = nn.Sequential(
        #     nn.Linear(win_size, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, win_size)
        # )

        # self.revin_trend = RevIN(enc_in).cuda()

        # self.tau_learner = Projector(enc_in=enc_in, seq_len=win_size, hidden_dims=p_hidden_dims,
        #                              hidden_layers=p_hidden_layers, output_dim=1)
        # self.delta_learner = Projector(enc_in=enc_in, seq_len=win_size,
        #                                hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers,
        #                                output_dim=win_size)

        # self.Decomp1 = hp_filter(lamb=1)
        self.Decomp1 = hp_filter(lamb=6400)
        self.line_trend = nn.Linear(win_size, win_size)
        self.line_trend.weight = nn.Parameter((1 / win_size) * torch.ones([win_size, win_size]))

        self.channel_line1 = nn.Linear(d_model, 4 * d_model)
        self.channel_line2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x_enc):

        # x = x_enc
        # x, trend = self.Decomp1(x_enc.permute(0,2,1))
        # print(x.shape)
        # x = x.permute(0,2,1)

        x, trend = self.decompsition(x_enc)
        B, L, M = x.shape  # Batch win_size channel

        # revin_layer_L = RevIN(num_features=M) #M:90 -> 1
        # x = revin_layer_L(x, 'norm')

        # revin_layer_L = RevIN(num_features=L)  # M:90 -> 1
        # x = revin_layer_L(x.permute(0,2,1), 'norm').permute(0,2,1)

        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev

        x_c = self.embedding_channel_size(x)  # b n d
        V0, series0 = self.encoder(x_c, -1, tau=None, delta=None)
        # x_c1 = self.channel_line1(x_c)
        # V0 = self.channel_line2(x_c1)
        rec = self.projection_c(V0).permute(0, 2, 1)[:, :, :M]

        # rec0, rec = x, x

        # rec = rec * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        # rec = rec + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        # rec = revin_layer_L(rec, 'denorm')
        # rec = revin_layer_L(rec.permute(0, 2, 1), 'denorm').permute(0, 2, 1)

        trend_rec = self.line_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        rec0 = rec + trend_rec

        series_patch_mean = []
        V_list = []

        # Instance Normalization Operation
        revin_layer = RevIN(num_features=M)
        x0 = revin_layer(rec, 'norm')
        x_ori = self.embedding_window_size(x0)
        # last, last_series = self.encoder_ALi(x_ori, -1, tau=None, delta=None)
        # last_V =  self.projection(last)
        # last_x = last_V + trend_rec

        # Mutil-scale Patching Operation
        for patch_index, patchsize in enumerate(self.patch_size):

            x_patch_num, x_patch_size = x_ori, x_ori
            x_patch_num = rearrange(x_patch_num, 'b (n p) m -> (b p) n m', p=patchsize)  # b*ps d pn
            # x_patch_size = rearrange(x_patch_size, 'b (p n) m -> (b n) p m', p=patchsize)

            V1, series1 = self.encoder_ALi[0](x_patch_num, patch_index, tau=None, delta=None)
            # series_patch_mean.append(series1)
            V1 = rearrange(V1, '(b p) n m -> (b n) p m', p=patchsize)
            # V1 = rearrange(V1, '(b n) p m -> (b p) n m', n=self.win_size//patchsize)

            # V2, series2 = self.encoder_ALi[1](x_patch_size, patch_index, tau=None, delta=None)

            V2, series2 = self.encoder_ALi[1](V1, patch_index, tau=None, delta=None)

            # for index1, ser1 in enumerate(series1):
            #     ser1 = 0.5 * ser1
            #     ser1 = self.dropout_series1(ser1)
            # for index2, ser2 in enumerate(series2):
                # ser2 = 0.5 * ser2
                # ser2 = self.dropout_series2(ser2)

            # series = series1 + series2
            # for index, ser in enumerate(series):
            #     ser = 0.5 * ser
            #     ser = self.dropout_series1(ser)

            series_patch_mean.append(serie2)
            # V_list.append(V2)

        # series_patch_mean.append([])
        # series_patch_mean.append([])

        if self.output_attention:
            return series_patch_mean[1], series_patch_mean[0], rec0
        else:
            return None

