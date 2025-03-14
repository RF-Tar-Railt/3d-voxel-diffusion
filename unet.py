import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate


# class TextEncoder(nn.Module):
#     def __init__(self, embed_dim=256):
#         super().__init__()
#         self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#         self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.proj = nn.Linear(512, embed_dim)  # CLIP输出512维 → 目标维度
#         self.clip.requires_grad_(False)  # 冻结CLIP参数，仅训练投影层
#
#     def forward(self, text):
#         inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.clip.device)
#         outputs = self.clip(**inputs)
#         return self.proj(outputs.last_hidden_state.mean(dim=1))

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, input, t_emb=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):  # type: ignore
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, input, t_emb=None):
        x = input
        for layer in self:
            if isinstance(layer, TimestepBlock) and t_emb is not None:
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half_dim).float() / half_dim).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        )

        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

    def forward(self, input, t_emb=None):
        out = self.conv1(input)
        if t_emb is not None:
            t_emb = self.time_mlp(t_emb)
            while len(t_emb.shape) < len(out.shape):
                t_emb = t_emb[..., None]
            out = out + t_emb
        out = self.conv2(out)
        return out + self.shortcut(input)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W, D = x.shape
        # x = x.reshape(B, C, -1)
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H*W*D).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W, D)
        h = self.proj(h)
        return x + h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = interpolate(
            x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
        )
        x = self.conv(x)
        return x


# 3D UNet模型
class VoxelUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels=32,
        num_classes=4,
        num_res_blocks=2,
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(4, 8),
    ):
        super().__init__()
        self.model_channels = model_channels
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.label_embed = nn.Embedding(num_classes, time_emb_dim)
        self.init_conv = nn.Conv3d(in_channels, model_channels, 3, padding=1)
        # 编码器
        self.encoder = nn.ModuleList([
            TimestepEmbedSequential(
                ResidualBlock(model_channels, model_channels, time_emb_dim, dropout),
                ResidualBlock(model_channels, model_channels, time_emb_dim, dropout),
                nn.Conv3d(model_channels, model_channels, 3, stride=(1, 2, 2), padding=1),
            ),
            TimestepEmbedSequential(
                ResidualBlock(model_channels, model_channels * 2, time_emb_dim, dropout),
                ResidualBlock(model_channels * 2, model_channels * 2, time_emb_dim, dropout),
                AttentionBlock(model_channels * 2),
                nn.Conv3d(model_channels * 2, model_channels * 2, 3, stride=(1, 2, 2), padding=1),
            ),
            TimestepEmbedSequential(
                ResidualBlock(model_channels * 2, model_channels * 4, time_emb_dim, dropout),
                ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim, dropout),
                AttentionBlock(model_channels * 4),
            )
        ])
        # input_block_channels = [model_channels]
        # ch = model_channels
        # ds = 1
        #
        # for level, mult in enumerate(channel_mult):
        #     for _ in range(num_res_blocks):
        #         layers: list = [ResidualBlock(ch, mult * model_channels, time_emb_dim, dropout)]
        #         ch = mult * model_channels
        #         if ds in attention_resolutions:
        #             layers.append(AttentionBlock(ch))
        #         self.encoder.append(TimestepEmbedSequential(*layers))
        #         input_block_channels.append(ch)
        #     if level != len(channel_mult) - 1:
        #         #self.encoder.append(TimestepEmbedSequential(nn.MaxPool3d(2)))
        #         self.encoder[-1].append(
        #             TimestepEmbedSequential(nn.Conv3d(ch, ch, 3, stride=(1, 2, 2), padding=1))
        #         )
        #         input_block_channels.append(ch)
        #         ds *= 2

        self.middle = TimestepEmbedSequential(
            ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim, dropout),
            AttentionBlock(model_channels * 4),
            ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim, dropout),
        )

        # 解码器
        self.decoder = nn.ModuleList([
            TimestepEmbedSequential(
                ResidualBlock(model_channels * (4 + 4), model_channels * 4, time_emb_dim, dropout),
                ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim, dropout),
                AttentionBlock(model_channels * 4),

            ),
            TimestepEmbedSequential(
                ResidualBlock(model_channels * (4 + 2), model_channels * 2, time_emb_dim, dropout),
                ResidualBlock(model_channels * 2, model_channels * 2, time_emb_dim, dropout),
                AttentionBlock(model_channels * 2),
                Upsample(model_channels * 2),
            ),
            TimestepEmbedSequential(
                ResidualBlock(model_channels * (2 + 1), model_channels, time_emb_dim, dropout),
                Upsample(model_channels * 1),
            )
        ])

        # for level, mult in list(enumerate(channel_mult))[::-1]:
        #     for i in range(num_res_blocks + 1):
        #         layers = [ResidualBlock(ch + input_block_channels.pop(), mult * model_channels, time_emb_dim, dropout)]
        #         ch = mult * model_channels
        #         if ds in attention_resolutions:
        #             layers.append(AttentionBlock(ch))
        #         if level and i == num_res_blocks:
        #             layers.append(Upsample(ch))
        #             ds //= 2
        #         self.decoder.append(TimestepEmbedSequential(*layers))

        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channels, 3, padding=1),
        )

    #
    # def positional_encoding(self, t):
    #     inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2).float().to(t.device) / 256))
    #     pos_enc = torch.einsum('i,j->ij', t.float(), inv_freq)
    #     return torch.cat([pos_enc.sin(), pos_enc.cos()], dim=-1)

    def forward(self, x, t, y=None):
        hs = []
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        if y is not None:
            emb = emb + self.label_embed(y)
        h = x
        h = self.init_conv(h)
        for mod in self.encoder:
            h = mod(h, emb)
            hs.append(h)
        h = self.middle(h, emb)
        for mod in self.decoder:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = mod(cat_in, emb)
        return self.final_conv(h)
