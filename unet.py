import torch
import torch.nn as nn
from torch.nn.functional import gelu, silu
from transformers import CLIPTextModel


class TextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Linear(512, embed_dim)  # CLIP输出512维 → 目标维度

    def forward(self, text):
        outputs = self.clip(text).last_hidden_state
        return self.proj(outputs.mean(dim=1))  # 取[CLS]标记的平均池化


class DoubleAdjConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            return gelu(x + out)
        else:
            return silu(out)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, upsample=False):
        super().__init__()
        self.conv1 = DoubleAdjConvBlock(in_channels, in_channels, residual=True)
        self.conv2 = DoubleAdjConvBlock(in_channels, out_channels, mid_channels=(in_channels // 2) if upsample else None)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        if time_emb_dim:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.time_mlp and t_emb is not None:
            t_emb = self.time_mlp(t_emb)
            out = out + t_emb[..., None, None, None]
        out += self.shortcut(x)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W, D = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, C, -1)
        q, k, v = qkv.unbind(1)

        attn = torch.softmax(q @ k.transpose(-2, -1) / (C ** 0.5), dim=-1)
        out = (attn @ v).reshape(B, C, H, W, D)
        return x + self.proj(out)


# 3D UNet模型
class VoxelUNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_emb_dim = 256
        self.text_encoder = TextEncoder()
        self.text_proj = nn.Linear(time_emb_dim, time_emb_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(256, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 编码器
        self.init_conv = nn.Conv3d(1, 32, 3, padding=1)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool3d(2),
                ResidualBlock(32, 64, time_emb_dim),
                AttentionBlock(64)
            ),
            nn.Sequential(
                nn.MaxPool3d(2),
                ResidualBlock(64, 128, time_emb_dim),
                AttentionBlock(128)
            ),
            nn.Sequential(
                nn.MaxPool3d(2),
                ResidualBlock(128, 256, time_emb_dim),
                AttentionBlock(256)
            )
        ])

        self.middle = nn.ModuleList([
            DoubleAdjConvBlock(256, 512),
            DoubleAdjConvBlock(512, 512),
            DoubleAdjConvBlock(512, 256)
        ])

        # 解码器
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                ResidualBlock(383, 128, time_emb_dim, upsample=True),
                AttentionBlock(256)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                ResidualBlock(192, 64, time_emb_dim, upsample=True),
                AttentionBlock(128)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                ResidualBlock(96, 32, time_emb_dim, upsample=True),
                AttentionBlock(32)
            )
        ])

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 1, 3, padding=1)
        )

        self.skip_connections = []

    def positional_encoding(self, t):
        #inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2).float().to(t.device) / 256))
        inv_freq = torch.reciprocal(
            torch.pow(torch.tensor(10000.0).to(t.device), torch.arange(0, 256, 2).float().to(t.device) / 256)
        )
        pos_enc = torch.einsum('i,j->ij', t.float(), inv_freq)
        return torch.cat([pos_enc.sin(), pos_enc.cos()], dim=-1)

    def forward(self, x, t, text_embed=None):
        t_emb = self.time_embed(self.positional_encoding(t))
        if text_embed is not None:
            text_cond = self.text_proj(text_embed)
            t_emb = t_emb + text_cond
        x = self.init_conv(x)
        skips = [x]

        for block in self.encoder:
            x = block[0](x)
            x = block[1](x, t_emb)
            x = block[2](x)
            skips.append(x)

        for block in self.middle:
            x = block(x)

        skips.pop()
        for i, block in enumerate(self.decoder):
            x = block[0](x)  # Upsample
            x = torch.cat([x, skips.pop()], dim=1)
            x = block[1](x, t_emb)
            x = block[2](x)

        return self.final_conv(x)
