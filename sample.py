import torch
from transformers import CLIPTokenizer

from D3D.unet import VoxelUNet


class Sampler:
    def __init__(self, device, T=1000):
        self.device = device
        self.T = T

        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size):
        return torch.randint(1, self.T + 1, (batch_size,), device=self.device)

    def noise_voxels(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t - 1])[:, None, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t - 1])[:, None, None, None, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

    def sample(self, model: VoxelUNet, num_samples=1):
        with torch.no_grad():
            x = torch.randn((num_samples, 1, 32, 32, 32)).to(self.device)
            for t in reversed(range(1, self.T + 1)):
                z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                alpha = self.alphas[t - 1]
                alpha_bar = self.alpha_bars[t - 1]
                beta = self.betas[t - 1]

                pred_noise = model(x, torch.full((num_samples,), t, device=self.device))
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(
                    beta) * z
        x = x.clamp(0, 1)
        x = torch.round(x)
        x = (x * 1000).to(torch.uint16)
        return x

    def sample_with_text(self, model: VoxelUNet, prompt: str, num_samples=1):
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(self.device)
        text_embed = model.text_encoder(inputs.input_ids).to(self.device)
        with torch.no_grad():
            x = torch.randn((num_samples, 1, 32, 32, 32)).to(self.device)
            for t in reversed(range(1, self.T + 1)):
                z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                alpha = self.alphas[t - 1]
                alpha_bar = self.alpha_bars[t - 1]
                beta = self.betas[t - 1]

                pred_noise = model(x, torch.full((num_samples,), t, device=self.device), text_embed)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(
                    beta) * z
        x = x.clamp(0, 1)
        x = torch.round(x)
        x = (x * 1000).to(torch.uint16)
        return x
