import torch
from torch.nn.functional import mse_loss
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from D3D.dataset import DummyDataset
from D3D.unet import VoxelUNet
from D3D.diffusion import Diffusion


epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
model = VoxelUNet(
    in_channels=4,
    out_channels=4,
    channel_mult=(1, 2, 2),
    num_res_blocks=1,
    attention_resolutions=(2, 4)
).to(device)

# 使用虚拟数据集进行训练
dataset = DummyDataset(size=1000)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator(device=device))

optimizer = AdamW(model.parameters(), lr=1e-4)
diffusion = Diffusion(device)
loss_history = []


def train_losses(diff, model, x_start, t, label=None):
    noise = torch.randn_like(x_start).to(diff.device)
    x_t = diff.q_sample(x_start, t, noise)
    pred_noise = model(x_t, t, label)
    loss_rgb = mse_loss(pred_noise[:, :3], noise[:, :3])
    loss_alpha = mse_loss(pred_noise[:, 3], noise[:, 3]) * 0.1
    return loss_rgb + loss_alpha


for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
        voxel, label = batch
        voxel = voxel.to(device)
        optimizer.zero_grad()

        t = diffusion.sample_timesteps(voxel.shape[0])
        loss = train_losses(diffusion, model, voxel, t, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f'Epoch {epoch + 1} Loss: {avg_loss:.4f}')

    # 可视化损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# 保存模型
torch.save(model.state_dict(), 'models/voxel_diffusion_labeled.pth')

