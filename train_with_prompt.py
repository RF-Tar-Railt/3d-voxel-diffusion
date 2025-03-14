import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from D3D.dataset import DummyDataset
from D3D.unet import VoxelUNet
from D3D.sample import Sampler


epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用虚拟数据集进行训练
dataset = DummyDataset(size=1600)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
model = VoxelUNet().to(device)
diffusion = Sampler(device)

optimizer = AdamW(
    [
        {'params': model.parameters()},
        {'params': diffusion.text_encoder.parameters(), 'lr': 1e-5}
    ],
    lr=1e-4
)
loss_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
        voxel, prompt = batch
        voxel = voxel.unsqueeze(1).to(device)
        t = diffusion.sample_timesteps(voxel.shape[0])
        x_t, noise = diffusion.noise_voxels(voxel, t)
        text_emb = diffusion.text_encoder(prompt).to(device)
        optimizer.zero_grad()
        pred_noise = model(x_t, t, text_emb)
        loss = nn.MSELoss()(pred_noise, noise)
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
torch.save(model.state_dict(), 'models/voxel_diffusion_text.pth')
