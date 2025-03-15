import numpy as np
import torch
from D3D.unet import VoxelUNet
from D3D.diffusion import Diffusion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VoxelUNet(
    in_channels=4,
    out_channels=4,
    channel_mult=(1, 2, 4),
    num_res_blocks=2,
    attention_resolutions=(2, 4)
).to(device)
model.load_state_dict(torch.load('models/voxel_diffusion_labeled.pth'))
model.eval()

diff = Diffusion(device)

samples = diff.sample(model, batch_size=4, channels=4, label=3)
print(samples.shape)
samples = samples.permute(0, 2, 3, 4, 1)
colors = ((samples[..., :3] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
alphas = (samples[..., 3].clamp(0, 1) > 0.5).float()
colors = colors.contiguous().cpu().numpy()
masks = alphas.contiguous().cpu().numpy()
fig = plt.figure(figsize=(10, 10))
for i in range(4):
    ax: Axes3D = fig.add_subplot(2, 2, i + 1, projection='3d')  # type: ignore
    ax.voxels(masks[i], facecolors=colors[i] / 255, edgecolor='k')
    ax.set_xlim([0, 32])
    ax.set_ylim([0, 32])
    ax.set_zlim([0, 32])
plt.show()
