import numpy as np
import torch
from D3D.unet import VoxelUNet
from D3D.diffusion import Diffusion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VoxelUNet(
    in_channels=1,
    out_channels=1,
    channel_mult=(1, 2, 4),
    num_res_blocks=2,
    attention_resolutions=(2, 4)
).to(device)
model.load_state_dict(torch.load('models/voxel_diffusion.pth'))
model.eval()

diff = Diffusion(device)

samples = diff.sample(model, batch_size=4, channels=1)
print(samples.shape)

fig = plt.figure(figsize=(10, 10))
for i in range(4):
    ax: Axes3D = fig.add_subplot(2, 2, i + 1, projection='3d')  # type: ignore
    # # origin data
    # sample = ((samples[i] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # voxel = sample.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    # b_voxel = np.any(voxel != 0, axis=3).astype(np.uint8)
    # ax.voxels(b_voxel, facecolors=voxel / 255, edgecolor='k')
    # origin data / grayscale
    sample = ((samples[i] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    voxel = sample.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    b_voxel = np.any(voxel != 0, axis=3).astype(np.uint8)
    ax.voxels(b_voxel, facecolors=np.repeat(voxel, 3, axis=3) / 255, edgecolor='k')
    # # only keep the binary information
    # sample = samples[i][0].clamp(0, 1).cpu().numpy()
    # ax.voxels(sample, edgecolor='k')
    ax.set_xlim([0, 32])
    ax.set_ylim([0, 32])
    ax.set_zlim([0, 32])
plt.show()
