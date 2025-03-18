import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from module.unet import VoxelUNet
from module.diffusion import Diffusion
from module.dataset import DummyDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = 16
model = VoxelUNet(
    model_channels=SIZE,
    in_channels=4,
    out_channels=8,
    channel_mult=(1, 2, 4),
    num_res_blocks=2,
    num_classes=DummyDataset().num_classes,
    attention_resolutions=(2, 4)
).to(device)
model.load_state_dict(torch.load(f'models/voxel_diffusion_{SIZE}_3_labeled.pth'))
model.eval()

diff = Diffusion(device)

samples = diff.ddim_sample(model, batch_size=16, size=SIZE, channels=4, label=6)
print(samples.shape)
samples = samples.permute(0, 2, 3, 4, 1)
colors = ((samples[..., :3] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
alphas = (((samples[..., 0] + 1) * 0.5) > 0.9)
colors = colors.contiguous().cpu().numpy()
masks = alphas.contiguous().cpu().numpy()
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    ax: Axes3D = fig.add_subplot(4, 4, i + 1, projection='3d')  # type: ignore
    ax.voxels(masks[i], facecolors=colors[i] / 255, edgecolor='k')
    ax.set_xlim([0, SIZE])
    ax.set_ylim([0, SIZE])
    ax.set_zlim([0, SIZE])
plt.show()
