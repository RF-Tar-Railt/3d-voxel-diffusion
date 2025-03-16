import torch
from module.unet import VoxelUNet
from module.diffusion import Diffusion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = 16
model = VoxelUNet(
    model_channels=SIZE,
    in_channels=1,
    out_channels=2,
    channel_mult=(1, 2, 4),
    num_res_blocks=2,
    num_classes=6,
    attention_resolutions=(2, 4)
).to(device)
model.load_state_dict(torch.load(f'models/voxel_diffusion_{SIZE}_3_only_mask_labeled.pth'))
model.eval()

diff = Diffusion(device)

samples = diff.ddim_sample(model, batch_size=16, channels=1, size=SIZE, label=4)
print(samples.shape)
samples = samples.permute(0, 2, 3, 4, 1)
voxels = (((samples[..., 0] + 1) * 0.5) > 0.9).cpu().numpy()
#colors = ((samples[..., :3] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
#alphas = (((samples[..., 3] + 1) * 127.5).clamp(0, 255) > 200)
#colors = colors.contiguous().cpu().numpy()
#masks = alphas.contiguous().cpu().numpy()
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    ax: Axes3D = fig.add_subplot(4, 4, i + 1, projection='3d')  # type: ignore
    ax.voxels(voxels[i], edgecolor='k')
    ax.set_xlim([0, SIZE])
    ax.set_ylim([0, SIZE])
    ax.set_zlim([0, SIZE])
plt.show()
