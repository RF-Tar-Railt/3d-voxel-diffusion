import torch
from D3D.unet import VoxelUNet
from D3D.sample import Sampler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VoxelUNet().to(device)
model.load_state_dict(torch.load('models/voxel_diffusion.pth'))
model.eval()

diff = Sampler(device)

test_ = diff.sample(model, num_samples=1)
print(test_.shape)

fig = plt.figure(figsize=(10, 10))
ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')  # type: ignore
ax.voxels(test_[0][0], edgecolor='k')
ax.set_xlim([0, 32])
ax.set_ylim([0, 32])
ax.set_zlim([0, 32])
plt.show()
