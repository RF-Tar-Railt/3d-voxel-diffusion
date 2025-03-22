import numpy as np
import torch
import math
import re
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

from module.unet import VoxelUNet
from module.diffusion import Diffusion
from module.dataset import DummyDataset


dataset = DummyDataset()


def get_args():
    parser = ArgumentParser("Sample from Voxel Diffusion Model")
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('--size', type=int, default=16, help='Size of the model', choices=[16, 32, 64])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for sampling', dest='batch_size', choices=[4, 9, 16, 25])
    parser.add_argument('--label', type=str, default=None, help='Label for sampling', choices=list(dataset.shapes.keys()))
    parser.add_argument('--only-mask', action='store_true', help='Sample only mask', default=False, dest='only_mask')
    parser.add_argument('--alpha', type=float, default=0.9, help='Alpha value for mask')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = args.device
    only_mask = args.only_mask
    if "only_mask" in args.model:
        only_mask = True
    SIZE = args.size
    mat = re.match(r"(\d+)_3", args.model)
    if mat is not None:
        SIZE = int(mat.group(1))
    if args.label is not None:
        label = dataset.shape_index[args.label]
    else:
        label = None
    model = VoxelUNet(
        model_channels=SIZE,
        in_channels=1 if only_mask else 4,
        out_channels=2 if only_mask else 8,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        num_classes=dataset.num_classes,
        attention_resolutions=(2, 4)
    ).to(device)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()

    diff = Diffusion(device)

    samples = diff.ddim_sample(model, batch_size=args.batch_size, channels=1 if only_mask else 4, size=SIZE, label=label)
    print(samples.shape)
    samples = samples.permute(0, 2, 3, 4, 1)
    if only_mask:
        voxels = (((samples[..., 0] + 1) * 0.5) > args.alpha).contiguous().cpu().numpy()
        colors = np.array([])
    else:
        voxels = (((samples[..., 3] + 1) * 0.5) > args.alpha).contiguous().cpu().numpy()
        colors = ((samples[..., :3] + 1) * 127.5).clamp(0, 255).to(torch.uint8).contiguous().cpu().numpy()
    fig = plt.figure(figsize=(10, 10))
    for i in range(args.batch_size):
        ax: Axes3D = fig.add_subplot(int(math.sqrt(args.batch_size)), int(math.sqrt(args.batch_size)), i + 1, projection='3d')  # type: ignore
        if only_mask:
            ax.voxels(voxels[i], edgecolor='k')
        else:
            ax.voxels(voxels[i], facecolors=colors[i] / 255, edgecolor='k')
        ax.set_xlim([0, SIZE])
        ax.set_ylim([0, SIZE])
        ax.set_zlim([0, SIZE])
    plt.show()
