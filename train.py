import torch
from argparse import ArgumentParser
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from module.dataset import DummyDataset
from module.unet import VoxelUNet
from module.diffusion import Diffusion


def get_args():
    parser = ArgumentParser("Train Voxel Diffusion Model")
    parser.add_argument('--size', type=int, default=16, help='Size of the model', choices=[16, 32, 64])
    parser.add_argument('--length', type=int, default=80000, help='Size of the dataset')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for training', dest='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--only-mask', action='store_true', help='Train only with mask', default=False, dest='only_mask')
    parser.add_argument('--with-label', action='store_true', help='Train with label', default=True, dest='with_label')
    parser.add_argument('--out-dir', type=str, default='models', help='Output directory for saving models')
    parser.add_argument('--schedule', choices=['linear', 'cosine'], default='linear', help='Diffusion Beta schedule')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps for diffusion')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = args.device
    torch.set_default_device(device)
    epochs = args.epoch
    only_mask = args.only_mask
    SIZE = args.size

    # 使用虚拟数据集进行训练
    dataset = DummyDataset(length=args.length, size=SIZE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device))

    model = VoxelUNet(
        model_channels=SIZE,
        in_channels=1 if only_mask else 4,
        out_channels=2 if only_mask else 8,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        num_classes=dataset.num_classes,
        attention_resolutions=(2, 4)
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(device, args.schedule, args.timesteps)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            voxel, label = batch
            if only_mask:
                voxel = voxel[:, 3:, ...].to(device)
            else:
                voxel = voxel.to(device)
            optimizer.zero_grad()

            t = diffusion.sample_timesteps(voxel.shape[0])
            loss = diffusion.train_losses(model, voxel, t, label if args.with_label else None)
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

    model_name = f'voxel_diffusion_'
    suffix = [f'{SIZE}_3']
    if only_mask:
        suffix.append('only_mask')
    if args.with_label:
        suffix.append('labeled')
    model_name += '_'.join(suffix)

    # 保存模型
    torch.save(model.state_dict(), f'{args.out_dir}/{model_name}.pth')
