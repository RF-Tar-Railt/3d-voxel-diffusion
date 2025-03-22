import math
import re

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.dataset import DummyDataset
from module.unet import VoxelUNet
from module.diffusion import Diffusion


dummy = DummyDataset()


def voxels_to_mesh(
    voxel_list,
    grid_shape,
    only_mask=False,
    add_wireframe=True,
    wireframe_thickness=0.1
):
    cube_scale = 1.0 - wireframe_thickness if add_wireframe else 1.0

    cubes = []
    for idx, voxels in enumerate(voxel_list):
        row = idx // grid_shape[1]
        col = idx % grid_shape[1]
        offset_x = row * voxels.shape[0] * 1.5
        offset_z = col * voxels.shape[1] * 1.5
        for x in range(voxels.shape[0]):
            for y in range(voxels.shape[1]):
                for z in range(voxels.shape[2]):
                    r, g, b, a = voxels[x, y, z]
                    if a > 0:  # 仅处理非透明体素
                        # 创建立方体，稍小一些以形成线框效果
                        cube = o3d.geometry.TriangleMesh.create_box(width=cube_scale, height=cube_scale, depth=cube_scale)
                        # 居中放置立方体，留出间隙形成线框效果
                        offset = (1 - cube_scale) / 2
                        cube.translate([x + offset + offset_x, y + offset, z + offset + offset_z], relative=False)
                        # 设置颜色（顶点颜色）
                        color = np.array([r, g, b], dtype=np.float32) / 255.0
                        cube.vertex_colors = o3d.utility.Vector3dVector(
                            np.tile(color, (len(cube.vertices), 1))
                        )
                        cubes.append(cube)
    # 合并所有立方体
    combined_mesh = o3d.geometry.TriangleMesh()
    for cube in cubes:
        combined_mesh += cube
    if only_mask:
        combined_mesh.compute_vertex_normals()
    return combined_mesh


# 训练功能封装
def train_model(length, size, epoch, batch_size, lr, only_mask, with_label, progress=gr.Progress(track_tqdm=True)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    epochs = int(epoch)
    only_mask = bool(only_mask)
    SIZE = int(size)

    # 使用虚拟数据集进行训练
    dataset = DummyDataset(length=length, size=SIZE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    model = VoxelUNet(
        model_channels=SIZE,
        in_channels=1 if only_mask else 4,
        out_channels=2 if only_mask else 8,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        num_classes=dataset.num_classes,
        attention_resolutions=(2, 4)
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    diffusion = Diffusion(device)
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
            loss = diffusion.train_losses(model, voxel, t, label if with_label else None)
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
    plt.savefig("results/loss_curve.png")

    model_name = f'voxel_diffusion_'
    suffix = [f'{SIZE}_3']
    if only_mask:
        suffix.append('only_mask')
    if with_label:
        suffix.append('labeled')
    model_name += '_'.join(suffix)

    # 保存模型
    torch.save(model.state_dict(), f'models/{model_name}.pth')
    return f'models/{model_name}.pth', "results/loss_curve.png"


# 生成样本功能封装
def generate_sample(model_path, size, batch_size, label, alpha, only_mask, progress=gr.Progress(track_tqdm=True)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "only_mask" in model_path:
        only_mask = True
    SIZE = int(size)
    mat = re.match(r"(\d+)_3", model_path)
    if mat is not None:
        SIZE = int(mat.group(1))
    if label is not None:
        _label = dummy.shape_index[label]
    else:
        _label = None
    model = VoxelUNet(
        model_channels=SIZE,
        in_channels=1 if only_mask else 4,
        out_channels=2 if only_mask else 8,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        num_classes=dummy.num_classes,
        attention_resolutions=(2, 4)
    ).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    diff = Diffusion(device)

    samples = diff.ddim_sample(model, batch_size=batch_size, channels=1 if only_mask else 4, size=SIZE,
                               label=_label)
    samples = samples.permute(0, 2, 3, 4, 1)
    voxel_list = []
    for i in range(batch_size):
        sample = samples[i]
        if only_mask:
            voxels = np.zeros((SIZE, SIZE, SIZE, 4), dtype=np.uint8)
            voxels[..., 3] = (((sample[..., 0] + 1) * 0.5) > alpha).cpu().numpy() * 255
            voxels[..., :3] = 75
        else:
            sample[...,:3] = ((sample[...,:3] + 1) * 127.5).clamp(0, 255)
            sample[..., 3] = (((sample[..., 3] + 1) * 0.5) > alpha) * 255
            voxels = sample.cpu().numpy().astype(np.uint8)
        voxels = np.rot90(np.swapaxes(voxels, 0, 1), k=3, axes=(1, 2))
        voxel_list.append(voxels)
    mesh = voxels_to_mesh(voxel_list, (int(math.sqrt(batch_size)), int(math.sqrt(batch_size))), only_mask=only_mask)
    o3d.io.write_triangle_mesh("results/sample.obj", mesh)
    return "results/sample.obj"


# Gradio界面
with gr.Blocks(title="3D Voxel Diffusion") as app:
    gr.Markdown("# 3D体素扩散模型训练与生成")

    with gr.Tabs():
        with gr.TabItem("训练"):
            with gr.Row():
                with gr.Column():
                    dataset_length = gr.Number(label="数据集大小", value=40000)
                    train_size = gr.Dropdown([16, 32, 64], label="模型尺寸", value=16)
                    train_epoch = gr.Number(label="训练轮数", value=10)
                    train_batch = gr.Dropdown([4, 8, 16], label="批次大小", value=4)
                    train_lr = gr.Number(label="学习率", value=1e-4)
                    train_only_mask = gr.Checkbox(label="仅训练掩码")
                    train_with_label = gr.Checkbox(label="使用标签", value=True)
                    train_btn = gr.Button("开始训练")

                with gr.Column():
                    model_output = gr.File(label="输出模型")
                    loss_plot = gr.Image(label="损失曲线")

        with gr.TabItem("生成"):
            with gr.Row():
                with gr.Column():
                    model_input = gr.File(label="上传模型文件")
                    gen_size = gr.Dropdown([16, 32, 64], label="生成尺寸", value=16)
                    gen_batch = gr.Dropdown([4, 9, 16, 25], label="生成数量", value=4)
                    gen_label = gr.Dropdown(list(dummy.shapes.keys()), label="生成标签")
                    gen_alpha = gr.Slider(0.1, 1.0, value=0.9, label="Alpha阈值")
                    gen_only_mask = gr.Checkbox(label="仅生成掩码")
                    gen_btn = gr.Button("生成样本")

                with gr.Column():
                    #gr.Interface
                    gen_preview = gr.Model3D(label="生成预览", clear_color=(0, 0, 0, 0))


    # 事件处理
    train_btn.click(
        train_model,
        inputs=[dataset_length, train_size, train_epoch, train_batch, train_lr, train_only_mask, train_with_label],
        outputs=[model_output, loss_plot]
    )

    gen_btn.click(
        generate_sample,
        inputs=[model_input, gen_size, gen_batch, gen_label, gen_alpha, gen_only_mask],
        outputs=[gen_preview]
    )

if __name__ == "__main__":
    app.launch()
