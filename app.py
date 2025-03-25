import math
import re
import threading
from datetime import datetime
from pathlib import Path

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
stop_training_event = threading.Event()


def voxels_to_mesh(
    voxel_list,
    grid_shape,
    only_mask=False,
):
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
                    if a > 0:  # only non-transparent voxels are processed
                        # Create a cube, slightly smaller to create a wireframe effect
                        if only_mask:
                            cube_scale = 0.9
                        else:
                            cube_scale = 1.0
                        cube = o3d.geometry.TriangleMesh.create_box(width=cube_scale, height=cube_scale, depth=cube_scale)
                        # Place the cube in the center, leaving a gap to create a wireframe effect
                        offset = (1 - cube_scale) / 2
                        cube.translate([x + offset + offset_x, y + offset, z + offset + offset_z], relative=False)
                        # Set color (vertex color)
                        color = np.array([r, g, b], dtype=np.float32) / 255.0
                        cube.vertex_colors = o3d.utility.Vector3dVector(
                            np.tile(color, (len(cube.vertices), 1))
                        )
                        cubes.append(cube)
    # Merge all cubes
    combined_mesh = o3d.geometry.TriangleMesh()
    for cube in cubes:
        combined_mesh += cube
    if only_mask:
        combined_mesh.paint_uniform_color([0.5, 0.5, 0.5])
    return combined_mesh


def train_model(length, size, epoch, batch_size, lr, only_mask, with_label, schedule, timestep, progress=gr.Progress(track_tqdm=True)):
    global stop_training_event
    stop_training_event.clear()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    epochs = int(epoch)
    only_mask = bool(only_mask)
    SIZE = int(size)

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
    diffusion = Diffusion(device, schedule, timestep)
    loss_history = []

    print(f"Start training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(epochs):
        if stop_training_event.is_set():
            break
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            if stop_training_event.is_set():
                break
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

    torch.save(model.state_dict(), f'models/{model_name}.pth')
    yield "results/loss_curve.png", f'models/{model_name}.pth'


def scan_model_files():
    model_dir = "./models"
    if not Path(model_dir).exists():
        return []
    files = [file.as_posix() for file in Path(model_dir).iterdir() if file.is_file() and file.suffix == ".pth"]
    return files


def update_gen_only_mask(model_file, only_mask, size):
    if model_file is None:
        return only_mask, size, gr.Dropdown(choices=scan_model_files())
    if "only_mask" in model_file:
        only_mask = True
    else:
        only_mask = False
    mat = re.search(r"_(\d+)_3", model_file)
    if mat is not None:
        size = int(mat.group(1))
    return only_mask, gr.Dropdown(value=size), gr.Dropdown(choices=scan_model_files())


def generate_sample(model_path, size, batch_size, label, alpha, only_mask, schedule, timestep, progress=gr.Progress(track_tqdm=True)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "only_mask" in model_path:
        only_mask = True
    SIZE = int(size)
    mat = re.search(r"_(\d+)_3", model_path)
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

    diff = Diffusion(device, schedule, timestep)
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
    gr.Markdown("# 3D voxel diffusion model training and generation")

    with gr.Tabs():
        with gr.TabItem("Train"):
            with gr.Row():
                with gr.Column():
                    dataset_length = gr.Number(label="Dataset size", value=40000)
                    train_size = gr.Dropdown([16, 32, 64], label="Model size", value=16)
                    train_epoch = gr.Number(label="Epoch", value=10)
                    train_batch = gr.Dropdown([4, 8, 16], label="Batch size", value=4)
                    train_lr = gr.Number(label="Learning rate", value=1e-4)
                    train_schedule = gr.Dropdown(["linear", "cosine"], label="Schedule of `beta`", value="linear")
                    train_timesteps = gr.Number(label="Timestep", value=1000)

                with gr.Column():
                    loss_plot = gr.Image(label="Loss curve")
                    model_output = gr.File(label="Output model")
                    train_only_mask = gr.Checkbox(label="Only mask (no color)")
                    train_with_label = gr.Checkbox(label="Use label", value=True)
                    train_btn = gr.Button("Start", interactive=True)
                    stop_train_btn = gr.Button("Stop", interactive=False)

        with gr.TabItem("Sample"):
            with gr.Column():
                with gr.Row():
                    gen_preview = gr.Model3D(label="Preview", clear_color=(0, 0, 0, 0), height=240)

                with gr.Row():
                    with gr.Column():
                        gen_size = gr.Dropdown([16, 32, 64], label="Model size", value=16)
                        gen_schedule = gr.Dropdown(["linear", "cosine"], label="Schedule of `beta`", value="linear")
                        gen_timesteps = gr.Number(label="Timestep", value=1000)
                        gen_alpha = gr.Slider(0.1, 1.0, value=0.9, label="Alpha threshold")

                    with gr.Column():
                        model_input = gr.Dropdown(label="Model", choices=scan_model_files())
                        gen_batch = gr.Dropdown([1, 4, 9, 16, 25], label="Count", value=4)
                        gen_label = gr.Dropdown(list(dummy.shapes.keys()), label="Label")
                        gen_only_mask = gr.Checkbox(label="Only mask (no color)")
                        gen_btn = gr.Button("Generate")

    # 事件处理
    train_event = train_btn.click(
        train_model,
        inputs=[dataset_length, train_size, train_epoch, train_batch, train_lr, train_only_mask, train_with_label, train_schedule, train_timesteps],
        outputs=[loss_plot, model_output],
        concurrency_limit=1
    )

    stop_train_btn.click(
        lambda: stop_training_event.set(),
        inputs=None,
        outputs=None,
    )

    train_btn.click(
        lambda: [gr.Button(interactive=False), gr.Button(interactive=True)],
        outputs=[train_btn, stop_train_btn]
    )
    train_event.then(
        lambda: [gr.Button(interactive=True), gr.Button(interactive=False), gr.Dropdown(choices=scan_model_files())],
        outputs=[train_btn, stop_train_btn, model_input]
    )

    gen_btn.click(
        generate_sample,
        inputs=[model_input, gen_size, gen_batch, gen_label, gen_alpha, gen_only_mask, gen_schedule, gen_timesteps],
        outputs=[gen_preview]
    )

    model_input.change(
        update_gen_only_mask,
        inputs=[model_input, gen_only_mask, gen_size],
        outputs=[gen_only_mask, gen_size, model_input]
    )

if __name__ == "__main__":
    app.launch()
