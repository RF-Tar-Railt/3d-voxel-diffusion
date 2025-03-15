import numpy as np
import torch
from torch.utils.data import Dataset
import random


class ColorStrategy:
    def get_colors(self, x, y, z):
        raise NotImplementedError


class SolidColor(ColorStrategy):
    def __init__(self, color):
        self.color = np.array(color, dtype=np.uint8)

    def get_colors(self, x, y, z):
        return np.tile(self.color, (len(x), 1))


class LayeredColor(ColorStrategy):
    def __init__(self, axis, color_map):
        self.axis = axis
        self.color_map = color_map

    def get_colors(self, x, y, z):
        if self.axis == 0:
            coords = x
        elif self.axis == 1:
            coords = y
        else:
            coords = z
        return self.color_map[coords]


def generate_color_strategy(axis=None, min_coord=0, max_coord=31):
    if random.random() < 0.5:
        return SolidColor(np.random.randint(0, 255, 3))

    if axis is None:
        axis = random.choice([0, 1, 2])
    num_colors = random.randint(2, 5)
    color_map = np.zeros((32, 3), dtype=np.uint8)

    total_length = max_coord - min_coord + 1
    if total_length <= 0:
        return SolidColor(np.random.randint(0, 255, 3))

    step = total_length / num_colors
    for i in range(num_colors):
        start = min_coord + int(i * step)
        end = min_coord + int((i + 1) * step) - 1
        if i == num_colors - 1:
            end = max_coord
        color = np.random.randint(0, 255, 3)
        color_map[start:end + 1] = color

    return LayeredColor(axis, color_map)


def generate_sphere(voxel, cx, cy, cz, r, color_strategy):
    x, y, z = np.ogrid[:32, :32, :32]
    mask = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r ** 2

    coords = np.argwhere(mask)
    x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]

    colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
    voxel[x_coords, y_coords, z_coords, :3] = colors
    voxel[x_coords, y_coords, z_coords, 3] = 255
    return f"A sphere with radius {r}"


def generate_cuboid(voxel, x1, y1, z1, x2, y2, z2, color_strategy):
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    z_min, z_max = sorted([z1, z2])

    x_coords, y_coords, z_coords = np.mgrid[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = z_coords.flatten()

    colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
    voxel[x_coords, y_coords, z_coords, :3] = colors
    voxel[x_coords, y_coords, z_coords, 3] = 255

    if (z_max - z_min) == (y_max - y_min) == (x_max - x_min):
        return f"A cube with side length {x_max - x_min + 1}"
    return f"A cuboid with width {x_max - x_min + 1}, height {y_max - y_min + 1}, depth {z_max - z_min + 1}"


def generate_pyramid(voxel, cx, cy, cz, base_length, height, color_strategy):
    for layer in range(height):
        current_length = base_length - 2 * layer
        if current_length < 1:
            break

        half = current_length // 2
        x_min = max(0, cx - half)
        x_max = min(31, cx + half)
        y_min = max(0, cy - half)
        y_max = min(31, cy + half)
        z = min(cz + layer, 31)

        x, y = np.mgrid[x_min:x_max + 1, y_min:y_max + 1]
        x = x.flatten()
        y = y.flatten()
        z = np.full_like(x, z)

        colors = color_strategy.get_colors(x, y, z)
        voxel[x, y, z, :3] = colors
        voxel[x, y, z, 3] = 255

    return f"A pyramid with base length {base_length}, height {height}"


def generate_cylinder(voxel, cx, cy, z1, z2, radius, color_strategy):
    """生成一个圆柱体"""
    z_start, z_end = sorted([z1, z2])  # 确保 z1 < z2
    z_start = max(0, z_start)
    z_end = min(31, z_end)

    x, y = np.ogrid[:32, :32]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    for z in range(z_start, z_end + 1):
        coords = np.argwhere(mask)
        x_coords, y_coords = coords[:, 0], coords[:, 1]
        z_coords = np.full_like(x_coords, z)

        colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
        voxel[x_coords, y_coords, z_coords, :3] = colors
        voxel[x_coords, y_coords, z_coords, 3] = 255

    return f"A cylinder with radius {radius}, height {z_end - z_start + 1}"


class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.shapes = ['sphere', 'cuboid', 'pyramid', 'cylinder']
        self.data = [self.generate() for _ in range(size)]

    def __len__(self):
        return self.size

    def generate(self):
        voxel = np.zeros((32, 32, 32, 4), dtype=np.uint8)

        shape_type = random.choice(self.shapes)

        if shape_type == 'sphere':
            cx = random.randint(8, 24)
            cy = random.randint(8, 24)
            cz = random.randint(8, 24)
            r = random.randint(3, 6)

            x_min = max(0, cx - r)
            x_max = min(31, cx + r)
            y_min = max(0, cy - r)
            y_max = min(31, cy + r)
            z_min = max(0, cz - r)
            z_max = min(31, cz + r)

            axis = random.choice([0, 1, 2])
            min_c, max_c = [x_min, y_min, z_min][axis], [x_max, y_max, z_max][axis]
            strategy = generate_color_strategy(axis, min_c, max_c)
            generate_sphere(voxel, cx, cy, cz, r, strategy)
            return voxel, shape_type

        elif shape_type == 'cuboid':
            x1, y1, z1 = np.random.randint(0, 25, 3)
            x2, y2, z2 = np.random.randint(x1 + 3, 32, 3)

            axis = random.choice([0, 1, 2])
            min_c = [x1, y1, z1][axis]
            max_c = [x2, y2, z2][axis]
            strategy = generate_color_strategy(axis, min_c, max_c)
            generate_cuboid(voxel, x1, y1, z1, x2, y2, z2, strategy)
            return voxel, shape_type

        elif shape_type == 'pyramid':
            cx = random.randint(8, 24)
            cy = random.randint(8, 24)
            cz = random.randint(0, 16)
            base = random.randint(6, 12)
            height = random.randint(3, 6)

            axis = random.choice([0, 1, 2])
            min_c = [cx - base // 2, cy - base // 2, cz][axis]
            max_c = [cx + base // 2, cy + base // 2, cz + height][axis]
            strategy = generate_color_strategy(axis, min_c, max_c)
            generate_pyramid(voxel, cx, cy, cz, base, height, strategy)
            return voxel, shape_type

        else:
            cx = random.randint(8, 24)
            cy = random.randint(8, 24)
            z1 = random.randint(0, 16)
            z2 = z1 + random.randint(5, 15)
            radius = random.randint(3, 6)

            axis = random.choice([0, 1, 2])
            min_c = [cx - radius, cy - radius, z1][axis]
            max_c = [cx + radius, cy + radius, z2][axis]
            strategy = generate_color_strategy(axis, min_c, max_c)
            generate_cylinder(voxel, cx, cy, z1, z2, radius, strategy)
            return voxel, shape_type

    def __getitem__(self, idx):
        voxel, shape = self.data[idx]
        rgb = voxel[..., :3].astype(np.float32) / 127.5 - 1.
        alpha = voxel[..., 3:4].astype(np.float32) / 255.
        voxel_tensor = np.concatenate([rgb, alpha], axis=-1)
        return torch.from_numpy(voxel_tensor).permute(3, 0, 1, 2).float(), self.shapes.index(shape)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    dataset = DummyDataset()

    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax: Axes3D = fig.add_subplot(3, 3, i + 1, projection='3d')  # type: ignore
        voxel, shape = dataset.data[i]
        alpha_mask = voxel[..., 3] > 127
        face_colors = voxel[..., :3] / 255
        face_colors[~alpha_mask] = 0
        ax.voxels(alpha_mask, facecolors=face_colors, edgecolor='k')
        ax.set_title(shape)

    plt.show()
