import numpy as np
import torch
from torch.utils.data import Dataset
import random
import itertools


class ColorStrategy:
    def get_colors(self, x, y, z):
        raise NotImplementedError


class SolidColor(ColorStrategy):
    def __init__(self, color):
        self.color = np.array(color, dtype=np.uint8)

    def get_colors(self, x, y, z):
        return np.tile(self.color, (len(x), 1))


class LayeredColor(ColorStrategy):
    def __init__(self, color_map):
        self.color_map = color_map

    def get_colors(self, x, y, z):
        return self.color_map[z]


def generate_color_strategy(min_coord=0, max_coord=31):
    if random.random() < 0.5:
        return SolidColor(np.random.randint(0, 255, 3))

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

    return LayeredColor(color_map)


def generate_sphere(voxel, cx, cy, cz, r, color_strategy, hollow=False):
    size = voxel.shape[0]
    x, y, z = np.ogrid[:size, :size, :size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r ** 2
    if hollow:
        inner_mask = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= (r - 1) ** 2
        mask &= ~inner_mask

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return "A Empty sphere"
    x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]

    colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
    voxel[x_coords, y_coords, z_coords, :3] = colors
    voxel[x_coords, y_coords, z_coords, 3] = 255
    return f"A{' hollow' if hollow else ''} sphere with radius {r}"


def generate_cuboid(voxel, x1, y1, z1, x2, y2, z2, color_strategy, hollow=False):
    size = voxel.shape[0]
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    z_min, z_max = sorted([z1, z2])
    x_max = min(size - 1, x_max)
    y_max = min(size - 1, y_max)
    z_max = min(size - 1, z_max)

    if hollow and (x_max - x_min > 1) and (y_max - y_min > 1) and (z_max - z_min > 1):
        # Create outer shell
        # Top and bottom faces
        for z in [z_min, z_max]:
            x_coords, y_coords = np.mgrid[x_min:x_max + 1, y_min:y_max + 1]
            x_coords = x_coords.flatten()
            y_coords = y_coords.flatten()
            z_coords = np.full_like(x_coords, z)

            colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
            voxel[x_coords, y_coords, z_coords, :3] = colors
            voxel[x_coords, y_coords, z_coords, 3] = 255

        # Front and back faces (excluding corners already counted)
        for y in [y_min, y_max]:
            x_coords, z_coords = np.mgrid[x_min:x_max + 1, z_min + 1:z_max]
            x_coords = x_coords.flatten()
            z_coords = z_coords.flatten()
            y_coords = np.full_like(x_coords, y)

            colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
            voxel[x_coords, y_coords, z_coords, :3] = colors
            voxel[x_coords, y_coords, z_coords, 3] = 255

        # Left and right faces (excluding edges already counted)
        for x in [x_min, x_max]:
            y_coords, z_coords = np.mgrid[y_min + 1:y_max, z_min + 1:z_max]
            y_coords = y_coords.flatten()
            z_coords = z_coords.flatten()
            x_coords = np.full_like(y_coords, x)

            colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
            voxel[x_coords, y_coords, z_coords, :3] = colors
            voxel[x_coords, y_coords, z_coords, 3] = 255
    else:
        x_coords, y_coords, z_coords = np.mgrid[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        z_coords = z_coords.flatten()

        colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
        voxel[x_coords, y_coords, z_coords, :3] = colors
        voxel[x_coords, y_coords, z_coords, 3] = 255

    if (z_max - z_min) == (y_max - y_min) == (x_max - x_min):
        return f"A{'n hollow' if hollow else ''} cube with side length {x_max - x_min + 1}"
    return f"A{'n hollow' if hollow else ''} cuboid with width {x_max - x_min + 1}, height {y_max - y_min + 1}, depth {z_max - z_min + 1}"


def generate_pyramid(voxel, cx, cy, cz, base_length, height, color_strategy):
    size = voxel.shape[0]
    for layer in range(height):
        current_length = base_length - 2 * layer
        if current_length < 1:
            break

        half = current_length // 2
        x_min = max(0, cx - half)
        x_max = min(size - 1, cx + half)
        y_min = max(0, cy - half)
        y_max = min(size - 1, cy + half)
        z = min(cz + layer, size - 1)

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
    size = voxel.shape[0]
    z_start, z_end = sorted([z1, z2])  # 确保 z1 < z2
    z_start = max(0, z_start)
    z_end = min(size - 1, z_end)

    x, y = np.ogrid[:size, :size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    for z in range(z_start, z_end + 1):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
        x_coords, y_coords = coords[:, 0], coords[:, 1]
        z_coords = np.full_like(x_coords, z)

        colors = color_strategy.get_colors(x_coords, y_coords, z_coords)
        voxel[x_coords, y_coords, z_coords, :3] = colors
        voxel[x_coords, y_coords, z_coords, 3] = 255

    return f"A cylinder with radius {radius}, height {z_end - z_start + 1}"


def generate_table(voxel, color_strategy):
    size = voxel.shape[0]

    # 随机参数
    table_height = random.randint(4, 8)
    desktop_thickness = random.randint(1, 3)
    desktop_size = random.randint(6, 12)
    leg_size = random.choice([1, 2])
    leg_offset = random.randint(0, 2)  # 腿的偏移量

    # 桌面位置
    cx = random.randint(desktop_size // 2 + 2, size - desktop_size // 2 - 2)
    cy = random.randint(desktop_size // 2 + 2, size - desktop_size // 2 - 2)
    cz = size - table_height

    # 生成桌面
    generate_cuboid(voxel,
                    cx - desktop_size // 2, cy - desktop_size // 2, cz,
                    cx + desktop_size // 2, cy + desktop_size // 2, cz + desktop_thickness - 1,
                    color_strategy)

    # 统一腿部生成逻辑
    leg_z_end = cz - 1
    leg_positions = [
        (cx - desktop_size // 2 + leg_offset, cy - desktop_size // 2 + leg_offset),
        (cx - desktop_size // 2 + leg_offset, cy + desktop_size // 2 - leg_offset - leg_size + 1),
        (cx + desktop_size // 2 - leg_offset - leg_size + 1, cy - desktop_size // 2 + leg_offset),
        (cx + desktop_size // 2 - leg_offset - leg_size + 1, cy + desktop_size // 2 - leg_offset - leg_size + 1)
    ]

    for x, y in leg_positions:
        generate_cuboid(voxel,
                        x, y, 0,
                        x + leg_size - 1, y + leg_size - 1, leg_z_end,
                        color_strategy)

    return "Table"


def generate_chair(voxel, color_strategy):
    size = voxel.shape[0]

    # 随机参数（与桌子使用相同参数名）
    seat_size = random.randint(4, 8)
    backrest_height = random.randint(3, 6)
    leg_size = random.choice([1, 2])  # 与桌子相同的腿尺寸选项
    seat_height = random.randint(size // 4, size // 2)
    leg_offset = random.randint(0, 1)  # 使用与桌子相似的偏移参数

    # 座位位置
    cx = random.randint(seat_size // 2 + 2, size - seat_size // 2 - 2)
    cy = random.randint(seat_size // 2 + 2, size - seat_size // 2 - 2)

    # 生成座位
    generate_cuboid(voxel,
                    cx - seat_size // 2, cy - seat_size // 2, seat_height,
                    cx + seat_size // 2, cy + seat_size // 2, seat_height + 1,
                    color_strategy)

    # 统一靠背生成逻辑
    backrest_dir = random.choice(["back", "side"])
    if backrest_dir == "back":
        generate_cuboid(voxel,
                        cx - seat_size // 2, cy + seat_size // 2 + 1, seat_height,
                        cx + seat_size // 2, cy + seat_size // 2 + 2, seat_height + backrest_height,
                        color_strategy)
    else:
        generate_cuboid(voxel,
                        cx + seat_size // 2 + 1, cy - seat_size // 2, seat_height,
                        cx + seat_size // 2 + 2, cy + seat_size // 2, seat_height + backrest_height,
                        color_strategy)

    # 统一腿部生成逻辑（与桌子相同结构）
    leg_z_end = seat_height - 1
    leg_positions = [
        (cx - seat_size // 2 + leg_offset, cy - seat_size // 2 + leg_offset),
        (cx - seat_size // 2 + leg_offset, cy + seat_size // 2 - leg_offset - leg_size + 1),
        (cx + seat_size // 2 - leg_offset - leg_size + 1, cy - seat_size // 2 + leg_offset),
        (cx + seat_size // 2 - leg_offset - leg_size + 1, cy + seat_size // 2 - leg_offset - leg_size + 1)
    ]

    for x, y in leg_positions:
        generate_cuboid(voxel,
                        x, y, 0,
                        x + leg_size - 1, y + leg_size - 1, leg_z_end,
                        color_strategy)

    return "Chair"


class DummyDataset(Dataset):
    def __init__(self, size=1000, length=32):
        self.size = size
        self.voxel_size = length
        self.shapes = ['sphere', 'cuboid', 'pyramid', 'cylinder', 'table', 'chair', 'hollow_sphere', 'hollow_cuboid']
        self.num_classes = 6
        self.data = list(itertools.islice(map(self.generate, itertools.cycle(self.shapes)), size))

    def __len__(self):
        return self.size

    def generate(self, shape_type: str):
        size = self.voxel_size
        voxel = np.zeros((size, size, size, 4), dtype=np.uint8)

        if shape_type in ["sphere", "hollow_sphere"]:
            cx = random.randint(8, size - 8)
            cy = random.randint(8, size - 8)
            cz = random.randint(8, size - 8)
            r = random.randint(3, 7)

            strategy = generate_color_strategy(cz-r, cz+r)
            generate_sphere(voxel, cx, cy, cz, r, strategy, shape_type.startswith('hollow'))
            return voxel, shape_type.split('_')[-1]

        elif shape_type in ["cuboid", "hollow_cuboid"]:
            max_dim = size // 2
            x1, y1, z1 = np.random.randint(0, size-max_dim, 3)
            x2 = x1 + np.random.randint(4, max_dim)
            y2 = y1 + np.random.randint(4, max_dim)
            z2 = z1 + np.random.randint(4, max_dim)

            strategy = generate_color_strategy(z1, z2)
            generate_cuboid(voxel, x1, y1, z1, x2, y2, z2, strategy, shape_type.startswith('hollow'))
            return voxel, shape_type.split('_')[-1]

        elif shape_type == 'pyramid':
            cx = random.randint(int(0.25 * size), int(0.75 * size))
            cy = random.randint(int(0.25 * size), int(0.75 * size))
            cz = random.randint(0, size // 2)
            base = random.randint(6, 12)
            height = random.randint(3, 6)

            strategy = generate_color_strategy( cz, cz + height)
            generate_pyramid(voxel, cx, cy, cz, base, height, strategy)
            return voxel, shape_type

        elif shape_type == "cylinder":
            cx = random.randint(8, size - 8)
            cy = random.randint(8, size - 8)
            z1 = random.randint(0, size - 10)
            z2 = z1 + random.randint(5, 10)
            radius = random.randint(3, 6)

            strategy = generate_color_strategy( z1, z2)
            generate_cylinder(voxel, cx, cy, z1, z2, radius, strategy)
            return voxel, shape_type.split('_')[-1]
        elif shape_type == 'table':
            strategy = generate_color_strategy(size//2-4, size//2+4)
            generate_table(voxel, strategy)
            return voxel, shape_type
        else:
            strategy = generate_color_strategy(size//2-3, size//2+3)
            generate_chair(voxel, strategy)
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

    dataset = DummyDataset(length=16)

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
