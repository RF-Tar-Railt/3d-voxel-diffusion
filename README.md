# 3d-voxel-diffusion

This is an implementation of the 3D Voxel Diffusion model for 3D object generation. 

~~In the future, this model will be used to generate Minecraft building with NBT format.~~

## Current Progress

In size 16^3 and only 1 layer (Alpha channel), the model can generate some simple 3D objects:

~~My computer is too slow to train the model with size 32^3.~~

### Sphere
![shpere](./results/only_mask/16/sphere_2_size4000+batch4.png)

### Cylinder
![cylinder](./results/only_mask/16/cylinder_1_size4000+batch4.png)

### Table
![table](./results/only_mask/16/table_1_size40000+batch1.png)

### Chair
![chair](./results/only_mask/16/chair_1_size80000+batch4.png)

### Usage

**Train the model**

```bash
python train.py --size 16 --batch 4 --epoch 10 --length 40000 --only-mask --with-label
```

**Sample the model**

```bash
python sample.py .\models\voxel_diffusion_16_3_labeled.pth --label sphere --batch 9
```