# 3D Voxel Diffusion

## Abstract
The three-dimensional diffusion model shows great potential in generating complex 3D data, but existing methods are often limited by high computational cost and insufficient semantic control. This study introduces a lightweight 3D diffusion model based on voxel representation, aiming to achieve semantically constrained 3D generation via label embedding and providing a reproducible baseline framework for both educational and practical applications. The model utilizes low-resolution voxels and an optimized 3D UNet architecture, combined with dynamic noise scheduling and conditional embedding mechanisms, effectively reducing training complexity. Experimental results demonstrate that the framework can generate diverse basic geometric shapes (e.g., spheres, cubes) and simple composite objects (e.g., tables, chairs) with controllable generation through category labels. Further analysis reveals limitations in detail resolution and generalization to real-world objects, laying the groundwork for future high-resolution and multi-modal controlled generation.

## Keywords
3D diffusion, voxel representation, 3D UNet, label embedding, dynamic noise scheduling, conditional embedding, controllable generation
