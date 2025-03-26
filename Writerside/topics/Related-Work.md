# 2. Related Work

Diffusion models have gradually become mainstream in the field of 3D generation, with the core idea of generating data through progressive denoising. Early works such as DreamFusion extended 2D diffusion models to 3D space by using Score Distillation Sampling (SDS) for text-to-3D generation [7], though they were limited by computational efficiency. Later, models like Magic3D improved generation quality through multi-resolution optimization [8], ATT3D adopted an amortized generation strategy to accelerate the text-to-3D mapping [9], and Score Jacobian Chaining proposed using the chain rule of gradients to transfer gradient information from pre-trained 2D diffusion models to 3D space, reducing training costs [10]. Recently, Latent 3D Graph Diffusion further explored generating 3D graph structures in latent space to reduce complexity [11]. However, these methods rely heavily on pre-trained 2D diffusion models, and direct research on integrating label information into 3D space is still limited.

## 2.1 Voxel-based 3D Diffusion Models

Voxel representations, due to their regular grid structure, became the mainstream choice in early 3D diffusion models. For example, Diffusion-SDF achieves conditional generation through voxelized Signed Distance Functions (SDF), but suffers from detail loss due to voxel resolution limitations [12]. Triplane Latent Diffusion further combines triplane latent diffusion by decomposing voxel features into three orthogonal planes to enhance texture continuity [13]. However, voxel-based methods generally face high memory consumption and difficulty in modeling complex topologies.

## 2.2 Implicit Neural Field Diffusion

Implicit representations parameterize 3D structures using neural networks, overcoming voxel resolution limits. MeshDiffusion applies the diffusion process to mesh vertex distributions to support high-fidelity shape generation [14]; HyperDiffusion diffuses the weights of Neural Radiance Fields (NeRF) to generate dynamic scenes [15]. Although these methods can generate continuous surfaces, they entail high training costs and require complex sampling strategies.

## 2.3 Latent Space Diffusion

To improve efficiency, researchers have explored encoding 3D data into a compact latent space for diffusion. The LION model generates 3D shapes via latent diffusion on point clouds [16], while Shap-E combines conditional diffusion with implicit functions to achieve text-driven multimodal generation [17]. More recently, XCube utilizes a sparse voxel hierarchy to reduce computational complexity while preserving details [18].

## 2.4 Comparison and Summary of the Above Methods

| Method Type            | Representative Model | Advantage                           | Limitation                                       |
|------------------------|----------------------|-------------------------------------|--------------------------------------------------|
| Voxel-based Diffusion  | Diffusion-SDF        | Regular structure, easy to optimize | Low resolution, high memory consumption          |
| Implicit Diffusion     | MeshDiffusion        | Continuous surfaces, detailed       | Slow sampling, complex training                  |
| Latent Space Diffusion | Shap-E; XCube        | Efficient, multimodal support       | Additional design needed for conditional control |