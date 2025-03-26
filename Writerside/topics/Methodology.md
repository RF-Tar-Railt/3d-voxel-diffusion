# 3 Methodology

### 3.1 Technical Overview

1. **Diffusion Model Principles**: Learn the data distribution by gradually adding noise (forward process) and denoising (reverse process), using either linear or cosine schedules to control the noise intensity.

2. **3D Voxel Representation**: Represent 3D objects as a 16×16×16 voxel grid with RGB and alpha channels.

3. **Conditional Generation Mechanism**: Implement class-conditional control via label embedding, supporting the generation of specified shapes (e.g., spheres, cubes, etc.).

### 3.2 Diffusion Model

The diffusion model generates data by gradually adding noise (forward process) and then removing it (reverse process).

#### 3.2.1 Forward Process

Given the input data $x_0$, Gaussian noise is added over $T$ time steps until pure noise $x_t$ is reached. The noise intensity at each step is controlled by a preset schedule (e.g., linear or cosine). The forward process is defined as:
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I})
$$
where $\beta_t$ is the variance used at each step, also known as the noise schedule.

A key property of the diffusion process is that it allows sampling of $x_t$ for any step $t$ based on the original data $x_0$, using:
$$
x_t \sim q(x_t | x_0) = \sqrt{\bar{\alpha_t}}\, x_0 + \sqrt{1-\bar{\alpha_t}}\, \epsilon,\quad \epsilon \sim N(0, 1)
$$

$$
\alpha_t = 1 - \beta_t,\quad \bar{\alpha_t} = \prod^{t}_{i=1}\alpha_i
$$

#### 3.2.2 Reverse Process

The reverse process learns to recover the original data $x_0$ from the noisy $x_t$. By training a neural network to predict the noise or directly reconstruct the data, the model gradually denoises the input. Each step is defined as:
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))
$$
or
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \Bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}}\,\epsilon_{\theta}(x_t, t)\Bigr) + \sigma_t\, \text{z}
$$

$$
\text{z} \sim N(0, 1)
$$

where $\epsilon_{\theta}$ is the noise estimation function (i.e., the prediction model), and $\sigma_t\, \text{z}$ accounts for the error between the predicted and true noise.

#### 3.2.3 Optimization Objective

The model is trained by minimizing the noise prediction error with the loss function defined as:
$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \Bigl[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \Bigr]
= \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha_t}}\, x_{t-1} + \sqrt{1-\bar{\alpha_t}}\, \epsilon, t) \|^2
$$

### 3.3 3D UNet Architecture

The core of the diffusion model is the noise prediction model. This project uses a U-Net model based on residual blocks and attention mechanisms, following an encoder-decoder structure:

- **Encoder**: Uses residual blocks and strided convolutions for progressive downsampling and extraction of multi-scale features. Each layer includes time embedding and (optionally) label embedding to integrate time-step and class information.
- **Decoder**: Upsamples through residual blocks and concatenates with encoder skip connections to recover resolution and details.

#### 3.3.1 Key Components

- **Residual Blocks**: Incorporate GroupNorm, SiLU activation, and 3D convolutions, supporting the additive fusion of time embeddings.
- **Attention Mechanism**: Inserted at specific resolutions to enhance global modeling.
- **Multi-scale Features**: Channel multipliers adjust feature dimensions to suit different resolutions.

#### 3.3.2 Time and Label Embedding

- The time step $t$ is converted to a high-dimensional vector via sinusoidal positional encoding, processed through an MLP, and added to the residual block features.
- Class labels are transformed into a [20]-dimensional vector through an embedding layer and added to the time embedding, enabling conditional generation.

### 3.4 Dataset

The experiments use dynamically generated 3D voxel data of geometric shapes. Shape parameters (e.g., positions, dimensions) and colors are randomly generated to cover diverse configurations.

#### 3.4.1 Shape Generation

- **Basic Shapes**: Include spheres, cubes, cylinders, and pyramids, with support for hollow structures.
- **Complex Objects**: Such as tables and chairs, constructed by combining multiple cubes.

#### 3.4.2 Color Strategy

A random color strategy is used to ensure diversity.

- **Solid Color**: All voxels are filled with a fixed color.
- **Layered Color**: Colors are assigned in layers along the z-axis to enhance diversity.