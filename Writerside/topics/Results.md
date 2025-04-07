# 5. Results

## Training Loss

![Left Panel](bs_loss.png)

![Right Panel](lr_loss.png)

<!-- ![loss compare](loss_compare.jpg) -->

### Detailed Analysis of the Loss Plot

#### Left Panel: Comparison of Different Batch Sizes

- X-axis (Epoch): Number of training iterations.
- Y-axis (Loss): Training loss.

Observations:

1. Initial Loss (around Epoch 0):
   - Batch Size 8 (blue curve) starts at approximately 0.3.
   - Batch Size 16 (orange curve) shows a slightly higher initial loss, around 0.4.
   - Batch Size 32 (green curve) begins near 0.5.
   - Batch Size 64 (red curve) exhibits the highest initial loss of about 0.8.

2. Convergence Behavior:
   - The blue curve (Batch Size 8) quickly reduces below 0.2 in early epochs and eventually converges to roughly 0.05.
   - The orange curve (Batch Size 16) decreases at a similar pace, converging near 0.1.
   - The green curve (Batch Size 32) declines moderately, with the loss stabilizing between 0.08 and 0.1.
   - The red curve (Batch Size 64) shows a slower descent, ultimately stabilizing around 0.15.

3. Summary:
   - Under a fixed learning rate (1e-4), smaller batch sizes (e.g., 8 or 16) facilitate a faster decrease in loss and achieve lower convergence values, although they may lead to instability and longer training time per epoch.
   - Conversely, larger batch sizes (e.g., 64) exhibit a slower decline but ensure more stable training due to processing more samples per update.

#### Right Panel: Comparison of Different Learning Rates (for Batch Size 16)

- X-axis (Epoch): Number of training iterations.
- Y-axis (Loss): Training loss.
- Curve Colors Represent:
  - Blue: LR = 1e-1
  - Orange: LR = 1e-2
  - Green: LR = 1e-3
  - Red: LR = 1e-4

Observations:

1. Initial Loss (Epoch 0):
   - Higher learning rates (e.g., 1e-1, blue curve) result in a higher and more volatile initial loss.
   - Lower learning rates (1e-2, 1e-3, 1e-4) demonstrate more stable initial loss values.

2. Convergence Behavior:
   - The blue curve (LR = 1e-1) drops from approximately 1.5 to about 1.0, then oscillates between 0.8 and 1.0.
   - The orange curve (LR = 1e-2) descends from around 1.0 to 0.6–0.7.
   - The green curve (LR = 1e-3) starts at about 0.5, decreasing steadily to below 0.2 and stabilizing near 0.1.
   - The red curve (LR = 1e-4) begins comparatively low and continuously declines below 0.1.

3. Summary:
   - Excessively high learning rates (1e-1) may yield rapid initial improvements but can also cause oscillations that hinder achieving low loss levels.
   - Lower rates (1e-3 and 1e-4) produce a smoother and more stable descent, reaching lower ultimate loss values despite a slower reduction at the outset.

#### Overall Summary

1. Smaller batch sizes promote faster loss reduction but may incur longer per-epoch training times and potential instability.
2. Larger batch sizes ensure stability at the cost of a slower convergence rate.
3. An optimal balance between batch size and learning rate is crucial.

## Forward Diffusion

![forward_diff](forward_diff.png)

The forward diffusion process illustrates the gradual transition of data from its original state to being covered by Gaussian noise. This process intuitively reflects the changing trends in data distribution, providing a basis for later models to extract meaningful features under noisy conditions.

## Reverse Diffusion

![reverse_diff](reverse_diff.png)

The reverse diffusion stage gradually removes the noise and successfully restores the main features of the data. This process verifies the model's efficiency in detail recovery and structural reconstruction, gradually restoring the full original view.

## Sampling Results

- **Sphere**  

  ![sample_sphere](sample_sphere.png)

- **Cuboid**  

  ![sample_cuboid](sample_cuboid.png)

- **Pyramid**  

  ![sample_pyramid](sample_pyramid.png)

- **Cylinder**  

  ![sample_cylinder](sample_cylinder.png)

- **Table**  

  ![sample_table](sample_table.png)

- **Chair**  

  ![sample_chair](sample_chair.png)

The sampling results cover a range of shapes from basic geometric forms to common objects in everyday scenarios, showcasing the model's balanced performance in detail recovery and overall structure generation. These results demonstrate that the model can precisely capture local features while maintaining global consistency, providing substantial validation for further applications.

## Evaluation

### Evaluation Metrics

The evaluation metrics used to assess the model's performance include:

- Occupancy diff: Measures the difference in spatial occupancy probability between generated and real.

- Components diff: The difference between the generated and the real in the number of connected components.

- Smoothness diff: The difference in smoothness between generated and real.

- L1 and L2 dist: Measures the absolute or squared difference between generated and real at the voxel level.

- Variance ratio: The ratio of the variance of the generated in feature space to the variance of the real data.

- Class JS Divergence: Jensen-Shannon divergence between generated and real in category distribution.

### Evaluation Results

![Evalation](evaluate_results.png)

We conducted an evaluation on one of the models (batch-size 16, epoch 10, SIZE 16, dataset size 80000), and the metrics are as follows:

- occupancy_diff: 0.29
- components_diff: 19.65
- smoothness_diff: 0.191
- L1 dist and L2 dist: 0.082 and 0.092
- variance_ratio: 2.618
- class_js_divergence: 0.013
  
It can be seen that the model performs well in local details and category distribution, while still needing improvement in structural rationality, spatial distribution, and diversity control.
