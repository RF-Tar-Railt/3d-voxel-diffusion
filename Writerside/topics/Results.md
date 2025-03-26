# 5. Results

## Training Loss

![Loss](loss_only_mask.png)

The image displays the loss curve during training. A steadily decreasing loss indicates that the model is effectively learning from the data.

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
