# NeuralPDE.jl - Physics-Informed Point Networks (PIPNs)

## Overview

This repository contains the implementation of **Physics-Informed Point Networks (PIPNs)**, a specialized class of neural networks designed for solving Partial Differential Equations (PDEs) in irregular geometries. PIPNs, a subset of Physics-Informed Neural Networks (PINNs), directly incorporate physical laws into the learning process, making them especially suited for complex and irregular domains.

PIPNs utilize point clouds to handle irregular geometries that are challenging for traditional numerical methods, offering a flexible and scalable approach. This project focuses on the Julia implementation of PIPNs using Flux.jl and associated libraries, and was developed as part of Google Summer of Code (GSoC) 2024 under the mentorship of Dr. Chris Rackauckas.

## Features

- **PointNet Architecture:** Implementation of the PointNet architecture using shared multi-layer perceptrons (MLPs), max pooling, and global feature encoding.
- **Physics-Informed Loss Function:** Flexible framework for constructing loss functions based on PDE residuals and boundary conditions.
- **Automatic Differentiation:** Leveraging Julia's automatic differentiation capabilities to efficiently compute derivatives required for the physics-informed loss.
- **Advanced Optimization:** Techniques for improving convergence and stability, including learning rate schedules, momentum, and gradient clipping.
- **Validation & Testing:** Comprehensive validation against benchmark problems with complex geometries and integration with visualization tools like Plots.jl and Makie.jl.

## Installation

To install and use this package, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/ka-bear/NeuralPDE.jl.git
   cd NeuralPDE.jl
   ```

2. Install the required dependencies:
   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```

3. Load the package in your Julia environment:
   ```julia
   using NeuralPDE
   ```

## Usage

### Example: Solving PDEs with PIPNs

This repository includes examples demonstrating how to use PIPNs for solving PDEs in irregular geometries. Below is a basic example:

```julia
using Flux

# Define shared MLP layers
shared_mlp1 = Flux.Chain(
    Dense(2, 64, tanh),
    Dense(64, 64, tanh)
)

# Define the PointNet model
function pointnet_model(points)
    points = apply_shared_mlp(shared_mlp1, points)
    global_feature = aggregate_global_feature(points)
    combined_features = after_pool_mlp(global_feature)
    return final_layer(combined_features)
end

# Training loop (sample)
for epoch in 1:num_epochs
    for (input, target) in final_dataset
        gs = Flux.gradient(Flux.params(shared_mlp1, after_pool_mlp, final_layer)) do
            predictions = pointnet_model(input)
            loss = loss_fn(target, predictions)
        end
        Flux.Optimise.update!(optimizer, Flux.params(shared_mlp1, after_pool_mlp, final_layer), gs)
    end
end
```

For more examples and detailed explanations, refer to the [documentation](https://github.com/ka-bear/NeuralPDE.jl/docs).

## Contributions

We welcome contributions from the community! Whether it's bug fixes, new features, or improving documentation, your help is appreciated. Please refer to our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of Google Summer of Code (GSoC) 2024 under the mentorship of Dr. Chris Rackauckas. Special thanks to the Julia community for their continuous support.

For more details on the project, check out the GSoC proposal [here](https://github.com/ka-bear/PIPN).
