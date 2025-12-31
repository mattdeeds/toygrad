# ToyGrad

A minimal GPU-accelerated tensor library with automatic differentiation, built with Rust and WebGPU.

Trying to be readable and hackable.

Currently ~10x slower than PyTorch.

## Features

- **GPU Acceleration**: All tensor operations run on the GPU via WebGPU compute shaders
- **Automatic Differentiation**: Reverse-mode autodiff with dynamic computation graphs
- **Broadcasting**: NumPy-style broadcasting for element-wise operations with automatic gradient reduction
- **Neural Network Operations**: Matrix multiplication, activations (ReLU, Sigmoid), loss functions (MSE)
- **Optimizers**: SGD and Adam (adaptive learning rates) with parameter updates

## Quick Example

```rust
use toygrad::{GpuContext, Tensor, SGD};

fn main() {
    let ctx = GpuContext::new();

    // Create trainable parameters
    let mut w = Tensor::new(&[0.5], vec![1, 1], ctx.clone()).with_grad();
    let mut b = Tensor::new(&[0.0], vec![1, 1], ctx.clone()).with_grad();

    // Training data
    let x = Tensor::new(&[2.0], vec![1, 1], ctx.clone());
    let y_true = Tensor::new(&[5.0], vec![1, 1], ctx.clone());

    let optimizer = SGD::new(0.1);

    for _ in 0..100 {
        w.zero_grad();
        b.zero_grad();

        // Forward pass: y = w*x + b
        let pred = w.matmul(&x).add(&b);
        let loss = pred.mse_loss(&y_true);

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer.step(&mut w);
        optimizer.step(&mut b);
    }

    println!("Learned: w = {:?}, b = {:?}", w.to_vec(), b.to_vec());
}
```

## Examples

Run the included examples to see the library in action:

```bash
# Train a neural network to solve XOR (demonstrates multi-layer networks)
cargo run --example xor_train

# XOR with batch training using broadcasting (more efficient)
cargo run --example xor_broadcast

# XOR training with Adam optimizer (faster convergence)
cargo run --example adam_optimizer

# Broadcasting examples and tests
cargo run --example broadcast_test

# Simple linear regression
cargo run --example train

# Basic SGD optimization
cargo run --example simple_sgd

# Gradient computation test
cargo run --example simple_train
```

## Core API

### Tensor Operations
- `Tensor::new(data, shape, context)` - Create a tensor from data
- `.with_grad()` - Enable gradient tracking
- `.matmul(&other)` - Matrix multiplication
- `.add(&other)` / `.sub(&other)` / `.mul(&other)` / `.div(&other)` - Element-wise operations
- `.relu()` / `.sigmoid()` - Activation functions
- `.mse_loss(&target)` - Mean squared error loss

### Autograd
- `.backward()` - Compute gradients via backpropagation
- `.get_grad()` - Retrieve computed gradient
- `.zero_grad()` - Reset gradient to zero

### Optimization
- `SGD::new(learning_rate)` - Create SGD optimizer
- `Adam::new(learning_rate)` - Create Adam optimizer with default hyperparameters (β₁=0.9, β₂=0.999, ε=1e-8)
- `Adam::with_params(lr, beta1, beta2, epsilon)` - Create Adam with custom hyperparameters
- `optimizer.step(&mut param)` - Update parameter using optimizer's algorithm

## Requirements

- Rust 2024 edition
- GPU with WebGPU support

## Architecture

ToyGrad uses WebGPU compute shaders for all tensor operations, making it suitable for both CPU and GPU backends. The automatic differentiation system builds a dynamic computation graph during the forward pass and traverses it in reverse during backpropagation.

## License

MIT
