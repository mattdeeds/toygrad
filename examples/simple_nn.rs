use toygrad::{GpuContext, Tensor};

fn main() {
    println!("=== Simple Neural Network Forward Pass Example ===\n");

    // Initialize GPU
    let ctx = GpuContext::new();

    // Create a simple 2-layer network for XOR problem
    // Input: 2 features
    // Hidden: 4 neurons
    // Output: 1 neuron

    println!("Creating network weights...");

    // Layer 1: (2, 4) weights + (4,) bias
    let w1_data = vec![
        0.5, -0.3, 0.2, 0.7,   // weights for input 1
        -0.4, 0.6, -0.1, 0.3,  // weights for input 2
    ];
    let w1 = Tensor::new(&w1_data, vec![2, 4], ctx.clone());
    let b1 = Tensor::new(&[0.1, -0.2, 0.3, -0.1], vec![1, 4], ctx.clone());

    // Layer 2: (4, 1) weights + (1,) bias
    let w2 = Tensor::new(&[0.8, -0.5, 0.4, 0.6], vec![4, 1], ctx.clone());
    let b2 = Tensor::new(&[0.2], vec![1, 1], ctx.clone());

    // Create XOR training data (4 samples)
    // XOR: [0, 0] -> 0, [0, 1] -> 1, [1, 0] -> 1, [1, 1] -> 0
    let x = Tensor::new(
        &[
            0.0, 0.0,  // sample 1
            0.0, 1.0,  // sample 2
            1.0, 0.0,  // sample 3
            1.0, 1.0,  // sample 4
        ],
        vec![4, 2],
        ctx.clone(),
    );

    let y_true = Tensor::new(&[0.0, 1.0, 1.0, 0.0], vec![4, 1], ctx.clone());

    println!("Input shape: {:?}", x.shape);
    println!("Input data:\n{:?}\n", x.to_vec());

    // Forward pass
    println!("Running forward pass...");

    // Layer 1: x @ w1 + b1, then ReLU
    let z1 = x.matmul(&w1);
    println!("After matmul with W1: {:?}", z1.shape);

    // Broadcast bias by reshaping
    let z1_with_bias_data = z1.to_vec();
    let b1_data = b1.to_vec();
    let mut z1_biased = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            z1_biased.push(z1_with_bias_data[i * 4 + j] + b1_data[j]);
        }
    }
    let z1_b = Tensor::new(&z1_biased, vec![4, 4], ctx.clone());

    let h1 = z1_b.relu();
    println!("After ReLU: {:?}", h1.shape);
    println!("Hidden activations:\n{:?}\n", h1.to_vec());

    // Layer 2: h1 @ w2 + b2, then sigmoid
    let z2 = h1.matmul(&w2);

    // Broadcast bias
    let z2_data = z2.to_vec();
    let b2_data = b2.to_vec();
    let z2_biased: Vec<f32> = z2_data.iter().map(|&x| x + b2_data[0]).collect();
    let z2_b = Tensor::new(&z2_biased, vec![4, 1], ctx.clone());

    let y_pred = z2_b.sigmoid();
    println!("Output shape: {:?}", y_pred.shape);
    println!("Predictions: {:?}", y_pred.to_vec());
    println!("Targets:     {:?}\n", y_true.to_vec());

    // Compute loss
    let loss = y_pred.mse_loss(&y_true);
    println!("MSE Loss: {:?}", loss.to_vec());

    println!("\n=== Forward pass complete! ===");
    println!("\nNote: This is a forward-pass-only example.");
    println!("To train the network, implement the backward pass and optimizer.");
}
