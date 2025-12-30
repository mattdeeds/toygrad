use toygrad::{GpuContext, Tensor, Adam};

fn main() {
    env_logger::init();
    println!("=== Adam Optimizer Demo: Training XOR ===\n");

    // Initialize GPU
    let ctx = GpuContext::new();

    // Network architecture: 2 -> 4 -> 1
    println!("Initializing network parameters...");

    // Layer 1: (2, 4) weights + (1, 4) bias
    let scale1 = (2.0_f32 / 2.0).sqrt();
    let mut w1 = Tensor::new(
        &[
            0.5 * scale1, -0.3 * scale1, 0.2 * scale1, 0.7 * scale1,
            -0.4 * scale1, 0.6 * scale1, -0.1 * scale1, 0.3 * scale1,
        ],
        vec![2, 4],
        ctx.clone(),
    )
    .with_grad();

    let mut b1 = Tensor::new(&[0.0, 0.0, 0.0, 0.0], vec![1, 4], ctx.clone()).with_grad();

    // Layer 2: (4, 1) weights + (1, 1) bias
    let scale2 = (2.0_f32 / 4.0).sqrt();
    let mut w2 = Tensor::new(
        &[0.8 * scale2, -0.5 * scale2, 0.4 * scale2, 0.6 * scale2],
        vec![4, 1],
        ctx.clone(),
    )
    .with_grad();

    let mut b2 = Tensor::new(&[0.0], vec![1, 1], ctx.clone()).with_grad();

    // XOR dataset
    let x_data = [
        [0.0, 0.0],  // -> 0
        [0.0, 1.0],  // -> 1
        [1.0, 0.0],  // -> 1
        [1.0, 1.0],  // -> 0
    ];
    let y_data = [0.0, 1.0, 1.0, 0.0];

    println!("Dataset: XOR truth table");
    println!("  Architecture: 2 -> 4 (ReLU) -> 1 (Sigmoid)\n");

    // Create Adam optimizer with default parameters
    // lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8
    let optimizer = Adam::new(0.01);
    let epochs = 1000;

    println!("Training with Adam optimizer:");
    println!("  Learning rate: {}", optimizer.lr);
    println!("  Beta1: {}", optimizer.beta1);
    println!("  Beta2: {}", optimizer.beta2);
    println!("  Epsilon: {}\n", optimizer.epsilon);

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        // Process each sample
        for (x_sample, &y_target) in x_data.iter().zip(y_data.iter()) {
            // Zero gradients
            w1.zero_grad();
            b1.zero_grad();
            w2.zero_grad();
            b2.zero_grad();

            // Create input tensor (1, 2)
            let x = Tensor::new(x_sample, vec![1, 2], ctx.clone());
            let y_true = Tensor::new(&[y_target], vec![1, 1], ctx.clone());

            // Forward pass
            let z1 = x.matmul(&w1).add(&b1);
            let h1 = z1.relu();
            let z2 = h1.matmul(&w2).add(&b2);
            let y_pred = z2.sigmoid();

            // Compute loss
            let loss = y_pred.mse_loss(&y_true);
            epoch_loss += loss.to_vec()[0];

            // Backward pass
            loss.backward();

            // Update parameters with Adam
            optimizer.step(&mut w1);
            optimizer.step(&mut b1);
            optimizer.step(&mut w2);
            optimizer.step(&mut b2);
        }

        // Print progress
        if epoch % 100 == 0 || epoch == epochs - 1 {
            let avg_loss = epoch_loss / x_data.len() as f32;
            println!("Epoch {:4}: Average Loss = {:.6}", epoch, avg_loss);

            // Show predictions
            let mut preds = Vec::new();
            for x_sample in &x_data {
                let x = Tensor::new(x_sample, vec![1, 2], ctx.clone());
                let z1 = x.matmul(&w1).add(&b1);
                let h1 = z1.relu();
                let z2 = h1.matmul(&w2).add(&b2);
                let y_pred = z2.sigmoid();
                preds.push(y_pred.to_vec()[0]);
            }
            println!("  Predictions: [{:.3}, {:.3}, {:.3}, {:.3}]",
                     preds[0], preds[1], preds[2], preds[3]);
            println!("  Targets:     [0.000, 1.000, 1.000, 0.000]");
        }
    }

    println!("\n=== Training Complete ===");

    // Final evaluation
    println!("\nFinal Test:");
    let mut correct = 0;
    for (i, x_sample) in x_data.iter().enumerate() {
        let x = Tensor::new(x_sample, vec![1, 2], ctx.clone());
        let z1 = x.matmul(&w1).add(&b1);
        let h1 = z1.relu();
        let z2 = h1.matmul(&w2).add(&b2);
        let y_pred = z2.sigmoid();

        let pred_val = y_pred.to_vec()[0];
        let target_val = y_data[i];
        let predicted_class = if pred_val > 0.5 { 1.0 } else { 0.0 };

        println!("Input [{:.0}, {:.0}] -> Prediction: {:.4} (Target: {:.0}) {}",
                 x_sample[0], x_sample[1], pred_val, target_val,
                 if predicted_class == target_val { "✓" } else { "✗" });

        if predicted_class == target_val {
            correct += 1;
        }
    }

    println!("\nAccuracy: {}/4 ({:.0}%)", correct, (correct as f32 / 4.0) * 100.0);

    println!("\nNote: Adam typically converges faster than SGD and is less sensitive");
    println!("to learning rate choice. Try comparing this with simple_sgd example!");
}
