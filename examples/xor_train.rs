use std::time::Instant;
use toygrad::{GpuContext, SGD, Tensor};

fn main() {
    env_logger::init();
    println!("=== Training Neural Network on XOR Problem ===\n");

    // Initialize GPU
    let ctx = GpuContext::new();

    // Network architecture: 2 -> 4 -> 1
    // Input: 2 features (x1, x2)
    // Hidden: 4 neurons with ReLU activation
    // Output: 1 neuron with Sigmoid activation

    println!("Initializing network parameters...");

    // Layer 1: (2, 4) weights + (1, 4) bias
    // Xavier initialization: scale by sqrt(2/n_in)
    let scale1 = (2.0_f32 / 2.0).sqrt();
    let mut w1 = Tensor::new(
        &[
            0.5 * scale1,
            -0.3 * scale1,
            0.2 * scale1,
            0.7 * scale1,
            -0.4 * scale1,
            0.6 * scale1,
            -0.1 * scale1,
            0.3 * scale1,
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

    // XOR dataset: 4 samples
    // [0, 0] -> 0, [0, 1] -> 1, [1, 0] -> 1, [1, 1] -> 0
    let x_data = [
        [0.0, 0.0], // -> 0
        [0.0, 1.0], // -> 1
        [1.0, 0.0], // -> 1
        [1.0, 1.0], // -> 0
    ];
    let y_data = [0.0, 1.0, 1.0, 0.0];

    println!("Dataset: XOR truth table");
    println!("  Architecture: 2 -> 4 (ReLU) -> 1 (Sigmoid)\n");

    // Create optimizer
    let optimizer = SGD::new(0.5);
    let epochs = 2000;

    println!(
        "Training for {} epochs with learning rate {}\n",
        epochs, optimizer.lr
    );

    // Start timing
    let start_time = Instant::now();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        // Process each sample individually (true stochastic gradient descent)
        for (x_sample, &y_target) in x_data.iter().zip(y_data.iter()) {
            // Zero gradients
            w1.zero_grad();
            b1.zero_grad();
            w2.zero_grad();
            b2.zero_grad();

            // Create input tensor (1, 2) - single sample
            let x = Tensor::new(x_sample, vec![1, 2], ctx.clone());
            let y_true = Tensor::new(&[y_target], vec![1, 1], ctx.clone());

            // Forward pass
            // Layer 1: z1 = x @ w1 + b1
            let z1 = x.matmul(&w1).add(&b1);

            // Activation: h1 = relu(z1)
            let h1 = z1.relu();

            // Layer 2: z2 = h1 @ w2 + b2
            let z2 = h1.matmul(&w2).add(&b2);

            // Output activation: y_pred = sigmoid(z2)
            let y_pred = z2.sigmoid();

            // Compute loss: MSE
            let loss = y_pred.mse_loss(&y_true);
            epoch_loss += loss.to_vec()[0];

            // Backward pass
            loss.backward();

            // Update parameters
            optimizer.step(&mut w1);
            optimizer.step(&mut b1);
            optimizer.step(&mut w2);
            optimizer.step(&mut b2);
        }

        // Print progress
        if epoch % 200 == 0 || epoch == epochs - 1 {
            let avg_loss = epoch_loss / x_data.len() as f32;
            println!("Epoch {:4}: Average Loss = {:.6}", epoch, avg_loss);

            // Show predictions for all samples
            let mut preds = Vec::new();
            for x_sample in &x_data {
                let x = Tensor::new(x_sample, vec![1, 2], ctx.clone());
                let z1 = x.matmul(&w1).add(&b1);
                let h1 = z1.relu();
                let z2 = h1.matmul(&w2).add(&b2);
                let y_pred = z2.sigmoid();
                preds.push(y_pred.to_vec()[0]);
            }
            println!(
                "  Predictions: [{:.3}, {:.3}, {:.3}, {:.3}]",
                preds[0], preds[1], preds[2], preds[3]
            );
            println!("  Targets:     [0.000, 1.000, 1.000, 0.000]");
        }
    }

    // End timing
    let training_time = start_time.elapsed();

    println!("\n=== Training Complete ===");
    println!(
        "Training time: {:.4} seconds\n",
        training_time.as_secs_f64()
    );

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

        println!(
            "Input [{:.0}, {:.0}] -> Prediction: {:.4} (Target: {:.0}) {}",
            x_sample[0],
            x_sample[1],
            pred_val,
            target_val,
            if predicted_class == target_val {
                "✓"
            } else {
                "✗"
            }
        );

        if predicted_class == target_val {
            correct += 1;
        }
    }

    println!(
        "\nAccuracy: {}/4 ({:.0}%)",
        correct,
        (correct as f32 / 4.0) * 100.0
    );
}
