use toygrad::{GpuContext, Tensor, SGD};

fn main() {
    env_logger::init();
    println!("=== XOR with Broadcasting (Batch Training) ===\n");

    let ctx = GpuContext::new();

    // Network: 2 -> 4 -> 1
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

    let scale2 = (2.0_f32 / 4.0).sqrt();
    let mut w2 = Tensor::new(
        &[0.8 * scale2, -0.5 * scale2, 0.4 * scale2, 0.6 * scale2],
        vec![4, 1],
        ctx.clone(),
    )
    .with_grad();

    let mut b2 = Tensor::new(&[0.0], vec![1, 1], ctx.clone()).with_grad();

    // XOR dataset - ALL 4 samples in a batch!
    let x = Tensor::new(
        &[
            0.0, 0.0,  // [0, 0] -> 0
            0.0, 1.0,  // [0, 1] -> 1
            1.0, 0.0,  // [1, 0] -> 1
            1.0, 1.0,  // [1, 1] -> 0
        ],
        vec![4, 2],
        ctx.clone(),
    );

    let y_true = Tensor::new(&[0.0, 1.0, 1.0, 0.0], vec![4, 1], ctx.clone());

    println!("Using broadcasting for batch training!");
    println!("  Training with ALL 4 samples simultaneously\n");

    let optimizer = SGD::new(0.5);
    let epochs = 2000;

    for epoch in 0..epochs {
        // Zero gradients
        w1.zero_grad();
        b1.zero_grad();
        w2.zero_grad();
        b2.zero_grad();

        // Forward pass - process entire batch at once!
        // Broadcasting automatically handles bias addition
        let z1 = x.matmul(&w1).add(&b1);  // (4,2) @ (2,4) + (1,4) -> (4,4)
        let h1 = z1.relu();
        let z2 = h1.matmul(&w2).add(&b2);  // (4,4) @ (4,1) + (1,1) -> (4,1)
        let y_pred = z2.sigmoid();

        // Compute loss
        let loss = y_pred.mse_loss(&y_true);

        // Print progress
        if epoch % 200 == 0 || epoch == epochs - 1 {
            let loss_val = loss.to_vec()[0];
            let predictions = y_pred.to_vec();
            println!("Epoch {:4}: Loss = {:.6}", epoch, loss_val);
            println!("  Predictions: [{:.3}, {:.3}, {:.3}, {:.3}]",
                     predictions[0], predictions[1], predictions[2], predictions[3]);
            println!("  Targets:     [0.000, 1.000, 1.000, 0.000]");
        }

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer.step(&mut w1);
        optimizer.step(&mut b1);
        optimizer.step(&mut w2);
        optimizer.step(&mut b2);
    }

    println!("\n=== Training Complete ===");

    // Final evaluation
    println!("\nFinal Test:");
    let z1 = x.matmul(&w1).add(&b1);
    let h1 = z1.relu();
    let z2 = h1.matmul(&w2).add(&b2);
    let y_pred = z2.sigmoid();

    let final_preds = y_pred.to_vec();
    let mut correct = 0;
    let targets = [0.0, 1.0, 1.0, 0.0];

    for i in 0..4 {
        let x_vals = [x.to_vec()[i * 2], x.to_vec()[i * 2 + 1]];
        let predicted_class = if final_preds[i] > 0.5 { 1.0 } else { 0.0 };

        println!("Input [{:.0}, {:.0}] -> Prediction: {:.4} (Target: {:.0}) {}",
                 x_vals[0], x_vals[1], final_preds[i], targets[i],
                 if predicted_class == targets[i] { "✓" } else { "✗" });

        if predicted_class == targets[i] {
            correct += 1;
        }
    }

    println!("\nAccuracy: {}/4 ({:.0}%)", correct, (correct as f32 / 4.0) * 100.0);

    println!("\nNote: With broadcasting, we can process all samples in a single batch!");
    println!("This is much more efficient than processing samples one-by-one.");
}
