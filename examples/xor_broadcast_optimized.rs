use std::time::Instant;
use toygrad::{GpuContext, SGD, Tensor};

fn main() {
    env_logger::init();
    println!("=== XOR Broadcast - Optimized (No Sync Points) ===\n");

    let ctx = GpuContext::new();

    // Network: 2 -> 4 -> 1
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

    let scale2 = (2.0_f32 / 4.0).sqrt();
    let mut w2 = Tensor::new(
        &[0.8 * scale2, -0.5 * scale2, 0.4 * scale2, 0.6 * scale2],
        vec![4, 1],
        ctx.clone(),
    )
    .with_grad();

    let mut b2 = Tensor::new(&[0.0], vec![1, 1], ctx.clone()).with_grad();

    // XOR dataset - ALL 4 samples in a batch
    let x = Tensor::new(
        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        vec![4, 2],
        ctx.clone(),
    );

    let y_true = Tensor::new(&[0.0, 1.0, 1.0, 0.0], vec![4, 1], ctx.clone());

    println!("Training WITHOUT print statements (no GPU-to-CPU sync)");
    println!("This eliminates synchronization overhead\n");

    let optimizer = SGD::new(0.5);
    let epochs = 2000;

    // Start timing
    let start_time = Instant::now();

    for _epoch in 0..epochs {
        // Zero gradients
        w1.zero_grad();
        b1.zero_grad();
        w2.zero_grad();
        b2.zero_grad();

        // Forward pass
        let z1 = x.matmul(&w1).add(&b1);
        let h1 = z1.relu();
        let z2 = h1.matmul(&w2).add(&b2);
        let y_pred = z2.sigmoid();

        // Compute loss
        let loss = y_pred.mse_loss(&y_true);

        // NO PRINTING HERE - this avoids to_vec() sync!

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer.step(&mut w1);
        optimizer.step(&mut b1);
        optimizer.step(&mut w2);
        optimizer.step(&mut b2);
    }

    // End timing (before any GPU sync)
    let training_time = start_time.elapsed();

    println!("\n=== Training Complete ===");
    println!("Training time: {:.4} seconds", training_time.as_secs_f64());
    println!("(No GPU sync during training loop)\n");

    // NOW sync to get final results
    println!("Final Test:");
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

        println!(
            "Input [{:.0}, {:.0}] -> Prediction: {:.4} (Target: {:.0}) {}",
            x_vals[0],
            x_vals[1],
            final_preds[i],
            targets[i],
            if predicted_class == targets[i] {
                "✓"
            } else {
                "✗"
            }
        );

        if predicted_class == targets[i] {
            correct += 1;
        }
    }

    println!(
        "\nAccuracy: {}/4 ({:.0}%)",
        correct,
        (correct as f32 / 4.0) * 100.0
    );
}
