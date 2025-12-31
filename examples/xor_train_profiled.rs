use std::time::Instant;
use toygrad::{GpuContext, SGD, Tensor};

fn main() {
    env_logger::init();
    println!("=== Profiling XOR Training ===\n");

    let ctx = GpuContext::new();

    // Initialize weights (same as xor_train)
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

    let x_data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y_data = [0.0, 1.0, 1.0, 0.0];

    let optimizer = SGD::new(0.5);
    let epochs = 100; // Reduced for profiling

    // Timing buckets
    let mut time_tensor_creation = 0.0;
    let mut time_forward = 0.0;
    let mut time_loss_compute = 0.0;
    let mut time_loss_sync = 0.0;
    let mut time_backward = 0.0;
    let mut time_optimizer = 0.0;
    let mut time_zero_grad = 0.0;

    println!("Running {} epochs with detailed timing...\n", epochs);

    let total_start = Instant::now();

    for epoch in 0..epochs {
        for (x_sample, &y_target) in x_data.iter().zip(y_data.iter()) {
            // Time: Zero gradients
            let t = Instant::now();
            w1.zero_grad();
            b1.zero_grad();
            w2.zero_grad();
            b2.zero_grad();
            time_zero_grad += t.elapsed().as_secs_f64();

            // Time: Tensor creation
            let t = Instant::now();
            let x = Tensor::new(x_sample, vec![1, 2], ctx.clone());
            let y_true = Tensor::new(&[y_target], vec![1, 1], ctx.clone());
            time_tensor_creation += t.elapsed().as_secs_f64();

            // Time: Forward pass
            let t = Instant::now();
            let z1 = x.matmul(&w1).add(&b1);
            let h1 = z1.relu();
            let z2 = h1.matmul(&w2).add(&b2);
            let y_pred = z2.sigmoid();
            time_forward += t.elapsed().as_secs_f64();

            // Time: Loss computation
            let t = Instant::now();
            let loss = y_pred.mse_loss(&y_true);
            time_loss_compute += t.elapsed().as_secs_f64();

            // Time: GPU-to-CPU sync (reading loss value)
            let t = Instant::now();
            let _loss_val = loss.to_vec()[0];
            time_loss_sync += t.elapsed().as_secs_f64();

            // Time: Backward pass
            let t = Instant::now();
            loss.backward();
            time_backward += t.elapsed().as_secs_f64();

            // Time: Optimizer step
            let t = Instant::now();
            optimizer.step(&mut w1);
            optimizer.step(&mut b1);
            optimizer.step(&mut w2);
            optimizer.step(&mut b2);
            time_optimizer += t.elapsed().as_secs_f64();
        }
    }

    let total_time = total_start.elapsed().as_secs_f64();

    println!("=== Profiling Results ===\n");
    println!("Total time: {:.4} seconds\n", total_time);

    println!("Breakdown:");
    println!(
        "  Tensor creation:    {:.4}s ({:.1}%)",
        time_tensor_creation,
        100.0 * time_tensor_creation / total_time
    );
    println!(
        "  Zero gradients:     {:.4}s ({:.1}%)",
        time_zero_grad,
        100.0 * time_zero_grad / total_time
    );
    println!(
        "  Forward pass:       {:.4}s ({:.1}%)",
        time_forward,
        100.0 * time_forward / total_time
    );
    println!(
        "  Loss computation:   {:.4}s ({:.1}%)",
        time_loss_compute,
        100.0 * time_loss_compute / total_time
    );
    println!(
        "  Loss sync (to_vec): {:.4}s ({:.1}%)",
        time_loss_sync,
        100.0 * time_loss_sync / total_time
    );
    println!(
        "  Backward pass:      {:.4}s ({:.1}%)",
        time_backward,
        100.0 * time_backward / total_time
    );
    println!(
        "  Optimizer step:     {:.4}s ({:.1}%)",
        time_optimizer,
        100.0 * time_optimizer / total_time
    );

    let accounted = time_tensor_creation
        + time_zero_grad
        + time_forward
        + time_loss_compute
        + time_loss_sync
        + time_backward
        + time_optimizer;
    println!(
        "  Other/Overhead:     {:.4}s ({:.1}%)",
        total_time - accounted,
        100.0 * (total_time - accounted) / total_time
    );

    println!("\nNote: Each operation includes GPU kernel launch overhead");
    println!("      and potential CPU-GPU synchronization.");
}
