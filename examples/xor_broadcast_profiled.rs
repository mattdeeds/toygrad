use std::time::Instant;
use toygrad::{GpuContext, SGD, Tensor};

fn main() {
    env_logger::init();
    println!("=== Profiling XOR Broadcast ===\n");

    let ctx = GpuContext::new();

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

    let x = Tensor::new(
        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        vec![4, 2],
        ctx.clone(),
    );

    let y_true = Tensor::new(&[0.0, 1.0, 1.0, 0.0], vec![4, 1], ctx.clone());

    let optimizer = SGD::new(0.5);
    let epochs = 100; // Reduced for profiling

    // Timing buckets
    let mut time_zero_grad = 0.0;
    let mut time_forward = 0.0;
    let mut time_matmul1 = 0.0;
    let mut time_add1 = 0.0;
    let mut time_relu = 0.0;
    let mut time_matmul2 = 0.0;
    let mut time_add2 = 0.0;
    let mut time_sigmoid = 0.0;
    let mut time_loss = 0.0;
    let mut time_backward = 0.0;
    let mut time_optimizer = 0.0;

    println!("Running {} epochs with detailed timing...\n", epochs);

    let total_start = Instant::now();

    for _epoch in 0..epochs {
        // Time: Zero gradients
        let t = Instant::now();
        w1.zero_grad();
        b1.zero_grad();
        w2.zero_grad();
        b2.zero_grad();
        time_zero_grad += t.elapsed().as_secs_f64();

        // Time: Forward pass - DETAILED
        let t = Instant::now();
        let z1 = x.matmul(&w1);
        time_matmul1 += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let z1 = z1.add(&b1);
        time_add1 += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let h1 = z1.relu();
        time_relu += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let z2 = h1.matmul(&w2);
        time_matmul2 += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let z2 = z2.add(&b2);
        time_add2 += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let y_pred = z2.sigmoid();
        time_sigmoid += t.elapsed().as_secs_f64();

        // Time: Loss
        let t = Instant::now();
        let loss = y_pred.mse_loss(&y_true);
        time_loss += t.elapsed().as_secs_f64();

        // Time: Backward
        let t = Instant::now();
        loss.backward();
        time_backward += t.elapsed().as_secs_f64();

        // Time: Optimizer
        let t = Instant::now();
        optimizer.step(&mut w1);
        optimizer.step(&mut b1);
        optimizer.step(&mut w2);
        optimizer.step(&mut b2);
        time_optimizer += t.elapsed().as_secs_f64();
    }

    let total_time = total_start.elapsed().as_secs_f64();

    println!("=== Profiling Results ===\n");
    println!("Total time: {:.4} seconds\n", total_time);

    time_forward = time_matmul1 + time_add1 + time_relu + time_matmul2 + time_add2 + time_sigmoid;

    println!("High-level breakdown:");
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
        time_loss,
        100.0 * time_loss / total_time
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

    println!("\nForward pass details:");
    println!(
        "  Matmul (x @ w1):    {:.4}s ({:.1}%)",
        time_matmul1,
        100.0 * time_matmul1 / total_time
    );
    println!(
        "  Add (z1 + b1):      {:.4}s ({:.1}%)",
        time_add1,
        100.0 * time_add1 / total_time
    );
    println!(
        "  ReLU:               {:.4}s ({:.1}%)",
        time_relu,
        100.0 * time_relu / total_time
    );
    println!(
        "  Matmul (h1 @ w2):   {:.4}s ({:.1}%)",
        time_matmul2,
        100.0 * time_matmul2 / total_time
    );
    println!(
        "  Add (z2 + b2):      {:.4}s ({:.1}%)",
        time_add2,
        100.0 * time_add2 / total_time
    );
    println!(
        "  Sigmoid:            {:.4}s ({:.1}%)",
        time_sigmoid,
        100.0 * time_sigmoid / total_time
    );

    let accounted = time_zero_grad + time_forward + time_loss + time_backward + time_optimizer;
    println!(
        "\n  Other/Overhead:     {:.4}s ({:.1}%)",
        total_time - accounted,
        100.0 * (total_time - accounted) / total_time
    );

    println!("\n=== Extrapolated to 2000 epochs ===");
    println!("Estimated time: {:.2} seconds", total_time * 20.0);
}
