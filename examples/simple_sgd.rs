use toygrad::{GpuContext, SGD, Tensor};

fn main() {
    println!("=== Simple SGD Test ===\n");

    let ctx = GpuContext::new();

    // Simple test: learn to fit y = 3.0
    // Start with x = 1.0, try to minimize (x - 3)^2
    let mut x = Tensor::new(&[1.0], vec![1], ctx.clone()).with_grad();
    let target = Tensor::new(&[3.0], vec![1], ctx.clone());

    let optimizer = SGD::new(0.1);

    println!("Initial x: {:?}", x.to_vec());
    println!("Target: {:?}\n", target.to_vec());

    for i in 0..20 {
        // Zero gradients
        x.zero_grad();

        // Forward: loss = (x - target)^2  -> mean
        let diff = x.sub(&target);
        let squared = diff.mul(&diff);
        let loss = squared.mean();

        println!(
            "Iteration {}: x = {:?}, loss = {:?}",
            i,
            x.to_vec(),
            loss.to_vec()
        );

        // Backward
        loss.backward();

        // Check gradient
        if let Some(grad) = x.get_grad() {
            println!("  Gradient: {:?}", grad.to_vec());
        } else {
            println!("  No gradient!");
        }

        // Also check intermediate gradients
        if let Some(grad) = diff.get_grad() {
            println!("  diff gradient: {:?}", grad.to_vec());
        }
        if let Some(grad) = squared.get_grad() {
            println!("  squared gradient: {:?}", grad.to_vec());
        }

        // Update
        optimizer.step(&mut x);
    }

    println!("\nFinal x: {:?}", x.to_vec());
    println!("Expected: [3.0]");
}
