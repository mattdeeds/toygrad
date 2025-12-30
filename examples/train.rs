use toygrad::{GpuContext, Tensor, SGD};

fn main() {
    println!("=== Training a Simple Neural Network ===\n");

    // Initialize GPU
    let ctx = GpuContext::new();

    // Create a simple network to learn f(x) = 2*x + 3
    // We'll use a single linear layer: y = W*x + b

    // Initialize parameters
    let mut w = Tensor::new(&[1.0], vec![1, 1], ctx.clone()).with_grad();
    let mut b = Tensor::new(&[0.0], vec![1, 1], ctx.clone()).with_grad();

    // Training data: x -> y
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let y_data = vec![5.0, 7.0, 9.0, 11.0]; // 2*x + 3

    let optimizer = SGD::new(0.01); // learning rate = 0.01

    println!("Target function: y = 2*x + 3");
    println!("Initial parameters: w = {:?}, b = {:?}\n", w.to_vec(), b.to_vec());

    // Training loop
    for epoch in 0..50 {
        let mut total_loss = 0.0;

        // Train on each example
        for (x_val, y_val) in x_data.iter().zip(y_data.iter()) {
            // Zero gradients
            w.zero_grad();
            b.zero_grad();

            // Forward pass
            let x = Tensor::new(&[*x_val], vec![1, 1], ctx.clone());
            let y_true = Tensor::new(&[*y_val], vec![1, 1], ctx.clone());

            let pred = w.matmul(&x).add(&b);
            let diff = pred.sub(&y_true);
            let squared = diff.mul(&diff);
            let loss = squared.mean();

            total_loss += loss.to_vec()[0];

            // Backward pass
            loss.backward();

            // Update parameters
            optimizer.step(&mut w);
            optimizer.step(&mut b);
        }

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}, w = {:?}, b = {:?}",
                epoch, total_loss / x_data.len() as f32,
                w.to_vec(), b.to_vec());
        }
    }

    println!("\n=== Training Complete ===");
    println!("Final parameters: w = {:?}, b = {:?}", w.to_vec(), b.to_vec());
    println!("Expected: w = [2.0], b = [3.0]");

    // Test the trained model
    println!("\n=== Testing ===");
    let test_x = vec![5.0, 6.0, 7.0];
    for x_val in test_x {
        let x = Tensor::new(&[x_val], vec![1, 1], ctx.clone());
        let pred = w.matmul(&x).add(&b);
        let expected = 2.0 * x_val + 3.0;
        println!("Input: {:.1}, Predicted: {:.3}, Expected: {:.1}",
            x_val, pred.to_vec()[0], expected);
    }
}
