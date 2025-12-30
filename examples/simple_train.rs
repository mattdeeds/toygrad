use toygrad::{GpuContext, Tensor};

fn main() {
    println!("=== Simple Gradient Test ===\n");

    // Initialize GPU
    let ctx = GpuContext::new();

    // Test simple backward pass: z = x + y
    let x = Tensor::new(&[2.0], vec![1], ctx.clone()).with_grad();
    let y = Tensor::new(&[3.0], vec![1], ctx.clone()).with_grad();

    println!("x = {:?}", x.to_vec());
    println!("y = {:?}", y.to_vec());

    // Forward pass
    let z = x.add(&y);
    println!("\nz = x + y = {:?}", z.to_vec());

    // Backward pass
    println!("\nCalling z.backward()...");
    z.backward();

    // Check gradients
    println!("\nGradients:");
    if let Some(grad_x) = x.get_grad() {
        println!("dz/dx = {:?} (expected: [1.0])", grad_x.to_vec());
    } else {
        println!("No gradient for x!");
    }

    if let Some(grad_y) = y.get_grad() {
        println!("dz/dy = {:?} (expected: [1.0])", grad_y.to_vec());
    } else {
        println!("No gradient for y!");
    }

    println!("\n=== Test Complete ===");
}
