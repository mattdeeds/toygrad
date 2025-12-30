use toygrad::{GpuContext, Tensor};

fn main() {
    env_logger::init();
    println!("=== Broadcasting Test ===\n");

    let ctx = GpuContext::new();

    // Test 1: Broadcast bias (1, 4) to (3, 4)
    println!("Test 1: Broadcasting bias addition");
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ],
        vec![3, 4],
        ctx.clone(),
    )
    .with_grad();

    let bias = Tensor::new(&[0.1, 0.2, 0.3, 0.4], vec![1, 4], ctx.clone()).with_grad();

    println!("  x shape: {:?}", x.shape);
    println!("  bias shape: {:?}", bias.shape);

    let result = x.add(&bias);
    println!("  result shape: {:?}", result.shape);
    println!("  result: {:?}", result.to_vec());

    // Expected: each row of x gets bias added
    // Row 1: [1.1, 2.2, 3.3, 4.4]
    // Row 2: [5.1, 6.2, 7.3, 8.4]
    // Row 3: [9.1, 10.2, 11.3, 12.4]

    // Test backward pass - need to sum to scalar first
    println!("  result requires_grad: {}", result.requires_grad);
    let loss = result.sum();
    println!("  loss value: {:?}", loss.to_vec());
    println!("  loss requires_grad: {}", loss.requires_grad);
    loss.backward();

    println!("  x has grad: {}", x.get_grad().is_some());
    println!("  bias has grad: {}", bias.get_grad().is_some());

    if let Some(grad) = x.get_grad() {
        println!("  x gradient: {:?}", grad.to_vec());
    } else {
        println!("  x has no gradient!");
    }

    if let Some(grad) = bias.get_grad() {
        println!("  bias gradient: {:?}", grad.to_vec());
        println!("  Expected bias grad: [3.0, 3.0, 3.0, 3.0] (sum over 3 rows)");
    } else {
        println!("  bias has no gradient!");
    }

    println!("\nTest 2: Broadcasting column vector (3, 1) to (3, 4)");
    let col_vec = Tensor::new(&[0.5, 1.0, 1.5], vec![3, 1], ctx.clone()).with_grad();
    let mat = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ],
        vec![3, 4],
        ctx.clone(),
    )
    .with_grad();

    println!("  col_vec shape: {:?}", col_vec.shape);
    println!("  mat shape: {:?}", mat.shape);

    let result2 = mat.add(&col_vec);
    println!("  result shape: {:?}", result2.shape);
    println!("  result: {:?}", result2.to_vec());

    // Expected: each column gets corresponding row value
    // Row 1: [1.5, 2.5, 3.5, 4.5]  (all + 0.5)
    // Row 2: [6.0, 7.0, 8.0, 9.0]  (all + 1.0)
    // Row 3: [10.5, 11.5, 12.5, 13.5]  (all + 1.5)

    let loss2 = result2.sum();
    loss2.backward();
    println!("  col_vec gradient: {:?}", col_vec.get_grad().unwrap().to_vec());
    println!("  Expected col_vec grad: [4.0, 4.0, 4.0] (sum over 4 columns)");

    println!("\nTest 3: Scalar broadcast");
    let scalar = Tensor::new(&[10.0], vec![1], ctx.clone());
    let vec = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone());

    let result3 = vec.add(&scalar);
    println!("  scalar + vector: {:?}", result3.to_vec());
    println!("  Expected: [11.0, 12.0, 13.0, 14.0]");

    println!("\n=== All tests complete! ===");
}
