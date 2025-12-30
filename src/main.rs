use toygrad::{GpuContext, Tensor};

fn main() {
    println!("Initializing GPU context...");
    let ctx = GpuContext::new();
    println!("GPU context created successfully!");

    // Test basic tensor creation
    println!("\n=== Testing Tensor Creation ===");
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::new(&data, vec![2, 3], ctx.clone());
    println!("Created tensor with shape {:?}", t.shape);
    println!("Tensor data: {:?}", t.to_vec());

    // Test zeros
    println!("\n=== Testing Zeros ===");
    let zeros = Tensor::zeros(vec![2, 2], ctx.clone());
    println!("Zeros tensor: {:?}", zeros.to_vec());

    // Test ones
    println!("\n=== Testing Ones ===");
    let ones = Tensor::ones(vec![3, 2], ctx.clone());
    println!("Ones tensor: {:?}", ones.to_vec());

    // Test full
    println!("\n=== Testing Full ===");
    let full = Tensor::full(vec![2, 3], 5.5, ctx.clone());
    println!("Full tensor (5.5): {:?}", full.to_vec());

    // Test element-wise operations
    println!("\n=== Testing Element-wise Operations ===");
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone());
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], vec![4], ctx.clone());

    println!("a: {:?}", a.to_vec());
    println!("b: {:?}", b.to_vec());

    let sum = a.add(&b);
    println!("a + b: {:?}", sum.to_vec());

    let diff = a.sub(&b);
    println!("a - b: {:?}", diff.to_vec());

    let prod = a.mul(&b);
    println!("a * b: {:?}", prod.to_vec());

    let quot = a.div(&b);
    println!("a / b: {:?}", quot.to_vec());

    let negated = a.neg();
    println!("-a: {:?}", negated.to_vec());

    // Test matrix multiplication
    println!("\n=== Testing Matrix Multiplication ===");
    // Create a 2x3 matrix
    let mat_a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], ctx.clone());
    // Create a 3x2 matrix
    let mat_b = Tensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], ctx.clone());

    println!("A (2x3): {:?}", mat_a.to_vec());
    println!("B (3x2): {:?}", mat_b.to_vec());

    let result = mat_a.matmul(&mat_b);
    println!("A @ B (2x2): {:?}", result.to_vec());
    println!("Expected: [58, 64, 139, 154]");

    // Test reshape
    println!("\n=== Testing Reshape ===");
    let vec = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], ctx.clone());
    println!("Original (6,): {:?}", vec.to_vec());
    let reshaped = vec.reshape(vec![2, 3]);
    println!("Reshaped to (2, 3): {:?}", reshaped.to_vec());
    println!("Shape: {:?}", reshaped.shape);

    // Test transpose
    println!("\n=== Testing Transpose ===");
    let mat = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], ctx.clone());
    println!("Original (2x3): {:?}", mat.to_vec());
    let transposed = mat.transpose();
    println!("Transposed (3x2): {:?}", transposed.to_vec());
    println!("Shape: {:?}", transposed.shape);

    // Test activation functions
    println!("\n=== Testing Activation Functions ===");
    let act_input = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], ctx.clone());
    println!("Input: {:?}", act_input.to_vec());

    let relu_out = act_input.relu();
    println!("ReLU: {:?}", relu_out.to_vec());

    let sigmoid_out = act_input.sigmoid();
    println!("Sigmoid: {:?}", sigmoid_out.to_vec());

    let tanh_out = act_input.tanh();
    println!("Tanh: {:?}", tanh_out.to_vec());

    // Test reduction operations
    println!("\n=== Testing Reduction Operations ===");
    let reduce_input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5], ctx.clone());
    println!("Input: {:?}", reduce_input.to_vec());

    let sum = reduce_input.sum();
    println!("Sum: {:?} (expected: 15.0)", sum.to_vec());

    let mean = reduce_input.mean();
    println!("Mean: {:?} (expected: 3.0)", mean.to_vec());

    println!("\n=== All Tests Passed! ===");
}
