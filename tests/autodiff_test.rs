use toygrad::{GpuContext, Tensor};

fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

fn approx_eq_vec(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| approx_eq(*x, *y, epsilon))
}

#[test]
fn test_simple_add_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0], vec![2], ctx.clone()).with_grad();
    let b = Tensor::new(&[3.0, 4.0], vec![2], ctx.clone()).with_grad();

    let c = a.add(&b);
    let loss = c.sum(); // Create a scalar for backward
    loss.backward();

    // Gradient of addition is 1 for both inputs
    let grad_a = a.get_grad().unwrap().to_vec();
    let grad_b = b.get_grad().unwrap().to_vec();

    assert!(approx_eq_vec(&grad_a, &[1.0, 1.0], 1e-5));
    assert!(approx_eq_vec(&grad_b, &[1.0, 1.0], 1e-5));
}

#[test]
fn test_simple_mul_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[2.0, 3.0], vec![2], ctx.clone()).with_grad();
    let b = Tensor::new(&[4.0, 5.0], vec![2], ctx.clone()).with_grad();

    let c = a.mul(&b);
    let loss = c.sum();
    loss.backward();

    // Gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a
    let grad_a = a.get_grad().unwrap().to_vec();
    let grad_b = b.get_grad().unwrap().to_vec();

    assert!(approx_eq_vec(&grad_a, &[4.0, 5.0], 1e-5));
    assert!(approx_eq_vec(&grad_b, &[2.0, 3.0], 1e-5));
}

#[test]
fn test_simple_sub_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[10.0, 20.0], vec![2], ctx.clone()).with_grad();
    let b = Tensor::new(&[3.0, 5.0], vec![2], ctx.clone()).with_grad();

    let c = a.sub(&b);
    let loss = c.sum();
    loss.backward();

    // Gradient of subtraction: d(a-b)/da = 1, d(a-b)/db = -1
    let grad_a = a.get_grad().unwrap().to_vec();
    let grad_b = b.get_grad().unwrap().to_vec();

    assert!(approx_eq_vec(&grad_a, &[1.0, 1.0], 1e-5));
    assert!(approx_eq_vec(&grad_b, &[-1.0, -1.0], 1e-5));
}

#[test]
fn test_chain_rule() {
    let ctx = GpuContext::new();
    let x = Tensor::new(&[2.0], vec![1], ctx.clone()).with_grad();

    // y = x * x
    let y = x.mul(&x);
    // z = y + x
    let z = y.add(&x);

    z.backward();

    // dz/dx = dz/dy * dy/dx + dz/dx
    //       = 1 * 2*x + 1
    //       = 2*2 + 1 = 5
    let grad_x = x.get_grad().unwrap().to_vec();
    assert!(approx_eq(grad_x[0], 5.0, 1e-4));
}

#[test]
fn test_matmul_backward() {
    let ctx = GpuContext::new();
    // 2x2 matrix
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], ctx.clone()).with_grad();
    // 2x2 matrix
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], vec![2, 2], ctx.clone()).with_grad();

    let c = a.matmul(&b);
    let loss = c.sum();
    loss.backward();

    // For matmul: grad_a = grad_output @ b^T, grad_b = a^T @ grad_output
    let grad_a = a.get_grad().unwrap().to_vec();
    let grad_b = b.get_grad().unwrap().to_vec();

    // grad_output @ b^T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]]
    //                   = [[11, 15], [11, 15]]
    assert!(approx_eq_vec(&grad_a, &[11.0, 15.0, 11.0, 15.0], 1e-3));

    // a^T @ grad_output = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]]
    //                   = [[4, 4], [6, 6]]
    assert!(approx_eq_vec(&grad_b, &[4.0, 4.0, 6.0, 6.0], 1e-3));
}

#[test]
fn test_sum_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone()).with_grad();

    let sum = a.sum();
    sum.backward();

    // Gradient of sum distributes the output gradient to all inputs
    let grad_a = a.get_grad().unwrap().to_vec();
    assert!(approx_eq_vec(&grad_a, &[1.0, 1.0, 1.0, 1.0], 1e-5));
}

#[test]
fn test_mean_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone()).with_grad();

    let mean = a.mean();
    mean.backward();

    // Gradient of mean is 1/N for each element
    let grad_a = a.get_grad().unwrap().to_vec();
    assert!(approx_eq_vec(&grad_a, &[0.25, 0.25, 0.25, 0.25], 1e-5));
}

#[test]
fn test_relu_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], ctx.clone()).with_grad();

    let result = a.relu();
    let loss = result.sum();
    loss.backward();

    // ReLU gradient: 0 for x < 0, 1 for x > 0
    let grad_a = a.get_grad().unwrap().to_vec();
    assert!(approx_eq_vec(&grad_a, &[0.0, 0.0, 0.0, 1.0, 1.0], 1e-5));
}

#[test]
fn test_sigmoid_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[0.0], vec![1], ctx.clone()).with_grad();

    let result = a.sigmoid();
    result.backward();

    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    let grad_a = a.get_grad().unwrap().to_vec();
    assert!(approx_eq(grad_a[0], 0.25, 1e-5));
}

#[test]
fn test_tanh_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[0.0], vec![1], ctx.clone()).with_grad();

    let result = a.tanh();
    result.backward();

    // tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
    let grad_a = a.get_grad().unwrap().to_vec();
    assert!(approx_eq(grad_a[0], 1.0, 1e-5));
}

#[test]
fn test_neg_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0], vec![3], ctx.clone()).with_grad();

    let result = a.neg();
    let loss = result.sum();
    loss.backward();

    // Gradient of negation is -1
    let grad_a = a.get_grad().unwrap().to_vec();
    assert!(approx_eq_vec(&grad_a, &[-1.0, -1.0, -1.0], 1e-5));
}

#[test]
fn test_transpose_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], ctx.clone()).with_grad();

    let result = a.transpose();
    let loss = result.sum();
    loss.backward();

    // Gradient of transpose with sum is just all 1s
    let grad_a = a.get_grad().unwrap().to_vec();
    assert!(approx_eq_vec(&grad_a, &[1.0, 1.0, 1.0, 1.0], 1e-5));
}

#[test]
fn test_reshape_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], ctx.clone()).with_grad();

    let result = a.reshape(vec![4]);
    let loss = result.sum();
    loss.backward();

    // Gradient of reshape with sum is just all 1s
    let grad_a = a.get_grad().unwrap().to_vec();
    assert!(approx_eq_vec(&grad_a, &[1.0, 1.0, 1.0, 1.0], 1e-5));
}

#[test]
fn test_broadcast_backward() {
    let ctx = GpuContext::new();
    // Shape [2, 1]
    let a = Tensor::new(&[1.0, 2.0], vec![2, 1], ctx.clone()).with_grad();
    // Shape [1, 3]
    let b = Tensor::new(&[3.0, 4.0, 5.0], vec![1, 3], ctx.clone()).with_grad();

    let result = a.add(&b);
    // Result shape is [2, 3]
    let loss = result.sum();
    loss.backward();

    // Gradients should be reduced along broadcast dimensions
    let grad_a = a.get_grad().unwrap().to_vec();
    let grad_b = b.get_grad().unwrap().to_vec();

    // grad_a: sum along axis 1: [[3], [3]]
    assert!(approx_eq_vec(&grad_a, &[3.0, 3.0], 1e-5));
    // grad_b: sum along axis 0: [2, 2, 2]
    assert!(approx_eq_vec(&grad_b, &[2.0, 2.0, 2.0], 1e-5));
}

#[test]
fn test_zero_grad() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0], vec![2], ctx.clone()).with_grad();

    let b = a.mul(&a);
    let loss = b.sum();
    loss.backward();

    // Check gradient exists
    assert!(a.get_grad().is_some());

    // Zero the gradient
    a.zero_grad();

    // Check gradient is now None or zeros
    match a.get_grad() {
        None => assert!(true),
        Some(grad) => {
            let grad_vec = grad.to_vec();
            assert!(grad_vec.iter().all(|&x| x == 0.0));
        }
    }
}

#[test]
fn test_complex_computation_graph() {
    let ctx = GpuContext::new();
    let x = Tensor::new(&[2.0], vec![1], ctx.clone()).with_grad();

    // f(x) = (x + 2) * (x * 3) = (x + 2) * 3x = 3x^2 + 6x
    let a = x.add(&Tensor::new(&[2.0], vec![1], ctx.clone()));
    let b = x.mul(&Tensor::new(&[3.0], vec![1], ctx.clone()));
    let result = a.mul(&b);

    result.backward();

    // df/dx = 6x + 6 = 6*2 + 6 = 18
    let grad_x = x.get_grad().unwrap().to_vec();
    assert!(approx_eq(grad_x[0], 18.0, 1e-4));
}

#[test]
fn test_multiple_uses_of_tensor() {
    let ctx = GpuContext::new();
    let x = Tensor::new(&[3.0], vec![1], ctx.clone()).with_grad();

    // y = x + x + x = 3x
    let y = x.add(&x);
    let z = y.add(&x);

    z.backward();

    // dy/dx should be 3 (used three times)
    let grad_x = x.get_grad().unwrap().to_vec();
    assert!(approx_eq(grad_x[0], 3.0, 1e-4));
}

#[test]
fn test_div_backward() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[10.0], vec![1], ctx.clone()).with_grad();
    let b = Tensor::new(&[2.0], vec![1], ctx.clone()).with_grad();

    let c = a.div(&b);
    c.backward();

    // d(a/b)/da = 1/b = 1/2 = 0.5
    // d(a/b)/db = -a/b^2 = -10/4 = -2.5
    let grad_a = a.get_grad().unwrap().to_vec();
    let grad_b = b.get_grad().unwrap().to_vec();

    assert!(approx_eq(grad_a[0], 0.5, 1e-5));
    assert!(approx_eq(grad_b[0], -2.5, 1e-5));
}
