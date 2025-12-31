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
fn test_tensor_creation() {
    let ctx = GpuContext::new();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::new(&data, vec![2, 2], ctx.clone());

    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.size(), 4);
    assert_eq!(t.ndim(), 2);

    let result = t.to_vec();
    assert_eq!(result, data);
}

#[test]
fn test_zeros() {
    let ctx = GpuContext::new();
    let zeros = Tensor::zeros(vec![3, 2], ctx.clone());

    assert_eq!(zeros.shape, vec![3, 2]);
    let result = zeros.to_vec();
    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn test_ones() {
    let ctx = GpuContext::new();
    let ones = Tensor::ones(vec![2, 3], ctx.clone());

    assert_eq!(ones.shape, vec![2, 3]);
    let result = ones.to_vec();
    assert!(result.iter().all(|&x| x == 1.0));
}

#[test]
fn test_full() {
    let ctx = GpuContext::new();
    let full = Tensor::full(vec![2, 2], 5.5, ctx.clone());

    let result = full.to_vec();
    assert!(result.iter().all(|&x| approx_eq(x, 5.5, 1e-5)));
}

#[test]
fn test_add() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone());
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], vec![4], ctx.clone());

    let result = a.add(&b);
    let expected = vec![6.0, 8.0, 10.0, 12.0];

    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_sub() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[10.0, 8.0, 6.0, 4.0], vec![4], ctx.clone());
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone());

    let result = a.sub(&b);
    let expected = vec![9.0, 6.0, 3.0, 0.0];

    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_mul() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone());
    let b = Tensor::new(&[2.0, 3.0, 4.0, 5.0], vec![4], ctx.clone());

    let result = a.mul(&b);
    let expected = vec![2.0, 6.0, 12.0, 20.0];

    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_div() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[10.0, 20.0, 30.0, 40.0], vec![4], ctx.clone());
    let b = Tensor::new(&[2.0, 4.0, 5.0, 8.0], vec![4], ctx.clone());

    let result = a.div(&b);
    let expected = vec![5.0, 5.0, 6.0, 5.0];

    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_neg() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, -2.0, 3.0, -4.0], vec![4], ctx.clone());

    let result = a.neg();
    let expected = vec![-1.0, 2.0, -3.0, 4.0];

    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_matmul() {
    let ctx = GpuContext::new();
    // 2x3 matrix
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], ctx.clone());
    // 3x2 matrix
    let b = Tensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], ctx.clone());

    let result = a.matmul(&b);

    assert_eq!(result.shape, vec![2, 2]);

    // Expected: [[58, 64], [139, 154]]
    let expected = vec![58.0, 64.0, 139.0, 154.0];
    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-3));
}

#[test]
fn test_transpose() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], ctx.clone());

    let result = a.transpose();

    assert_eq!(result.shape, vec![3, 2]);

    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_reshape() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], ctx.clone());

    let result = a.reshape(vec![3, 2]);

    assert_eq!(result.shape, vec![3, 2]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_sum() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone());

    let result = a.sum();

    assert_eq!(result.shape, vec![1]);
    assert!(approx_eq(result.to_vec()[0], 10.0, 1e-5));
}

#[test]
fn test_mean() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone());

    let result = a.mean();

    assert_eq!(result.shape, vec![1]);
    assert!(approx_eq(result.to_vec()[0], 2.5, 1e-5));
}

#[test]
fn test_relu() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], ctx.clone());

    let result = a.relu();
    let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_sigmoid() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[0.0, 1.0, -1.0], vec![3], ctx.clone());

    let result = a.sigmoid();
    let result_vec = result.to_vec();

    // sigmoid(0) = 0.5
    assert!(approx_eq(result_vec[0], 0.5, 1e-5));
    // sigmoid(1) ≈ 0.7311
    assert!(approx_eq(result_vec[1], 0.7310586, 1e-5));
    // sigmoid(-1) ≈ 0.2689
    assert!(approx_eq(result_vec[2], 0.26894143, 1e-5));
}

#[test]
fn test_tanh() {
    let ctx = GpuContext::new();
    let a = Tensor::new(&[0.0, 1.0, -1.0], vec![3], ctx.clone());

    let result = a.tanh();
    let result_vec = result.to_vec();

    // tanh(0) = 0
    assert!(approx_eq(result_vec[0], 0.0, 1e-5));
    // tanh(1) ≈ 0.7616
    assert!(approx_eq(result_vec[1], 0.7615942, 1e-5));
    // tanh(-1) ≈ -0.7616
    assert!(approx_eq(result_vec[2], -0.7615942, 1e-5));
}

#[test]
fn test_broadcast_add() {
    let ctx = GpuContext::new();
    // Shape [2, 3]
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], ctx.clone());
    // Shape [3] - will broadcast to [1, 3]
    let b = Tensor::new(&[10.0, 20.0, 30.0], vec![3], ctx.clone());

    let result = a.add(&b);

    assert_eq!(result.shape, vec![2, 3]);
    let expected = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_broadcast_scalar() {
    let ctx = GpuContext::new();
    // Shape [3]
    let a = Tensor::new(&[1.0, 2.0, 3.0], vec![3], ctx.clone());
    // Shape [1] - scalar broadcast
    let b = Tensor::new(&[10.0], vec![1], ctx.clone());

    let result = a.add(&b);

    assert_eq!(result.shape, vec![3]);
    let expected = vec![11.0, 12.0, 13.0];
    assert!(approx_eq_vec(&result.to_vec(), &expected, 1e-5));
}

#[test]
fn test_update_data() {
    let ctx = GpuContext::new();
    let mut t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], vec![4], ctx.clone());

    t.update_data(&[5.0, 6.0, 7.0, 8.0]);

    let result = t.to_vec();
    assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
}
