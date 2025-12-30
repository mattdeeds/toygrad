// Matrix multiplication shader
// Computes C = A @ B where A is (M, K) and B is (K, N), result is (M, N)

@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> matrix_c: array<f32>;

// Dimensions: M, K, N
@group(0) @binding(3)
var<uniform> dims: vec3<u32>;

@compute @workgroup_size(8, 8)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < K; k++) {
        let a_idx = row * K + k;
        let b_idx = k * N + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }

    let c_idx = row * N + col;
    matrix_c[c_idx] = sum;
}
