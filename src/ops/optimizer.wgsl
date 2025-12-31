// GPU-based optimizer kernels
// Eliminates CPU-GPU round trips by performing parameter updates directly on GPU

struct OptimizerParams {
    size: u32,
    lr: f32,
    momentum: f32,      // For SGD with momentum (unused in basic SGD)
    beta1: f32,         // For Adam
    beta2: f32,         // For Adam
    epsilon: f32,       // For Adam
    t: u32,             // Timestep for Adam bias correction
}

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<uniform> params: OptimizerParams;

// SGD: param = param - lr * grad
@compute @workgroup_size(256)
fn sgd_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    param[idx] = param[idx] - params.lr * grad[idx];
}

// Adam optimizer state buffers
@group(0) @binding(3) var<storage, read_write> m: array<f32>;  // First moment
@group(0) @binding(4) var<storage, read_write> v: array<f32>;  // Second moment

// Adam: Full adaptive learning rate optimization
@compute @workgroup_size(256)
fn adam_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let g = grad[idx];

    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
    m[idx] = params.beta1 * m[idx] + (1.0 - params.beta1) * g;

    // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
    v[idx] = params.beta2 * v[idx] + (1.0 - params.beta2) * g * g;

    // Compute bias-corrected first moment estimate
    let m_hat = m[idx] / (1.0 - pow(params.beta1, f32(params.t)));

    // Compute bias-corrected second raw moment estimate
    let v_hat = v[idx] / (1.0 - pow(params.beta2, f32(params.t)));

    // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    param[idx] = param[idx] - params.lr * m_hat / (sqrt(v_hat) + params.epsilon);
}
