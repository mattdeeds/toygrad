// Broadcasting element-wise operations shader
// Supports add, sub, mul, div with automatic broadcasting

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct BroadcastParams {
    op_type: u32,           // 0=add, 1=sub, 2=mul, 3=div
    ndim: u32,              // number of dimensions
    // Shapes: output, a, b (max 4 dims, padded with 1s)
    output_shape: vec4<u32>,
    shape_a: vec4<u32>,
    shape_b: vec4<u32>,
    // Strides for indexing
    strides_a: vec4<u32>,
    strides_b: vec4<u32>,
}

@group(0) @binding(3) var<uniform> params: BroadcastParams;

// Convert linear output index to multi-dimensional index
fn linear_to_multi(idx: u32, shape: vec4<u32>, ndim: u32) -> vec4<u32> {
    var index = idx;
    var indices = vec4<u32>(0u, 0u, 0u, 0u);

    if ndim >= 4u {
        indices.w = index % shape.w;
        index = index / shape.w;
    }
    if ndim >= 3u {
        indices.z = index % shape.z;
        index = index / shape.z;
    }
    if ndim >= 2u {
        indices.y = index % shape.y;
        index = index / shape.y;
    }
    if ndim >= 1u {
        indices.x = index;
    }

    return indices;
}

// Convert multi-dimensional index to linear index using strides
fn multi_to_linear(indices: vec4<u32>, strides: vec4<u32>, shape: vec4<u32>, ndim: u32) -> u32 {
    var linear = 0u;

    // Broadcast by clamping indices to shape (if shape[i] == 1, index will be 0)
    var clamped = vec4<u32>(
        min(indices.x, shape.x - 1u),
        min(indices.y, shape.y - 1u),
        min(indices.z, shape.z - 1u),
        min(indices.w, shape.w - 1u)
    );

    if ndim >= 1u { linear += clamped.x * strides.x; }
    if ndim >= 2u { linear += clamped.y * strides.y; }
    if ndim >= 3u { linear += clamped.z * strides.z; }
    if ndim >= 4u { linear += clamped.w * strides.w; }

    return linear;
}

@compute @workgroup_size(64)
fn broadcast_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;

    // Calculate output size
    var output_size = 1u;
    for (var i = 0u; i < params.ndim; i++) {
        if i == 0u { output_size *= params.output_shape.x; }
        else if i == 1u { output_size *= params.output_shape.y; }
        else if i == 2u { output_size *= params.output_shape.z; }
        else if i == 3u { output_size *= params.output_shape.w; }
    }

    if output_idx >= output_size {
        return;
    }

    // Convert output index to multi-dimensional
    let multi_idx = linear_to_multi(output_idx, params.output_shape, params.ndim);

    // Get corresponding indices in input tensors (with broadcasting)
    let idx_a = multi_to_linear(multi_idx, params.strides_a, params.shape_a, params.ndim);
    let idx_b = multi_to_linear(multi_idx, params.strides_b, params.shape_b, params.ndim);

    let a = input_a[idx_a];
    let b = input_b[idx_b];

    var result: f32;
    switch params.op_type {
        case 0u: { result = a + b; }      // add
        case 1u: { result = a - b; }      // sub
        case 2u: { result = a * b; }      // mul
        case 3u: { result = a / b; }      // div
        default: { result = 0.0; }
    }

    output[output_idx] = result;
}
