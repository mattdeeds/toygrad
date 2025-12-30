// Transpose operation for 2D tensors

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// Original dimensions: rows, cols
@group(0) @binding(2)
var<uniform> dims: vec2<u32>;

@compute @workgroup_size(8, 8)
fn transpose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rows = dims.x;
    let cols = dims.y;

    let row = global_id.y;
    let col = global_id.x;

    if (row >= rows || col >= cols) {
        return;
    }

    let input_idx = row * cols + col;
    let output_idx = col * rows + row;

    output[output_idx] = input[input_idx];
}
