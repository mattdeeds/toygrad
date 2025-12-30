// Activation functions

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let array_length = arrayLength(&output);

    if (index >= array_length) {
        return;
    }

    output[index] = max(0.0, input[index]);
}

@compute @workgroup_size(64)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let array_length = arrayLength(&output);

    if (index >= array_length) {
        return;
    }

    output[index] = 1.0 / (1.0 + exp(-input[index]));
}

@compute @workgroup_size(64)
fn tanh_activation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let array_length = arrayLength(&output);

    if (index >= array_length) {
        return;
    }

    output[index] = tanh(input[index]);
}
