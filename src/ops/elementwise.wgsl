// Element-wise operations shader

@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

// Operation type: 0=add, 1=sub, 2=mul, 3=div
@group(0) @binding(3)
var<uniform> op_type: u32;

@compute @workgroup_size(64)
fn elementwise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let array_length = arrayLength(&output);

    if (index >= array_length) {
        return;
    }

    let a = input_a[index];
    let b = input_b[index];

    switch (op_type) {
        case 0u: {  // add
            output[index] = a + b;
        }
        case 1u: {  // sub
            output[index] = a - b;
        }
        case 2u: {  // mul
            output[index] = a * b;
        }
        case 3u: {  // div
            output[index] = a / b;
        }
        default: {
            output[index] = 0.0;
        }
    }
}

// Unary operations
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_unary: array<f32>;

@compute @workgroup_size(64)
fn negate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let array_length = arrayLength(&output_unary);

    if (index >= array_length) {
        return;
    }

    output_unary[index] = -input[index];
}
