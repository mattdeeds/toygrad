// Reduction operations using workgroup shared memory

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let input_length = arrayLength(&input);

    // Load data into shared memory
    if (gid < input_length) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Reduce within workgroup
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && gid + stride < input_length) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write result from first thread in workgroup
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
