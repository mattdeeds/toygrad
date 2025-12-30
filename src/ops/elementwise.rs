use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::broadcast::{BroadcastInfo, compute_strides};
use crate::graph::GraphNode;
use crate::tensor::Tensor;

#[repr(u32)]
enum OpType {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
}

/// Reduce a tensor's gradient along specified dimensions by summing
fn reduce_gradient_dims(grad: Vec<f32>, shape: &[usize], reduce_dims: &[usize]) -> Vec<f32> {
    if reduce_dims.is_empty() {
        return grad;
    }

    // Start with the original gradient
    let mut current_data = grad;
    let mut current_shape = shape.to_vec();

    // Reduce each dimension (must do in reverse order to maintain indices)
    let mut sorted_dims = reduce_dims.to_vec();
    sorted_dims.sort_unstable();
    sorted_dims.reverse();

    for &dim in &sorted_dims {
        let dim_size = current_shape[dim];
        let outer_size: usize = current_shape[..dim].iter().product();
        let inner_size: usize = current_shape[(dim + 1)..].iter().product();

        let mut reduced = vec![0.0; outer_size * inner_size];

        for outer in 0..outer_size {
            for d in 0..dim_size {
                for inner in 0..inner_size {
                    let src_idx = outer * (dim_size * inner_size) + d * inner_size + inner;
                    let dst_idx = outer * inner_size + inner;
                    reduced[dst_idx] += current_data[src_idx];
                }
            }
        }

        current_data = reduced;
        current_shape.remove(dim);
        if current_shape.is_empty() {
            current_shape.push(1);
        }
    }

    current_data
}

/// Execute an element-wise binary operation on the GPU
fn elementwise_binary_op(a: &Tensor, b: &Tensor, op_type: OpType) -> Tensor {
    assert_eq!(
        a.shape, b.shape,
        "Shapes must match for element-wise operations: {:?} vs {:?}",
        a.shape, b.shape
    );

    let context = a.context.clone();
    let size = a.size();

    // Create output buffer
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Elementwise Output"),
        size: (size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniform buffer for op_type
    let op_type_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Op Type"),
            contents: bytemuck::cast_slice(&[op_type as u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create bind group using cached layout
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Elementwise Bind Group"),
            layout: &context.pipelines.elementwise_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: op_type_buffer.as_entire_binding(),
                },
            ],
        });

    // Execute using cached pipeline
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Elementwise Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Elementwise Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&context.pipelines.elementwise_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_count = size.div_ceil(64);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    }

    context.queue.submit([encoder.finish()]);

    Tensor {
        shape: a.shape.clone(),
        buffer: output_buffer,
        context,
        shared_data: crate::tensor::SharedTensorData::new(),
        node: GraphNode::leaf(),
        requires_grad: a.requires_grad || b.requires_grad,
    }
}

/// Execute a broadcast element-wise binary operation on the GPU
fn broadcast_binary_op(a: &Tensor, b: &Tensor, op_type: OpType) -> Tensor {
    let broadcast_info = BroadcastInfo::compute(&a.shape, &b.shape)
        .expect("Shapes are not broadcast-compatible");

    let context = a.context.clone();
    let output_size: usize = broadcast_info.output_shape.iter().product();

    // Create output buffer
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Broadcast Output"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Prepare broadcast parameters
    let strides_a = compute_strides(&broadcast_info.shape_a);
    let strides_b = compute_strides(&broadcast_info.shape_b);
    let ndim = broadcast_info.output_shape.len().min(4); // Max 4 dims for shader

    // Pack shapes and strides into vec4 (pad with 1s if needed)
    let mut output_shape_packed = [1u32; 4];
    let mut shape_a_packed = [1u32; 4];
    let mut shape_b_packed = [1u32; 4];
    let mut strides_a_packed = [1u32; 4];
    let mut strides_b_packed = [1u32; 4];

    for i in 0..ndim {
        output_shape_packed[i] = broadcast_info.output_shape[i] as u32;
        shape_a_packed[i] = broadcast_info.shape_a[i] as u32;
        shape_b_packed[i] = broadcast_info.shape_b[i] as u32;
        strides_a_packed[i] = strides_a[i] as u32;
        strides_b_packed[i] = strides_b[i] as u32;
    }

    // Create uniform buffer with broadcast parameters
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BroadcastParams {
        op_type: u32,
        ndim: u32,
        _padding: [u32; 2],
        output_shape: [u32; 4],
        shape_a: [u32; 4],
        shape_b: [u32; 4],
        strides_a: [u32; 4],
        strides_b: [u32; 4],
    }

    let params = BroadcastParams {
        op_type: op_type as u32,
        ndim: ndim as u32,
        _padding: [0, 0],
        output_shape: output_shape_packed,
        shape_a: shape_a_packed,
        shape_b: shape_b_packed,
        strides_a: strides_a_packed,
        strides_b: strides_b_packed,
    };

    let params_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Broadcast Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create bind group
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Broadcast Bind Group"),
            layout: &context.pipelines.broadcast_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Execute shader
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Broadcast Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Broadcast Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&context.pipelines.broadcast_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_count = output_size.div_ceil(64);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    }

    context.queue.submit([encoder.finish()]);

    Tensor {
        shape: broadcast_info.output_shape,
        buffer: output_buffer,
        context,
        shared_data: crate::tensor::SharedTensorData::new(),
        node: GraphNode::leaf(),
        requires_grad: a.requires_grad || b.requires_grad,
    }
}

// Public API for element-wise operations with backward pass
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        // Check if broadcasting is needed
        let broadcast_info = BroadcastInfo::compute(&self.shape, &other.shape);

        let mut result = if let Some(ref info) = broadcast_info {
            if info.needs_broadcast() {
                broadcast_binary_op(self, other, OpType::Add)
            } else {
                elementwise_binary_op(self, other, OpType::Add)
            }
        } else {
            panic!("Shapes {:?} and {:?} are not broadcast-compatible", self.shape, other.shape);
        };

        if self.requires_grad || other.requires_grad {
            let a_requires_grad = self.requires_grad;
            let b_requires_grad = other.requires_grad;
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();

            // Get reduction dimensions for gradient accumulation
            let (reduce_dims_a, reduce_dims_b) = if let Some(ref info) = broadcast_info {
                info.get_reduction_dims(&a_shape, &b_shape)
            } else {
                (vec![], vec![])
            };

            // Clone Arc references for safe sharing
            let a_shared = Arc::clone(&self.shared_data);
            let b_shared = Arc::clone(&other.shared_data);
            let result_shared = Arc::clone(&result.shared_data);

            // Clone the nodes for recursive backward calls
            let a_node = Arc::clone(&self.node);
            let b_node = Arc::clone(&other.node);
            let output_shape = result.shape.clone();

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    let grad_clone = grad_data.clone();
                    drop(grad_lock);

                    if a_requires_grad {
                        // Reduce gradient if broadcasting occurred
                        let grad_a = reduce_gradient_dims(grad_clone.clone(), &output_shape, &reduce_dims_a);

                        let mut a_grad = a_shared.grad.lock().unwrap();
                        if let Some(ref mut existing) = *a_grad {
                            for (e, g) in existing.iter_mut().zip(grad_a.iter()) {
                                *e += g;
                            }
                        } else {
                            *a_grad = Some(grad_a);
                        }
                        drop(a_grad);

                        // Recursively call backward on input
                        if let Some(ref backward) = a_node.backward_fn {
                            backward();
                        }
                    }
                    if b_requires_grad {
                        // Reduce gradient if broadcasting occurred
                        let grad_b = reduce_gradient_dims(grad_clone, &output_shape, &reduce_dims_b);

                        let mut b_grad = b_shared.grad.lock().unwrap();
                        if let Some(ref mut existing) = *b_grad {
                            for (e, g) in existing.iter_mut().zip(grad_b.iter()) {
                                *e += g;
                            }
                        } else {
                            *b_grad = Some(grad_b);
                        }
                        drop(b_grad);

                        // Recursively call backward on input
                        if let Some(ref backward) = b_node.backward_fn {
                            backward();
                        }
                    }
                }
            });

            result.node = crate::graph::GraphNode::new(backward_fn);
        }

        result
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let mut result = elementwise_binary_op(self, other, OpType::Sub);

        if self.requires_grad || other.requires_grad {
            let a_requires_grad = self.requires_grad;
            let b_requires_grad = other.requires_grad;

            let a_shared = Arc::clone(&self.shared_data);
            let b_shared = Arc::clone(&other.shared_data);
            let result_shared = Arc::clone(&result.shared_data);

            let a_node = Arc::clone(&self.node);
            let b_node = Arc::clone(&other.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    let grad_clone = grad_data.clone();
                    drop(grad_lock);

                    if a_requires_grad {
                        let mut a_grad = a_shared.grad.lock().unwrap();
                        if let Some(ref mut existing) = *a_grad {
                            for (e, g) in existing.iter_mut().zip(grad_clone.iter()) {
                                *e += g;
                            }
                        } else {
                            *a_grad = Some(grad_clone.clone());
                        }
                        drop(a_grad);
                        if let Some(ref backward) = a_node.backward_fn {
                            backward();
                        }
                    }
                    if b_requires_grad {
                        let neg_grad: Vec<f32> = grad_clone.iter().map(|x| -x).collect();
                        let mut b_grad = b_shared.grad.lock().unwrap();
                        if let Some(ref mut existing) = *b_grad {
                            for (e, g) in existing.iter_mut().zip(neg_grad.iter()) {
                                *e += g;
                            }
                        } else {
                            *b_grad = Some(neg_grad);
                        }
                        drop(b_grad);
                        if let Some(ref backward) = b_node.backward_fn {
                            backward();
                        }
                    }
                }
            });

            result.node = crate::graph::GraphNode::new(backward_fn);
        }

        result
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let mut result = elementwise_binary_op(self, other, OpType::Mul);

        if self.requires_grad || other.requires_grad {
            let a_requires_grad = self.requires_grad;
            let b_requires_grad = other.requires_grad;

            let a_data = self.to_vec();
            let b_data = other.to_vec();

            let a_shared = Arc::clone(&self.shared_data);
            let b_shared = Arc::clone(&other.shared_data);
            let result_shared = Arc::clone(&result.shared_data);

            // Clone the nodes for recursive backward calls
            let a_node = Arc::clone(&self.node);
            let b_node = Arc::clone(&other.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    let grad_clone = grad_data.clone();
                    drop(grad_lock);

                    if a_requires_grad {
                        let grad_a: Vec<f32> = grad_clone.iter()
                            .zip(b_data.iter())
                            .map(|(g, b)| g * b)
                            .collect();
                        let mut a_grad = a_shared.grad.lock().unwrap();
                        if let Some(ref mut existing) = *a_grad {
                            for (e, g) in existing.iter_mut().zip(grad_a.iter()) {
                                *e += g;
                            }
                        } else {
                            *a_grad = Some(grad_a);
                        }
                        drop(a_grad);

                        // Recursively call backward on input
                        if let Some(ref backward) = a_node.backward_fn {
                            backward();
                        }
                    }
                    if b_requires_grad {
                        let grad_b: Vec<f32> = grad_clone.iter()
                            .zip(a_data.iter())
                            .map(|(g, a)| g * a)
                            .collect();
                        let mut b_grad = b_shared.grad.lock().unwrap();
                        if let Some(ref mut existing) = *b_grad {
                            for (e, g) in existing.iter_mut().zip(grad_b.iter()) {
                                *e += g;
                            }
                        } else {
                            *b_grad = Some(grad_b);
                        }
                        drop(b_grad);

                        // Recursively call backward on input
                        if let Some(ref backward) = b_node.backward_fn {
                            backward();
                        }
                    }
                }
            });

            result.node = crate::graph::GraphNode::new(backward_fn);
        }

        result
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        let mut result = elementwise_binary_op(self, other, OpType::Div);

        if self.requires_grad || other.requires_grad {
            let a_requires_grad = self.requires_grad;
            let b_requires_grad = other.requires_grad;

            let a_data = self.to_vec();
            let b_data = other.to_vec();

            let a_shared = Arc::clone(&self.shared_data);
            let b_shared = Arc::clone(&other.shared_data);
            let result_shared = Arc::clone(&result.shared_data);

            // Clone the nodes for recursive backward calls
            let a_node = Arc::clone(&self.node);
            let b_node = Arc::clone(&other.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    let grad_clone = grad_data.clone();
                    drop(grad_lock);

                    if a_requires_grad {
                        let grad_a: Vec<f32> = grad_clone.iter()
                            .zip(b_data.iter())
                            .map(|(g, b)| g / b)
                            .collect();
                        let mut a_grad = a_shared.grad.lock().unwrap();
                        if let Some(ref mut existing) = *a_grad {
                            for (e, g) in existing.iter_mut().zip(grad_a.iter()) {
                                *e += g;
                            }
                        } else {
                            *a_grad = Some(grad_a);
                        }
                        drop(a_grad);

                        // Recursively call backward on input
                        if let Some(ref backward) = a_node.backward_fn {
                            backward();
                        }
                    }
                    if b_requires_grad {
                        let grad_b: Vec<f32> = grad_clone.iter()
                            .zip(a_data.iter())
                            .zip(b_data.iter())
                            .map(|((g, a), b)| -g * a / (b * b))
                            .collect();
                        let mut b_grad = b_shared.grad.lock().unwrap();
                        if let Some(ref mut existing) = *b_grad {
                            for (e, g) in existing.iter_mut().zip(grad_b.iter()) {
                                *e += g;
                            }
                        } else {
                            *b_grad = Some(grad_b);
                        }
                        drop(b_grad);

                        // Recursively call backward on input
                        if let Some(ref backward) = b_node.backward_fn {
                            backward();
                        }
                    }
                }
            });

            result.node = crate::graph::GraphNode::new(backward_fn);
        }

        result
    }

    pub fn neg(&self) -> Tensor {
        let context = self.context.clone();
        let size = self.size();

        // Create output buffer
        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Negate Output"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group using cached layout
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Negate Bind Group"),
                layout: &context.pipelines.negate_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        // Execute using cached pipeline
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Negate Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Negate Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&context.pipelines.negate_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = size.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        context.queue.submit([encoder.finish()]);

        let mut result = Tensor {
            shape: self.shape.clone(),
            buffer: output_buffer,
            context,
            shared_data: crate::tensor::SharedTensorData::new(),
            node: GraphNode::leaf(),
            requires_grad: self.requires_grad,
        };

        if self.requires_grad {
            let input_shared = Arc::clone(&self.shared_data);
            let result_shared = Arc::clone(&result.shared_data);

            // Clone the node for recursive backward call
            let input_node = Arc::clone(&self.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    let neg_grad: Vec<f32> = grad_data.iter().map(|x| -x).collect();
                    drop(grad_lock);
                    let mut input_grad = input_shared.grad.lock().unwrap();
                    if let Some(ref mut existing) = *input_grad {
                        for (e, g) in existing.iter_mut().zip(neg_grad.iter()) {
                            *e += g;
                        }
                    } else {
                        *input_grad = Some(neg_grad);
                    }
                    drop(input_grad);

                    // Recursively call backward on input
                    if let Some(ref backward) = input_node.backward_fn {
                        backward();
                    }
                }
            });

            result.node = crate::graph::GraphNode::new(backward_fn);
        }

        result
    }
}
