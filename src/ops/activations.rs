use crate::graph::GraphNode;
use crate::tensor::Tensor;

fn activation_op(input: &Tensor, pipeline: &wgpu::ComputePipeline) -> Tensor {
    let context = input.context.clone();
    let size = input.size();

    // Create output buffer
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Activation Output"),
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
            label: Some("Activation Bind Group"),
            layout: &context.pipelines.activation_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
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
            label: Some("Activation Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Activation Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_count = size.div_ceil(64);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    }

    context.queue.submit([encoder.finish()]);

    Tensor {
        shape: input.shape.clone(),
        buffer: output_buffer,
        context,
        shared_data: crate::tensor::SharedTensorData::new(),
        node: GraphNode::leaf(),
        requires_grad: input.requires_grad,
    }
}

impl Tensor {
    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        let mut result = activation_op(self, &self.context.pipelines.relu_pipeline);

        if self.requires_grad {
            let input_data = self.to_vec();
            let input_shared = std::sync::Arc::clone(&self.shared_data);
            let result_shared = std::sync::Arc::clone(&result.shared_data);

            // Clone the node for recursive backward call
            let input_node = std::sync::Arc::clone(&self.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    // ReLU gradient: grad * (input > 0)
                    let grad_input: Vec<f32> = grad_data
                        .iter()
                        .zip(input_data.iter())
                        .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                        .collect();
                    drop(grad_lock);

                    let mut input_grad = input_shared.grad.lock().unwrap();
                    if let Some(ref mut existing) = *input_grad {
                        for (e, g) in existing.iter_mut().zip(grad_input.iter()) {
                            *e += g;
                        }
                    } else {
                        *input_grad = Some(grad_input);
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

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor {
        let mut result = activation_op(self, &self.context.pipelines.sigmoid_pipeline);

        if self.requires_grad {
            let output_data = result.to_vec();
            let input_shared = std::sync::Arc::clone(&self.shared_data);
            let result_shared = std::sync::Arc::clone(&result.shared_data);

            // Clone the node for recursive backward call
            let input_node = std::sync::Arc::clone(&self.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    // Sigmoid gradient: grad * output * (1 - output)
                    let grad_input: Vec<f32> = grad_data
                        .iter()
                        .zip(output_data.iter())
                        .map(|(g, s)| g * s * (1.0 - s))
                        .collect();
                    drop(grad_lock);

                    let mut input_grad = input_shared.grad.lock().unwrap();
                    if let Some(ref mut existing) = *input_grad {
                        for (e, g) in existing.iter_mut().zip(grad_input.iter()) {
                            *e += g;
                        }
                    } else {
                        *input_grad = Some(grad_input);
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

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        let mut result = activation_op(self, &self.context.pipelines.tanh_pipeline);

        if self.requires_grad {
            let output_data = result.to_vec();
            let input_shared = std::sync::Arc::clone(&self.shared_data);
            let result_shared = std::sync::Arc::clone(&result.shared_data);

            // Clone the node for recursive backward call
            let input_node = std::sync::Arc::clone(&self.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    // Tanh gradient: grad * (1 - output^2)
                    let grad_input: Vec<f32> = grad_data
                        .iter()
                        .zip(output_data.iter())
                        .map(|(g, t)| g * (1.0 - t * t))
                        .collect();
                    drop(grad_lock);

                    let mut input_grad = input_shared.grad.lock().unwrap();
                    if let Some(ref mut existing) = *input_grad {
                        for (e, g) in existing.iter_mut().zip(grad_input.iter()) {
                            *e += g;
                        }
                    } else {
                        *input_grad = Some(grad_input);
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
