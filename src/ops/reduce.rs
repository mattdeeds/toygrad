use crate::graph::GraphNode;
use crate::tensor::Tensor;

impl Tensor {
    /// Sum all elements in the tensor to produce a scalar
    pub fn sum(&self) -> Tensor {
        let context = self.context.clone();
        let size = self.size();

        // For small tensors, just read to CPU and compute
        // For a minimal implementation, this is acceptable
        if size <= 1024 {
            let data = self.to_vec();
            let sum_value: f32 = data.iter().sum();
            let mut result = Tensor::new(&[sum_value], vec![1], context.clone());
            result.requires_grad = self.requires_grad;

            // Add backward pass for sum
            if self.requires_grad {
                let input_shape = self.shape.clone();
                let input_shared = std::sync::Arc::clone(&self.shared_data);
                let result_shared = std::sync::Arc::clone(&result.shared_data);

                // Clone the node for recursive backward call
                let input_node = std::sync::Arc::clone(&self.node);

                let backward_fn = Box::new(move || {
                    let grad_lock = result_shared.grad.lock().unwrap();
                    if let Some(ref grad_data) = *grad_lock {
                        let grad_value = grad_data[0];
                        // Gradient of sum: broadcast scalar gradient to all input elements
                        let input_size: usize = input_shape.iter().product();
                        let grad_input = vec![grad_value; input_size];
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

            return result;
        }

        // For larger tensors, use GPU reduction
        let workgroup_size = 256;
        let num_workgroups = size.div_ceil(workgroup_size);

        // Create output buffer for partial sums
        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduction Output"),
            size: (num_workgroups * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group using cached layout
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Reduce Bind Group"),
                layout: &context.pipelines.reduce_layout,
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
                label: Some("Reduce Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reduce Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&context.pipelines.sum_reduce_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        context.queue.submit([encoder.finish()]);

        // Create temporary tensor with partial results
        let partial_result = Tensor {
            shape: vec![num_workgroups],
            buffer: output_buffer,
            context: context.clone(),
            shared_data: crate::tensor::SharedTensorData::new(),
            node: GraphNode::leaf(),
            requires_grad: false,
        };

        // Sum the partial results on CPU
        let partial_data = partial_result.to_vec();
        let final_sum: f32 = partial_data.iter().sum();

        let mut result = Tensor::new(&[final_sum], vec![1], context);
        result.requires_grad = self.requires_grad;

        // Add backward pass for sum
        if self.requires_grad {
            let input_shape = self.shape.clone();
            let input_shared = std::sync::Arc::clone(&self.shared_data);
            let result_shared = std::sync::Arc::clone(&result.shared_data);

            // Clone the node for recursive backward call
            let input_node = std::sync::Arc::clone(&self.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    let grad_value = grad_data[0];
                    // Gradient of sum: broadcast scalar gradient to all input elements
                    let input_size: usize = input_shape.iter().product();
                    let grad_input = vec![grad_value; input_size];
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

    /// Compute the mean of all elements in the tensor
    pub fn mean(&self) -> Tensor {
        let size = self.size();
        let sum = self.sum();
        let size_f32 = size as f32;

        // Read sum value
        let sum_data = sum.to_vec();
        let mean_value = sum_data[0] / size_f32;

        let mut result = Tensor::new(&[mean_value], vec![1], self.context.clone());
        result.requires_grad = self.requires_grad;

        // Add backward pass for mean
        if self.requires_grad {
            let input_shape = self.shape.clone();
            let input_shared = std::sync::Arc::clone(&self.shared_data);
            let result_shared = std::sync::Arc::clone(&result.shared_data);

            // Clone the node for recursive backward call
            let input_node = std::sync::Arc::clone(&self.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    let grad_value = grad_data[0];
                    // Gradient of mean: broadcast scalar gradient / size to all elements
                    let input_size: usize = input_shape.iter().product();
                    let grad_per_element = grad_value / input_size as f32;
                    let grad_input = vec![grad_per_element; input_size];
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
