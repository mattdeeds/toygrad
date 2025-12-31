use wgpu::util::DeviceExt;

use crate::graph::GraphNode;
use crate::tensor::Tensor;

impl Tensor {
    /// Reshape tensor to a new shape
    /// Total number of elements must remain the same
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        assert_eq!(
            old_size, new_size,
            "Cannot reshape: total size must remain constant ({} != {})",
            old_size, new_size
        );

        // Reshape is just a view change - we can reuse the same buffer
        let mut result = Tensor {
            shape: new_shape.clone(),
            buffer: self.buffer.clone(),
            context: self.context.clone(),
            shared_data: crate::tensor::SharedTensorData::new(),
            node: GraphNode::leaf(),
            requires_grad: self.requires_grad,
        };

        // Add backward pass for reshape
        if self.requires_grad {
            let input_shared = std::sync::Arc::clone(&self.shared_data);
            let result_shared = std::sync::Arc::clone(&result.shared_data);
            let input_node = std::rc::Rc::clone(&self.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    // Gradient just needs to be reshaped back - data is identical
                    let grad_clone = grad_data.clone();
                    drop(grad_lock);

                    let mut input_grad = input_shared.grad.lock().unwrap();
                    if let Some(ref mut existing) = *input_grad {
                        for (e, g) in existing.iter_mut().zip(grad_clone.iter()) {
                            *e += g;
                        }
                    } else {
                        *input_grad = Some(grad_clone);
                    }
                    drop(input_grad);

                    // Recursively call backward on input
                    if let Some(ref backward) = input_node.backward_fn {
                        backward();
                    }
                }
            });

            result.node = GraphNode::new(backward_fn);
        }

        result
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "transpose currently only supports 2D tensors, got {}D",
            self.ndim()
        );

        let rows = self.shape[0] as u32;
        let cols = self.shape[1] as u32;

        let context = self.context.clone();
        let output_size = self.size();

        // Create output buffer
        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transpose Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer for dimensions
        let dims = [rows, cols, 0u32, 0u32]; // pad to 16 bytes
        let dims_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Transpose Dims"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group using cached layout
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Transpose Bind Group"),
                layout: &context.pipelines.transpose_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dims_buffer.as_entire_binding(),
                    },
                ],
            });

        // Execute using cached pipeline
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Transpose Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Transpose Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&context.pipelines.transpose_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_x = cols.div_ceil(8);
            let workgroup_y = rows.div_ceil(8);
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        context.queue.submit([encoder.finish()]);

        let mut result = Tensor {
            shape: vec![cols as usize, rows as usize],
            buffer: output_buffer,
            context,
            shared_data: crate::tensor::SharedTensorData::new(),
            node: GraphNode::leaf(),
            requires_grad: self.requires_grad,
        };

        // Add backward pass for transpose
        if self.requires_grad {
            let input_rows = rows as usize;
            let input_cols = cols as usize;
            let input_shared = std::sync::Arc::clone(&self.shared_data);
            let result_shared = std::sync::Arc::clone(&result.shared_data);
            let input_node = std::rc::Rc::clone(&self.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    // Transpose the gradient back: grad has shape [cols, rows], need [rows, cols]
                    let mut transposed_grad = vec![0.0f32; grad_data.len()];
                    for i in 0..input_cols {
                        for j in 0..input_rows {
                            // grad_data is [cols, rows], output is [rows, cols]
                            transposed_grad[j * input_cols + i] = grad_data[i * input_rows + j];
                        }
                    }
                    drop(grad_lock);

                    let mut input_grad = input_shared.grad.lock().unwrap();
                    if let Some(ref mut existing) = *input_grad {
                        for (e, g) in existing.iter_mut().zip(transposed_grad.iter()) {
                            *e += g;
                        }
                    } else {
                        *input_grad = Some(transposed_grad);
                    }
                    drop(input_grad);

                    // Recursively call backward on input
                    if let Some(ref backward) = input_node.backward_fn {
                        backward();
                    }
                }
            });

            result.node = GraphNode::new(backward_fn);
        }

        result
    }
}
