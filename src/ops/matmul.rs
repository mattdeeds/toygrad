use wgpu::util::DeviceExt;

use crate::graph::GraphNode;
use crate::tensor::Tensor;

impl Tensor {
    /// Matrix multiplication: self @ other
    /// self must be 2D with shape (M, K)
    /// other must be 2D with shape (K, N)
    /// Result will be (M, N)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "matmul requires 2D tensors, got {}D",
            self.ndim()
        );
        assert_eq!(
            other.ndim(),
            2,
            "matmul requires 2D tensors, got {}D",
            other.ndim()
        );
        assert_eq!(
            self.shape[1], other.shape[0],
            "Matrix dimensions don't match for matmul: ({}, {}) @ ({}, {})",
            self.shape[0], self.shape[1], other.shape[0], other.shape[1]
        );

        let m = self.shape[0] as u32;
        let k = self.shape[1] as u32;
        let n = other.shape[1] as u32;

        let context = self.context.clone();

        // Create output buffer
        let output_size = (m * n) as usize;
        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matmul Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer for dimensions (M, K, N)
        let dims = [m, k, n, 0u32]; // pad to 16 bytes for alignment
        let dims_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matmul Dims"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group using cached layout
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Matmul Bind Group"),
                layout: &context.pipelines.matmul_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: other.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buffer.as_entire_binding(),
                    },
                ],
            });

        // Execute using cached pipeline
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Matmul Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&context.pipelines.matmul_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: each workgroup is 8x8, so we need to cover MxN
            let workgroup_x = n.div_ceil(8);
            let workgroup_y = m.div_ceil(8);
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        context.queue.submit([encoder.finish()]);

        let mut result = Tensor {
            shape: vec![m as usize, n as usize],
            buffer: output_buffer,
            context,
            shared_data: crate::tensor::SharedTensorData::new(),
            node: GraphNode::leaf(),
            requires_grad: self.requires_grad || other.requires_grad,
        };

        // Add backward pass for matmul
        if self.requires_grad || other.requires_grad {
            let a_requires_grad = self.requires_grad;
            let b_requires_grad = other.requires_grad;

            // Store tensors for backward pass
            let a_clone = Tensor::new(&self.to_vec(), self.shape.clone(), self.context.clone());
            let b_clone = Tensor::new(&other.to_vec(), other.shape.clone(), other.context.clone());

            let a_shared = std::sync::Arc::clone(&self.shared_data);
            let b_shared = std::sync::Arc::clone(&other.shared_data);
            let result_shared = std::sync::Arc::clone(&result.shared_data);

            // Clone the nodes for recursive backward calls
            let a_node = std::sync::Arc::clone(&self.node);
            let b_node = std::sync::Arc::clone(&other.node);

            let backward_fn = Box::new(move || {
                let grad_lock = result_shared.grad.lock().unwrap();
                if let Some(ref grad_data) = *grad_lock {
                    let grad_output = Tensor::new(grad_data, vec![m as usize, n as usize], a_clone.context.clone());
                    drop(grad_lock);

                    // grad_A = grad_output @ B^T
                    if a_requires_grad {
                        let b_t = b_clone.transpose();
                        let grad_a = grad_output.matmul(&b_t);
                        let mut a_grad = a_shared.grad.lock().unwrap();
                        let grad_vec = grad_a.to_vec();
                        if let Some(ref mut existing) = *a_grad {
                            for (e, g) in existing.iter_mut().zip(grad_vec.iter()) {
                                *e += g;
                            }
                        } else {
                            *a_grad = Some(grad_vec);
                        }
                        drop(a_grad);

                        // Recursively call backward on input
                        if let Some(ref backward) = a_node.backward_fn {
                            backward();
                        }
                    }

                    // grad_B = A^T @ grad_output
                    if b_requires_grad {
                        let a_t = a_clone.transpose();
                        let grad_b = a_t.matmul(&grad_output);
                        let mut b_grad = b_shared.grad.lock().unwrap();
                        let grad_vec = grad_b.to_vec();
                        if let Some(ref mut existing) = *b_grad {
                            for (e, g) in existing.iter_mut().zip(grad_vec.iter()) {
                                *e += g;
                            }
                        } else {
                            *b_grad = Some(grad_vec);
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
}
