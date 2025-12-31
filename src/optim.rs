use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    /// Learning rate
    pub lr: f32,
}

impl SGD {
    /// Create a new SGD optimizer with the given learning rate
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Perform one optimization step on a parameter
    /// Updates the parameter in-place: param = param - lr * grad
    /// This version uses GPU compute shader for maximum performance
    pub fn step(&self, param: &mut Tensor) {
        if let Some(grad) = param.get_grad() {
            let size = param.size();

            // Get gradient data and upload to GPU buffer
            let grad_data = grad.to_vec();
            let grad_buffer =
                param
                    .context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Gradient Buffer"),
                        contents: bytemuck::cast_slice(&grad_data),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });

            // Create uniform buffer for optimizer params
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct OptimizerParams {
                size: u32,
                lr: f32,
                momentum: f32,
                beta1: f32,
                beta2: f32,
                epsilon: f32,
                t: u32,
                _padding: u32,
            }

            let params = OptimizerParams {
                size: size as u32,
                lr: self.lr,
                momentum: 0.0,
                beta1: 0.0,
                beta2: 0.0,
                epsilon: 0.0,
                t: 0,
                _padding: 0,
            };

            let param_buffer =
                param
                    .context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Optimizer Params"),
                        contents: bytemuck::cast_slice(&[params]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            // Create bind group
            let bind_group = param
                .context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SGD Bind Group"),
                    layout: &param.context.pipelines.optimizer_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: param.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: grad_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: param_buffer.as_entire_binding(),
                        },
                    ],
                });

            // Execute GPU kernel
            let mut encoder =
                param
                    .context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("SGD Encoder"),
                    });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SGD Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&param.context.pipelines.sgd_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch workgroups: (size + 255) / 256 workgroups of 256 threads each
                let workgroups = (size as u32 + 255) / 256;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }

            param.context.queue.submit([encoder.finish()]);

            // Note: param.buffer is now updated in-place on GPU, no need to read back!
        }
    }

    /// Zero gradients for all parameters
    pub fn zero_grad(&self, params: &[&Tensor]) {
        for param in params {
            param.zero_grad();
        }
    }
}

/// State for a single parameter in Adam optimizer
#[derive(Clone)]
struct AdamState {
    /// First moment estimate (momentum)
    m: Vec<f32>,
    /// Second moment estimate (adaptive learning rate)
    v: Vec<f32>,
    /// Time step (for bias correction)
    t: usize,
}

impl AdamState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
        }
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
///
/// Adam combines ideas from RMSprop and momentum to provide adaptive learning rates
/// for each parameter. It maintains running averages of both the gradients and the
/// squared gradients.
///
/// Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
pub struct Adam {
    /// Learning rate (default: 0.001)
    pub lr: f32,
    /// Exponential decay rate for first moment estimates (default: 0.9)
    pub beta1: f32,
    /// Exponential decay rate for second moment estimates (default: 0.999)
    pub beta2: f32,
    /// Small constant for numerical stability (default: 1e-8)
    pub epsilon: f32,
    /// State for each parameter (indexed by tensor address)
    state: Arc<Mutex<HashMap<usize, AdamState>>>,
}

impl Adam {
    /// Create a new Adam optimizer with default hyperparameters
    /// - lr: 0.001
    /// - beta1: 0.9
    /// - beta2: 0.999
    /// - epsilon: 1e-8
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new Adam optimizer with custom hyperparameters
    pub fn with_params(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Perform one optimization step on a parameter
    ///
    /// Updates using the Adam algorithm:
    /// 1. Update biased first moment: m = beta1 * m + (1 - beta1) * grad
    /// 2. Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
    /// 3. Compute bias-corrected moments: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
    /// 4. Update parameter: param -= lr * m_hat / (sqrt(v_hat) + epsilon)
    pub fn step(&self, param: &mut Tensor) {
        if let Some(grad) = param.get_grad() {
            let param_data = param.to_vec();
            let grad_data = grad.to_vec();
            let size = param_data.len();

            // Use tensor's shared_data address as unique identifier
            let param_id = Arc::as_ptr(&param.shared_data) as usize;

            let mut state_map = self.state.lock().unwrap();
            let state = state_map
                .entry(param_id)
                .or_insert_with(|| AdamState::new(size));

            // Increment time step
            state.t += 1;
            let t = state.t as f32;

            // Update biased first moment estimate
            for i in 0..size {
                state.m[i] = self.beta1 * state.m[i] + (1.0 - self.beta1) * grad_data[i];
            }

            // Update biased second moment estimate
            for i in 0..size {
                state.v[i] =
                    self.beta2 * state.v[i] + (1.0 - self.beta2) * grad_data[i] * grad_data[i];
            }

            // Compute bias-corrected first moment estimate
            let m_hat_correction = 1.0 / (1.0 - self.beta1.powf(t));

            // Compute bias-corrected second moment estimate
            let v_hat_correction = 1.0 / (1.0 - self.beta2.powf(t));

            // Update parameters
            let updated_data: Vec<f32> = param_data
                .iter()
                .zip(state.m.iter())
                .zip(state.v.iter())
                .map(|((p, m), v)| {
                    let m_hat = m * m_hat_correction;
                    let v_hat = v * v_hat_correction;
                    p - self.lr * m_hat / (v_hat.sqrt() + self.epsilon)
                })
                .collect();

            param.update_data(&updated_data);
        }
    }

    /// Zero gradients for all parameters
    pub fn zero_grad(&self, params: &[&Tensor]) {
        for param in params {
            param.zero_grad();
        }
    }
}
