use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    pub fn step(&self, param: &mut Tensor) {
        if let Some(grad) = param.get_grad() {
            // Get current parameter data
            let param_data = param.to_vec();
            let grad_data = grad.to_vec();

            // Update: param -= lr * grad
            let updated_data: Vec<f32> = param_data.iter()
                .zip(grad_data.iter())
                .map(|(p, g)| p - self.lr * g)
                .collect();

            // Update data in-place (preserves gradient tracking)
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
            let state = state_map.entry(param_id).or_insert_with(|| AdamState::new(size));

            // Increment time step
            state.t += 1;
            let t = state.t as f32;

            // Update biased first moment estimate
            for i in 0..size {
                state.m[i] = self.beta1 * state.m[i] + (1.0 - self.beta1) * grad_data[i];
            }

            // Update biased second moment estimate
            for i in 0..size {
                state.v[i] = self.beta2 * state.v[i] + (1.0 - self.beta2) * grad_data[i] * grad_data[i];
            }

            // Compute bias-corrected first moment estimate
            let m_hat_correction = 1.0 / (1.0 - self.beta1.powf(t));

            // Compute bias-corrected second moment estimate
            let v_hat_correction = 1.0 / (1.0 - self.beta2.powf(t));

            // Update parameters
            let updated_data: Vec<f32> = param_data.iter()
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
