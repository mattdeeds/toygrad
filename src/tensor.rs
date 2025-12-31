use std::rc::Rc;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

use crate::gpu::GpuContext;
use crate::graph::GraphNode;

/// Shared data for a tensor that can be safely referenced in backward closures
pub struct SharedTensorData {
    /// Gradient tensor (computed during backward pass)
    pub grad: Mutex<Option<Vec<f32>>>,
}

impl SharedTensorData {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            grad: Mutex::new(None),
        })
    }
}

/// Core tensor type with GPU-backed storage
pub struct Tensor {
    /// Shape of the tensor (e.g., [2, 3, 4] for a 2x3x4 tensor)
    pub shape: Vec<usize>,
    /// GPU buffer containing the data
    pub buffer: wgpu::Buffer,
    /// GPU context (device and queue)
    pub context: GpuContext,
    /// Shared gradient storage
    pub(crate) shared_data: Arc<SharedTensorData>,
    /// Computational graph node for autodiff
    pub node: Rc<GraphNode>,
    /// Whether this tensor requires gradient computation
    pub requires_grad: bool,
}

impl Tensor {
    /// Create a new tensor from data on the CPU
    pub fn new(data: &[f32], shape: Vec<usize>, context: GpuContext) -> Self {
        // Verify shape matches data length
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        // Create GPU buffer with the data
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tensor Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            shape,
            buffer,
            context,
            shared_data: SharedTensorData::new(),
            node: GraphNode::leaf(),
            requires_grad: false,
        }
    }

    /// Create a tensor that requires gradient tracking
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Get the total number of elements in the tensor
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Read data from GPU to CPU
    pub fn to_vec(&self) -> Vec<f32> {
        // Create a staging buffer for reading
        let staging_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.size() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy from tensor buffer to staging buffer
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Encoder"),
                });

        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &staging_buffer,
            0,
            (self.size() * std::mem::size_of::<f32>()) as u64,
        );

        self.context.queue.submit([encoder.finish()]);

        // Map and read the buffer
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>, context: GpuContext) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![0.0f32; size];
        Self::new(&data, shape, context)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>, context: GpuContext) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![1.0f32; size];
        Self::new(&data, shape, context)
    }

    /// Create a tensor filled with a constant value
    pub fn full(shape: Vec<usize>, value: f32, context: GpuContext) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![value; size];
        Self::new(&data, shape, context)
    }

    /// Accumulate gradient (add to existing gradient or create new one)
    pub fn accumulate_grad(&self, grad_data: &[f32]) {
        let mut grad_lock = self.shared_data.grad.lock().unwrap();

        if let Some(existing_grad) = grad_lock.as_mut() {
            // Add to existing gradient
            let new_data: Vec<f32> = existing_grad
                .iter()
                .zip(grad_data.iter())
                .map(|(a, b)| a + b)
                .collect();
            *existing_grad = new_data;
        } else {
            // Create new gradient
            *grad_lock = Some(grad_data.to_vec());
        }
    }

    /// Get gradient as a cloned tensor
    pub fn get_grad(&self) -> Option<Tensor> {
        let grad_lock = self.shared_data.grad.lock().unwrap();
        grad_lock
            .as_ref()
            .map(|data| Tensor::new(data, self.shape.clone(), self.context.clone()))
    }

    /// Zero out the gradient
    pub fn zero_grad(&self) {
        let mut grad_lock = self.shared_data.grad.lock().unwrap();
        *grad_lock = None;
    }

    /// Run backward pass from this tensor (typically a loss scalar)
    /// This tensor should be a scalar (shape [1])
    pub fn backward(&self) {
        assert_eq!(
            self.shape,
            vec![1],
            "backward() should be called on a scalar"
        );

        // Initialize gradient to 1.0 for the output
        self.accumulate_grad(&[1.0]);

        // Call the backward function if it exists
        if let Some(ref backward_fn) = self.node.backward_fn {
            backward_fn();
        }
    }

    /// Update tensor data in-place (preserves shared_data and gradient tracking)
    pub fn update_data(&mut self, new_data: &[f32]) {
        assert_eq!(
            new_data.len(),
            self.size(),
            "Data length must match tensor size"
        );

        // Create new buffer with updated data
        self.buffer = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Updated Tensor Buffer"),
                contents: bytemuck::cast_slice(new_data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_vec();
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data", &data)
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}
