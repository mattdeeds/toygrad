use std::num::NonZeroU64;
use std::sync::Arc;
use wgpu;

/// Cached compute pipelines for GPU operations
pub struct PipelineCache {
    // Elementwise operations (add, sub, mul, div)
    pub elementwise_layout: wgpu::BindGroupLayout,
    pub elementwise_pipeline: wgpu::ComputePipeline,

    // Broadcast elementwise operations
    pub broadcast_layout: wgpu::BindGroupLayout,
    pub broadcast_pipeline: wgpu::ComputePipeline,

    // Negate operation
    pub negate_layout: wgpu::BindGroupLayout,
    pub negate_pipeline: wgpu::ComputePipeline,

    // Matrix multiplication
    pub matmul_layout: wgpu::BindGroupLayout,
    pub matmul_pipeline: wgpu::ComputePipeline,

    // Activation functions
    pub activation_layout: wgpu::BindGroupLayout,
    pub relu_pipeline: wgpu::ComputePipeline,
    pub sigmoid_pipeline: wgpu::ComputePipeline,
    pub tanh_pipeline: wgpu::ComputePipeline,

    // Reduction (sum)
    pub reduce_layout: wgpu::BindGroupLayout,
    pub sum_reduce_pipeline: wgpu::ComputePipeline,

    // Transpose
    pub transpose_layout: wgpu::BindGroupLayout,
    pub transpose_pipeline: wgpu::ComputePipeline,
}

impl PipelineCache {
    fn new(device: &wgpu::Device) -> Self {
        // === Elementwise pipelines ===
        let elementwise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elementwise Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ops/elementwise.wgsl").into()),
        });

        let elementwise_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Elementwise Bind Group Layout"),
            entries: &[
                buffer_entry(0, true),   // input a
                buffer_entry(1, true),   // input b
                buffer_entry(2, false),  // output
                uniform_entry(3),        // op_type
            ],
        });

        let elementwise_pipeline = create_pipeline(device, &elementwise_shader, &elementwise_layout, "elementwise");

        // === Broadcast pipeline ===
        let broadcast_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Broadcast Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ops/broadcast.wgsl").into()),
        });

        let broadcast_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Broadcast Bind Group Layout"),
            entries: &[
                buffer_entry(0, true),     // input a
                buffer_entry(1, true),     // input b
                buffer_entry(2, false),    // output
                uniform_entry_size(3, 96), // broadcast params (op_type, ndim, padding, shapes, strides) = 96 bytes
            ],
        });

        let broadcast_pipeline = create_pipeline(device, &broadcast_shader, &broadcast_layout, "broadcast_op");

        // === Negate pipeline ===
        let negate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Negate Bind Group Layout"),
            entries: &[
                buffer_entry(0, true),   // input
                buffer_entry(1, false),  // output
            ],
        });

        let negate_pipeline = create_pipeline(device, &elementwise_shader, &negate_layout, "negate");

        // === Matmul pipeline ===
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ops/matmul.wgsl").into()),
        });

        let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matmul Bind Group Layout"),
            entries: &[
                buffer_entry(0, true),   // input a
                buffer_entry(1, true),   // input b
                buffer_entry(2, false),  // output
                uniform_entry_size(3, 16), // dims (M, K, N, padding)
            ],
        });

        let matmul_pipeline = create_pipeline(device, &matmul_shader, &matmul_layout, "matmul");

        // === Activation pipelines ===
        let activation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Activation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ops/activations.wgsl").into()),
        });

        let activation_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Activation Bind Group Layout"),
            entries: &[
                buffer_entry(0, true),   // input
                buffer_entry(1, false),  // output
            ],
        });

        let relu_pipeline = create_pipeline(device, &activation_shader, &activation_layout, "relu");
        let sigmoid_pipeline = create_pipeline(device, &activation_shader, &activation_layout, "sigmoid");
        let tanh_pipeline = create_pipeline(device, &activation_shader, &activation_layout, "tanh_activation");

        // === Reduce pipeline ===
        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduce Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ops/reduce.wgsl").into()),
        });

        let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reduce Bind Group Layout"),
            entries: &[
                buffer_entry(0, true),   // input
                buffer_entry(1, false),  // output
            ],
        });

        let sum_reduce_pipeline = create_pipeline(device, &reduce_shader, &reduce_layout, "sum_reduce");

        // === Transpose pipeline ===
        let transpose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Transpose Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ops/shape.wgsl").into()),
        });

        let transpose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Transpose Bind Group Layout"),
            entries: &[
                buffer_entry(0, true),   // input
                buffer_entry(1, false),  // output
                uniform_entry_size(2, 8), // dims (rows, cols)
            ],
        });

        let transpose_pipeline = create_pipeline(device, &transpose_shader, &transpose_layout, "transpose");

        Self {
            elementwise_layout,
            elementwise_pipeline,
            broadcast_layout,
            broadcast_pipeline,
            negate_layout,
            negate_pipeline,
            matmul_layout,
            matmul_pipeline,
            activation_layout,
            relu_pipeline,
            sigmoid_pipeline,
            tanh_pipeline,
            reduce_layout,
            sum_reduce_pipeline,
            transpose_layout,
            transpose_pipeline,
        }
    }
}

// Helper functions for creating bind group layout entries
fn buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if read_only {
                wgpu::BufferBindingType::Storage { read_only: true }
            } else {
                wgpu::BufferBindingType::Storage { read_only: false }
            },
            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            has_dynamic_offset: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    uniform_entry_size(binding, 4)
}

fn uniform_entry_size(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            min_binding_size: Some(NonZeroU64::new(size).unwrap()),
            has_dynamic_offset: false,
        },
        count: None,
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{} Pipeline Layout", entry_point)),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{} Pipeline", entry_point)),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Shared GPU context containing device and queue.
/// This is wrapped in Arc so it can be shared across tensors.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipelines: Arc<PipelineCache>,
}

impl GpuContext {
    /// Initialize a new GPU context with default settings
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to create adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Toygrad Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .expect("Failed to create device");

        let pipelines = PipelineCache::new(&device);

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines: Arc::new(pipelines),
        }
    }
}
