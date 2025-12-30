pub mod broadcast;
pub mod gpu;
pub mod graph;
pub mod ops;
pub mod optim;
pub mod tensor;

// Re-export main types for convenience
pub use gpu::GpuContext;
pub use optim::{Adam, SGD};
pub use tensor::Tensor;
