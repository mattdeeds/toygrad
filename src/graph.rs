use std::sync::Arc;

/// Represents a node in the computational graph.
/// Each operation creates a new node that tracks its inputs and how to compute gradients.
pub struct GraphNode {
    /// Backward function that computes gradients for inputs and accumulates them
    pub backward_fn: Option<Box<dyn Fn()>>,
}

impl GraphNode {
    /// Create a leaf node (input/parameter with no parents)
    pub fn leaf() -> Arc<Self> {
        Arc::new(Self { backward_fn: None })
    }

    /// Create a node for an operation with a backward function
    pub fn new(backward_fn: Box<dyn Fn()>) -> Arc<Self> {
        Arc::new(Self {
            backward_fn: Some(backward_fn),
        })
    }
}
