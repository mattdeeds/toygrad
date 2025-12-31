/// Broadcasting utilities for tensor operations
///
/// Implements NumPy-style broadcasting rules:
/// 1. If tensors have different number of dimensions, prepend 1s to the smaller shape
/// 2. For each dimension, sizes must either match or one must be 1
/// 3. Output shape is the maximum of input shapes in each dimension

/// Information about how to broadcast two tensors
#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    /// Output shape after broadcasting
    pub output_shape: Vec<usize>,
    /// Shape of first tensor (with prepended 1s if needed)
    pub shape_a: Vec<usize>,
    /// Shape of second tensor (with prepended 1s if needed)
    pub shape_b: Vec<usize>,
    /// Which dimensions of A need broadcasting (are size 1 but output is larger)
    pub a_broadcast_dims: Vec<bool>,
    /// Which dimensions of B need broadcasting (are size 1 but output is larger)
    pub b_broadcast_dims: Vec<bool>,
}

impl BroadcastInfo {
    /// Compute broadcasting information for two shapes
    /// Returns None if shapes are not broadcast-compatible
    pub fn compute(shape_a: &[usize], shape_b: &[usize]) -> Option<Self> {
        let ndim_a = shape_a.len();
        let ndim_b = shape_b.len();
        let max_ndim = ndim_a.max(ndim_b);

        // Prepend 1s to make shapes the same length
        let mut expanded_a = vec![1; max_ndim - ndim_a];
        expanded_a.extend_from_slice(shape_a);

        let mut expanded_b = vec![1; max_ndim - ndim_b];
        expanded_b.extend_from_slice(shape_b);

        // Check compatibility and compute output shape
        let mut output_shape = Vec::with_capacity(max_ndim);
        let mut a_broadcast_dims = Vec::with_capacity(max_ndim);
        let mut b_broadcast_dims = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let size_a = expanded_a[i];
            let size_b = expanded_b[i];

            if size_a == size_b {
                output_shape.push(size_a);
                a_broadcast_dims.push(false);
                b_broadcast_dims.push(false);
            } else if size_a == 1 {
                output_shape.push(size_b);
                a_broadcast_dims.push(true);
                b_broadcast_dims.push(false);
            } else if size_b == 1 {
                output_shape.push(size_a);
                a_broadcast_dims.push(false);
                b_broadcast_dims.push(true);
            } else {
                // Incompatible shapes
                return None;
            }
        }

        Some(BroadcastInfo {
            output_shape,
            shape_a: expanded_a,
            shape_b: expanded_b,
            a_broadcast_dims,
            b_broadcast_dims,
        })
    }

    /// Check if any broadcasting is actually needed
    pub fn needs_broadcast(&self) -> bool {
        self.a_broadcast_dims.iter().any(|&b| b) || self.b_broadcast_dims.iter().any(|&b| b)
    }

    /// Get dimensions that need reduction for backward pass
    /// Returns (dims to reduce for A, dims to reduce for B)
    pub fn get_reduction_dims(
        &self,
        original_shape_a: &[usize],
        original_shape_b: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let ndim_a = original_shape_a.len();
        let ndim_b = original_shape_b.len();
        let max_ndim = self.output_shape.len();

        let mut reduce_dims_a = Vec::new();
        let mut reduce_dims_b = Vec::new();

        // For dimensions that were broadcast (size 1 became size N), we need to sum gradients
        for (i, (&broadcast_a, &broadcast_b)) in self
            .a_broadcast_dims
            .iter()
            .zip(&self.b_broadcast_dims)
            .enumerate()
        {
            if broadcast_a {
                reduce_dims_a.push(i);
            }
            if broadcast_b {
                reduce_dims_b.push(i);
            }
        }

        // Also need to reduce prepended dimensions (that didn't exist in original shape)
        let prepend_a = max_ndim - ndim_a;
        let prepend_b = max_ndim - ndim_b;

        for i in 0..prepend_a {
            if !reduce_dims_a.contains(&i) {
                reduce_dims_a.push(i);
            }
        }

        for i in 0..prepend_b {
            if !reduce_dims_b.contains(&i) {
                reduce_dims_b.push(i);
            }
        }

        reduce_dims_a.sort_unstable();
        reduce_dims_b.sort_unstable();

        (reduce_dims_a, reduce_dims_b)
    }
}

/// Compute strides for a given shape (row-major order)
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1; ndim];

    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_shape() {
        let info = BroadcastInfo::compute(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(info.output_shape, vec![3, 4]);
        assert!(!info.needs_broadcast());
    }

    #[test]
    fn test_scalar_broadcast() {
        let info = BroadcastInfo::compute(&[1], &[3, 4]).unwrap();
        assert_eq!(info.output_shape, vec![3, 4]);
        assert!(info.needs_broadcast());
        assert_eq!(info.shape_a, vec![1, 1]);
        assert_eq!(info.shape_b, vec![3, 4]);
    }

    #[test]
    fn test_row_broadcast() {
        let info = BroadcastInfo::compute(&[1, 4], &[3, 4]).unwrap();
        assert_eq!(info.output_shape, vec![3, 4]);
        assert!(info.needs_broadcast());
        assert!(info.a_broadcast_dims[0]);
        assert!(!info.a_broadcast_dims[1]);
    }

    #[test]
    fn test_column_broadcast() {
        let info = BroadcastInfo::compute(&[3, 1], &[3, 4]).unwrap();
        assert_eq!(info.output_shape, vec![3, 4]);
        assert!(info.needs_broadcast());
        assert!(!info.a_broadcast_dims[0]);
        assert!(info.a_broadcast_dims[1]);
    }

    #[test]
    fn test_incompatible() {
        let info = BroadcastInfo::compute(&[3, 5], &[3, 4]);
        assert!(info.is_none());
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[3, 4]), vec![4, 1]);
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }
}
