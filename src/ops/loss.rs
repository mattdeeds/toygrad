use crate::tensor::Tensor;

impl Tensor {
    /// Mean Squared Error loss
    /// Computes mean((predictions - targets)^2)
    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, target.shape,
            "Predictions and targets must have same shape for MSE loss"
        );

        // (pred - target)^2
        let diff = self.sub(target);
        let squared = diff.mul(&diff);

        // mean
        squared.mean()
    }

    /// Binary Cross Entropy loss
    /// Computes -mean(target * log(pred) + (1 - target) * log(1 - pred))
    /// Note: This is a simplified version that assumes predictions are already
    /// passed through sigmoid
    pub fn bce_loss(&self, target: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, target.shape,
            "Predictions and targets must have same shape for BCE loss"
        );

        // For numerical stability, we compute this on CPU for now
        let pred_data = self.to_vec();
        let target_data = target.to_vec();

        let epsilon = 1e-7f32;
        let mut loss_sum = 0.0f32;

        for (pred, targ) in pred_data.iter().zip(target_data.iter()) {
            // Clamp predictions to avoid log(0)
            let p = pred.max(epsilon).min(1.0 - epsilon);
            loss_sum += -( targ * p.ln() + (1.0 - targ) * (1.0 - p).ln());
        }

        let mean_loss = loss_sum / pred_data.len() as f32;

        Tensor::new(&[mean_loss], vec![1], self.context.clone())
    }
}
