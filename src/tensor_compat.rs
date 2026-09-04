use physics_in_parallel::prelude::advanced::RawStorage;
use physics_in_parallel::prelude::basic::{Tensor, TensorError};

/// Dense-storage operations used by GLV's allocation-sensitive kernels.
pub(crate) trait DenseTensorExt {
    fn as_slice(&self) -> &[f64];
    fn as_mut_slice(&mut self) -> &mut [f64];
    fn sum_serial(&self) -> f64;
    fn copy_from(&mut self, source: &Self) -> Result<(), TensorError>;
    fn zip_from<F>(&mut self, left: &Self, right: &Self, function: F) -> Result<(), TensorError>
    where
        F: Fn(f64, f64) -> f64;
    fn for_each_chunk_mut<F>(&mut self, chunk_size: usize, function: F)
    where
        F: FnMut(usize, &mut [f64]);
}

impl DenseTensorExt for Tensor<f64> {
    fn as_slice(&self) -> &[f64] {
        self.dense_values()
            .expect("GLV tensors always use the dense backend")
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.dense_values_mut()
            .expect("GLV tensors always use the dense backend")
    }

    fn sum_serial(&self) -> f64 {
        self.as_slice().iter().copied().sum()
    }

    fn copy_from(&mut self, source: &Self) -> Result<(), TensorError> {
        if self.shape() != source.shape() {
            return Err(TensorError::ShapeMismatch {
                lhs: self.shape().to_vec(),
                rhs: source.shape().to_vec(),
            });
        }
        self.as_mut_slice().copy_from_slice(source.as_slice());
        Ok(())
    }

    fn zip_from<F>(&mut self, left: &Self, right: &Self, function: F) -> Result<(), TensorError>
    where
        F: Fn(f64, f64) -> f64,
    {
        if self.shape() != left.shape() || self.shape() != right.shape() {
            return Err(TensorError::ShapeMismatch {
                lhs: self.shape().to_vec(),
                rhs: if self.shape() != left.shape() {
                    left.shape().to_vec()
                } else {
                    right.shape().to_vec()
                },
            });
        }
        for ((output, left), right) in self
            .as_mut_slice()
            .iter_mut()
            .zip(left.as_slice())
            .zip(right.as_slice())
        {
            *output = function(*left, *right);
        }
        Ok(())
    }

    fn for_each_chunk_mut<F>(&mut self, chunk_size: usize, mut function: F)
    where
        F: FnMut(usize, &mut [f64]),
    {
        for (index, chunk) in self.as_mut_slice().chunks_exact_mut(chunk_size).enumerate() {
            function(index, chunk);
        }
    }
}
