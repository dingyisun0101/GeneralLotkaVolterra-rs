#![allow(dead_code)]

use ecological_state_toolkit::interaction::{InteractionMatrix, InteractionMatrixError};
use physics_in_parallel::prelude::advanced::RawStorage;
use physics_in_parallel::prelude::basic::{Backend, Matrix, ResolvedRng, RngMethod, Tensor};

pub fn interaction_from_array(
    values: Matrix<f64>,
) -> Result<InteractionMatrix, InteractionMatrixError> {
    InteractionMatrix::from_matrix(values)
}

pub fn dense_matrix(rows: usize, columns: usize, values: Vec<f64>) -> Matrix<f64> {
    Matrix::from_values(rows, columns, Backend::Dense, values).unwrap()
}

pub fn zero_matrix(rows: usize, columns: usize) -> Matrix<f64> {
    Matrix::zeros(rows, columns, Backend::Dense).unwrap()
}

pub fn dense_tensor(shape: &[usize], values: Vec<f64>) -> Tensor<f64> {
    Tensor::from_values(shape, Backend::Dense, values).unwrap()
}

pub fn zero_tensor(shape: &[usize]) -> Tensor<f64> {
    Tensor::zeros(shape, Backend::Dense).unwrap()
}

pub fn indexed_rng(seed: u64) -> ResolvedRng {
    ResolvedRng::new(seed, RngMethod::IndexedSplitMix64)
}

pub fn stateful_rng(seed: u64) -> ResolvedRng {
    ResolvedRng::new(seed, RngMethod::ChaCha12)
}

pub trait DenseTensorExt {
    fn as_slice(&self) -> &[f64];
    fn as_mut_slice(&mut self) -> &mut [f64];
    fn sum_serial(&self) -> f64;
    fn copy_from(
        &mut self,
        source: &Self,
    ) -> Result<(), physics_in_parallel::prelude::basic::TensorError>;
}

impl DenseTensorExt for Tensor<f64> {
    fn as_slice(&self) -> &[f64] {
        self.dense_values().unwrap()
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.dense_values_mut().unwrap()
    }

    fn sum_serial(&self) -> f64 {
        self.as_slice().iter().sum()
    }

    fn copy_from(
        &mut self,
        source: &Self,
    ) -> Result<(), physics_in_parallel::prelude::basic::TensorError> {
        if self.shape() != source.shape() {
            return Err(
                physics_in_parallel::prelude::basic::TensorError::ShapeMismatch {
                    lhs: self.shape().to_vec(),
                    rhs: source.shape().to_vec(),
                },
            );
        }
        self.as_mut_slice().copy_from_slice(source.as_slice());
        Ok(())
    }
}
