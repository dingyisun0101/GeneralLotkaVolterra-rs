use ecological_model_core::interaction::{InteractionMatrix, InteractionMatrixError};
use ndarray::Array2;
use physics_in_parallel::prelude::basic::DenseMatrix;

pub fn interaction_from_array(
    values: Array2<f64>,
) -> Result<InteractionMatrix, InteractionMatrixError> {
    InteractionMatrix::from_matrix(DenseMatrix::from_ndarray(&values))
}
