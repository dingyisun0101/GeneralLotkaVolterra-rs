use ecological_model_core::interaction::{InteractionMatrix, InteractionMatrixError};
use physics_in_parallel::prelude::basic::DenseMatrix;

pub fn interaction_from_array(
    values: DenseMatrix<f64>,
) -> Result<InteractionMatrix, InteractionMatrixError> {
    InteractionMatrix::from_matrix(values)
}
