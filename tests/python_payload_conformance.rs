use std::fs;
use std::path::PathBuf;

use physics_in_parallel::prelude::basic::Tensor;
use serde_json::Value;

#[test]
fn python_decoder_fixture_is_exact_canonical_serde_output() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/tests/fixtures/payloads.json");
    let fixture: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let abundance = Tensor::from_vec(&[3], vec![0.2, 0.3, 0.5]);
    let space = Tensor::from_vec(&[1, 2, 2], vec![0.2, 0.8, 0.3, 0.7]);
    assert_eq!(
        serde_json::to_value(abundance).unwrap(),
        fixture["abundance"]
    );
    assert_eq!(serde_json::to_value(Some(space)).unwrap(), fixture["space"]);
    assert_eq!(
        serde_json::to_value(Option::<Tensor<f64>>::None).unwrap(),
        fixture["absent_space"]
    );
    assert_eq!(serde_json::to_value(1.0_f64).unwrap(), fixture["total"]);
}
