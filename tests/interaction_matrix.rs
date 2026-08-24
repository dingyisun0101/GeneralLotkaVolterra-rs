use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use general_lotka_volterra_rs::GlvInputs;
use general_lotka_volterra_rs::interaction::{
    ArtifactDisposition, GeneratorProvenance, INTERACTION_GENERATOR_RNG_NAMESPACE,
    INTERACTION_MATRIX_FORMAT, INTERACTION_MATRIX_METADATA_KEY, InteractionArtifactError,
    InteractionArtifactLoadError, InteractionMatrix, InteractionMatrixError, InteractionProvenance,
    InteractionSourceKind, load_verified_interaction_matrix, persist_interaction_matrix,
};
use physics_in_parallel::prelude::basic::{DenseMatrix, RngConfig, RngMethod};
use scientific_workflow::artifact::{ArtifactError, ArtifactLoadError};
use scientific_workflow::execution::ExecutionScope;
use serde_json::{Map, json};
use sha2::{Digest, Sha256};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("glv-{label}-{}-{sequence}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.path).unwrap();
    }
}

fn matrix(rows: usize, columns: usize, values: Vec<f64>) -> DenseMatrix<f64> {
    DenseMatrix::try_from_vec(rows, columns, values).unwrap()
}

fn assert_coefficients(actual: &InteractionMatrix, expected: &[f64]) {
    assert_eq!(actual.values().size(), expected.len());
    for row in 0..actual.species() {
        for column in 0..actual.species() {
            assert_eq!(
                actual.coefficient(row, column),
                expected[row * actual.species() + column]
            );
        }
    }
}

#[test]
fn matrices_validate_domain_and_reuse_shared_storage() {
    let allocation = Arc::new(matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]));
    let resolved = InteractionMatrix::from_shared(Arc::clone(&allocation)).unwrap();
    assert!(Arc::ptr_eq(&allocation, &resolved.shared_values()));
    assert_eq!(resolved.species(), 2);

    let inline = InteractionMatrix::from_rows(vec![vec![0.0, 1.0], vec![-1.0, 0.0]]).unwrap();
    assert_eq!(inline.provenance().kind(), InteractionSourceKind::Inline);
    assert_coefficients(&inline, &[0.0, 1.0, -1.0, 0.0]);

    assert!(matches!(
        InteractionMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0]]),
        Err(InteractionMatrixError::RaggedRows { .. })
    ));
    assert!(matches!(
        InteractionMatrix::from_matrix(matrix(2, 3, vec![0.0; 6])),
        Err(InteractionMatrixError::NonSquare { .. })
    ));
    assert!(matches!(
        InteractionMatrix::from_matrix(matrix(1, 1, vec![f64::INFINITY])),
        Err(InteractionMatrixError::NonFiniteEntry { .. })
    ));
}

#[test]
fn generated_matrices_record_explicit_parameters_version_and_seed() {
    let provenance = GeneratorProvenance::new(
        "test.diagonal",
        "1",
        json!({"diagonal": -0.25}),
        Some(RngConfig::new(Some(42), Some(RngMethod::SmallRng))),
    )
    .unwrap();
    let resolved = InteractionMatrix::from_generated(
        matrix(
            3,
            3,
            vec![-0.25, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, 0.0, -0.25],
        ),
        provenance,
    )
    .unwrap();

    let InteractionProvenance::Generated { generator } = resolved.provenance() else {
        panic!("expected generated provenance");
    };
    assert_eq!(generator.identity(), "test.diagonal");
    assert_eq!(generator.version(), "1");
    assert_eq!(generator.recipe()["diagonal"], -0.25);
    assert_eq!(generator.rng().unwrap().seed(), Some(42));
    let rng = resolved.generator_rng_record().unwrap().unwrap();
    assert_eq!(rng.namespace(), INTERACTION_GENERATOR_RNG_NAMESPACE);
    assert_eq!(rng.method(), "test.diagonal+small_rng");
    assert_eq!(rng.key(), "42");
}

#[test]
fn workflow_decodes_inline_values_and_resolves_pip_matrix_paths() {
    let directory = TestDirectory::new("workflow-source-boundary");
    let configuration = directory.path().join("config");
    let data = directory.path().join("data");
    fs::create_dir(&configuration).unwrap();
    fs::create_dir(&data).unwrap();
    fs::write(
        configuration.join("fixed.json"),
        br#"{"interaction_matrix":[[1.0,0.0],[0.0,1.0]]}"#,
    )
    .unwrap();
    fs::write(
        configuration.join("sweep.json"),
        br#"{"mode":"cartesian","axes":{}}"#,
    )
    .unwrap();
    fs::write(
        configuration.join("paths.json"),
        br#"{"interaction_matrix_file":"data/interaction.json"}"#,
    )
    .unwrap();
    fs::write(
        configuration.join("state.json"),
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas/state.json")).unwrap(),
    )
    .unwrap();
    fs::write(
        data.join("interaction.json"),
        br#"{"kind":"matrix","version":1,"scalar":"f64","shape":[2,2],"data":[0.0,1.0,-1.0,0.0]}"#,
    )
    .unwrap();

    let inputs = GlvInputs::load(directory.path()).unwrap();
    let task = inputs.combination(0).unwrap();
    let inline = InteractionMatrix::from_rows(
        task.decode_value::<Vec<Vec<f64>>>("/interaction_matrix")
            .unwrap(),
    )
    .unwrap();
    let path = task.resolve_path("interaction_matrix_file").unwrap();
    let file = InteractionMatrix::load_json(&path).unwrap();
    assert_coefficients(&inline, &[1.0, 0.0, 0.0, 1.0]);
    assert_coefficients(&file, &[0.0, 1.0, -1.0, 0.0]);
    assert!(matches!(
        file.provenance(),
        InteractionProvenance::JsonFile { path: source } if source == &path
    ));
}

#[test]
fn artifacts_use_pip_json_and_workflow_content_addressing() {
    let directory = TestDirectory::new("artifact-reuse");
    let scope = ExecutionScope::create_named(directory.path(), "execution").unwrap();
    let first_matrix =
        InteractionMatrix::from_matrix(matrix(2, 2, vec![2.0, -1.0, 0.5, 3.0])).unwrap();
    let second_matrix =
        InteractionMatrix::from_matrix(matrix(2, 2, vec![2.0, -1.0, 0.5, 3.0])).unwrap();

    let first = persist_interaction_matrix(&scope, &first_matrix).unwrap();
    let second = persist_interaction_matrix(&scope, &second_matrix).unwrap();
    assert_eq!(first.disposition(), ArtifactDisposition::Created);
    assert_eq!(second.disposition(), ArtifactDisposition::Reused);
    assert_eq!(first.descriptor(), second.descriptor());
    assert_eq!(first.descriptor().format(), INTERACTION_MATRIX_FORMAT);
    assert_eq!(first.descriptor().shape(), [2, 2]);

    let path = scope.directory().join(first.descriptor().path());
    let bytes = fs::read(&path).unwrap();
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(first.descriptor().sha256(), digest);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
        json!({
            "kind": "matrix",
            "version": 1,
            "scalar": "f64",
            "shape": [2, 2],
            "data": [2.0, -1.0, 0.5, 3.0]
        })
    );

    let verified = load_verified_interaction_matrix(scope.directory(), first.descriptor()).unwrap();
    assert_coefficients(&verified, &[2.0, -1.0, 0.5, 3.0]);
    let mut metadata = Map::new();
    assert!(
        first
            .descriptor()
            .insert_into_metadata(&mut metadata)
            .is_none()
    );
    assert!(metadata.contains_key(INTERACTION_MATRIX_METADATA_KEY));
    assert!(!serde_json::to_string(&metadata).unwrap().contains("data"));
}

#[test]
fn malformed_json_and_artifact_collisions_fail_closed() {
    let directory = TestDirectory::new("artifact-failures");
    let malformed = directory.path().join("malformed.json");
    fs::write(&malformed, b"{").unwrap();
    assert!(matches!(
        InteractionMatrix::load_json(&malformed),
        Err(InteractionMatrixError::Json { .. })
    ));

    let wrong_count = directory.path().join("wrong-count.json");
    fs::write(
        &wrong_count,
        br#"{"kind":"matrix","version":1,"scalar":"f64","shape":[2,2],"data":[1.0]}"#,
    )
    .unwrap();
    assert!(matches!(
        InteractionMatrix::load_json(&wrong_count),
        Err(InteractionMatrixError::Json { .. })
    ));

    let scope = ExecutionScope::create_named(directory.path(), "execution").unwrap();
    let matrix = InteractionMatrix::from_matrix(matrix(1, 1, vec![1.0])).unwrap();
    let persisted = persist_interaction_matrix(&scope, &matrix).unwrap();
    let path = scope.directory().join(persisted.descriptor().path());
    fs::write(&path, b"different bytes").unwrap();
    assert!(matches!(
        load_verified_interaction_matrix(scope.directory(), persisted.descriptor()),
        Err(InteractionArtifactLoadError::Workflow(
            ArtifactLoadError::DigestMismatch { .. }
        ))
    ));
    assert!(matches!(
        persist_interaction_matrix(&scope, &matrix),
        Err(InteractionArtifactError::Workflow(
            ArtifactError::DigestCollision { .. }
        ))
    ));
}
