use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use general_lotka_volterra_rs::kernel::{
    ArtifactDisposition, GeneratedSource, GeneratorRandomness, INTERACTION_MATRIX_FORMAT,
    INTERACTION_MATRIX_METADATA_KEY, InMemorySource, InteractionArtifactError,
    InteractionGenerator, InteractionProvenance, InteractionSource, InteractionSourceError,
    InteractionSourceKind, JsonInteractionSource, persist_interaction_matrix,
};
use ndarray::{Array2, arr2};
use scientific_workflow::execution::ExecutionScope;
use scientific_workflow::project::ScientificProject;
use serde::Serialize;
use serde_json::Map;
use sha2::{Digest, Sha256};
use thiserror::Error;

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

#[derive(Clone, Debug, Serialize)]
struct GeneratorParameters {
    diagonal: f64,
}

#[derive(Debug, Error)]
#[error("test generator failed")]
struct GeneratorError;

#[derive(Debug)]
struct DiagonalGenerator {
    parameters: GeneratorParameters,
    randomness: GeneratorRandomness,
}

impl InteractionGenerator for DiagonalGenerator {
    type Error = GeneratorError;
    type Parameters = GeneratorParameters;

    const IDENTITY: &'static str = "test.diagonal";
    const VERSION: &'static str = "1";

    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }

    fn randomness(&self) -> GeneratorRandomness {
        self.randomness
    }

    fn generate(self, species: usize) -> Result<Array2<f64>, Self::Error> {
        Ok(Array2::from_diag_elem(species, self.parameters.diagonal))
    }
}

#[test]
fn in_memory_and_inline_sources_validate_the_complete_domain() {
    let allocation = Arc::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    let resolved = InMemorySource::from_shared(Arc::clone(&allocation))
        .with_label("direct-test")
        .resolve(2)
        .unwrap();
    assert!(Arc::ptr_eq(&allocation, &resolved.shared_values()));
    assert_eq!(resolved.species(), 2);
    assert!(matches!(
        resolved.provenance(),
        InteractionProvenance::InMemory { label: Some(label) } if label == "direct-test"
    ));

    let inline = JsonInteractionSource::inline(vec![vec![0.0, 1.0], vec![-1.0, 0.0]])
        .resolve(2)
        .unwrap();
    assert_eq!(
        inline.provenance().kind(),
        InteractionSourceKind::InlineJson
    );
    assert_eq!(inline.values(), &arr2(&[[0.0, 1.0], [-1.0, 0.0]]));

    assert!(matches!(
        JsonInteractionSource::inline(vec![vec![1.0, 2.0], vec![3.0]]).resolve(2),
        Err(InteractionSourceError::RaggedRows { .. })
    ));
    assert!(matches!(
        InMemorySource::new(Array2::zeros((2, 3))).resolve(2),
        Err(InteractionSourceError::NonSquare { .. })
    ));
    assert!(matches!(
        InMemorySource::new(Array2::eye(2)).resolve(3),
        Err(InteractionSourceError::SpeciesMismatch { .. })
    ));
    assert!(matches!(
        InMemorySource::new(arr2(&[[f64::INFINITY]])).resolve(1),
        Err(InteractionSourceError::NonFiniteEntry { .. })
    ));
}

#[test]
fn generated_sources_record_typed_parameters_version_and_seed() {
    let resolved = GeneratedSource::new(DiagonalGenerator {
        parameters: GeneratorParameters { diagonal: -0.25 },
        randomness: GeneratorRandomness::Stochastic { seed: 42 },
    })
    .resolve(3)
    .unwrap();

    assert_eq!(resolved.values(), &Array2::from_diag_elem(3, -0.25));
    let InteractionProvenance::Generated { generator } = resolved.provenance() else {
        panic!("expected generated provenance");
    };
    assert_eq!(generator.identity(), "test.diagonal");
    assert_eq!(generator.version(), "1");
    assert_eq!(generator.parameters()["diagonal"], -0.25);
    assert_eq!(generator.seed(), Some(42));
}

#[test]
fn workflow_decodes_inline_values_and_resolves_named_file_paths() {
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
        br#"{"mode":"cartesian","axes":[]}"#,
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
        br#"{"format":"glv.interaction-matrix.v1","rows":2,"columns":2,"layout":"row_major","values":[0.0,1.0,-1.0,0.0]}"#,
    )
    .unwrap();

    let project = ScientificProject::load(directory.path()).unwrap();
    let task = project.task_config(0).unwrap();
    let inline_rows = task
        .decode_value::<Vec<Vec<f64>>>("interaction_matrix")
        .unwrap();
    let inline = JsonInteractionSource::inline(inline_rows)
        .resolve(2)
        .unwrap();
    let resolved_path = task.resolve_path("interaction_matrix_file").unwrap();
    let file = JsonInteractionSource::resolved_file(resolved_path)
        .resolve(2)
        .unwrap();

    assert_eq!(inline.values(), &Array2::<f64>::eye(2));
    assert_eq!(file.values(), &arr2(&[[0.0, 1.0], [-1.0, 0.0]]));
}

#[test]
fn artifacts_are_exact_content_addressed_and_reused() {
    let directory = TestDirectory::new("artifact-reuse");
    let scope = ExecutionScope::create_named(directory.path(), "execution").unwrap();
    let first_matrix = InMemorySource::new(arr2(&[[2.0, -1.0], [0.5, 3.0]]))
        .resolve(2)
        .unwrap();
    let second_matrix = InMemorySource::new(arr2(&[[2.0, -1.0], [0.5, 3.0]]))
        .resolve(2)
        .unwrap();

    let first = persist_interaction_matrix(&scope, &first_matrix).unwrap();
    let second = persist_interaction_matrix(&scope, &second_matrix).unwrap();
    assert_eq!(first.disposition(), ArtifactDisposition::Created);
    assert_eq!(second.disposition(), ArtifactDisposition::Reused);
    assert_eq!(first.descriptor(), second.descriptor());
    assert_eq!(first.descriptor().format(), INTERACTION_MATRIX_FORMAT);
    assert_eq!(first.descriptor().shape(), [2, 2]);
    assert_eq!(
        first.descriptor().source_kind(),
        InteractionSourceKind::InMemory
    );

    let artifact_path = scope.directory().join(first.descriptor().path());
    let bytes = fs::read(&artifact_path).unwrap();
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(first.descriptor().sha256(), digest);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
        serde_json::json!({
            "format": "glv.interaction-matrix.v1",
            "rows": 2,
            "columns": 2,
            "layout": "row_major",
            "values": [2.0, -1.0, 0.5, 3.0]
        })
    );

    let files = fs::read_dir(scope.directory().join("inputs"))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(files.len(), 1);
}

#[test]
fn persisted_artifact_is_a_checked_json_file_source() {
    let directory = TestDirectory::new("artifact-source");
    let scope = ExecutionScope::create_named(directory.path(), "execution").unwrap();
    let original = GeneratedSource::new(DiagonalGenerator {
        parameters: GeneratorParameters { diagonal: -0.5 },
        randomness: GeneratorRandomness::Deterministic,
    })
    .resolve(2)
    .unwrap();
    let persisted = persist_interaction_matrix(&scope, &original).unwrap();
    let path = scope.directory().join(persisted.descriptor().path());

    let loaded = JsonInteractionSource::resolved_file(&path)
        .resolve(2)
        .unwrap();
    assert_eq!(loaded.values(), original.values());
    assert!(matches!(
        loaded.provenance(),
        InteractionProvenance::JsonFile { path: source } if source == &path
    ));

    let mut metadata = Map::new();
    assert!(
        persisted
            .descriptor()
            .insert_into_metadata(&mut metadata)
            .is_none()
    );
    let encoded = serde_json::to_string(&metadata).unwrap();
    assert!(metadata.contains_key(INTERACTION_MATRIX_METADATA_KEY));
    assert!(!encoded.contains("values"));
    assert_eq!(
        metadata[INTERACTION_MATRIX_METADATA_KEY]["source_kind"],
        "generated"
    );
    assert_eq!(
        metadata[INTERACTION_MATRIX_METADATA_KEY]["generator"]["seed"],
        serde_json::Value::Null
    );
}

#[test]
fn malformed_json_and_artifact_collisions_fail_closed() {
    let directory = TestDirectory::new("artifact-failures");
    let malformed = directory.path().join("malformed.json");
    fs::write(&malformed, b"{").unwrap();
    assert!(matches!(
        JsonInteractionSource::resolved_file(&malformed).resolve(1),
        Err(InteractionSourceError::Json { .. })
    ));

    let wrong_count = directory.path().join("wrong-count.json");
    fs::write(
        &wrong_count,
        br#"{"format":"glv.interaction-matrix.v1","rows":2,"columns":2,"layout":"row_major","values":[1.0]}"#,
    )
    .unwrap();
    assert!(matches!(
        JsonInteractionSource::resolved_file(&wrong_count).resolve(2),
        Err(InteractionSourceError::ElementCountMismatch { .. })
    ));

    let scope = ExecutionScope::create_named(directory.path(), "execution").unwrap();
    let matrix = InMemorySource::new(arr2(&[[1.0]])).resolve(1).unwrap();
    let persisted = persist_interaction_matrix(&scope, &matrix).unwrap();
    let path = scope.directory().join(persisted.descriptor().path());
    fs::write(&path, b"different bytes").unwrap();
    assert!(matches!(
        persist_interaction_matrix(&scope, &matrix),
        Err(InteractionArtifactError::DigestCollision { .. })
    ));
}
