//! GLV-owned loading of configurations and named study paths.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use scientific_workflow::configuration::{
    ConfigurationError, ConfigurationSpace, ParameterKeyTuple, ResolvedConfiguration,
};
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

#[derive(Clone)]
pub struct GlvInputs {
    study_root: PathBuf,
    configurations: ConfigurationSpace,
    paths: Arc<BTreeMap<String, PathBuf>>,
}

#[derive(Clone)]
pub struct GlvConfiguration {
    configuration: ResolvedConfiguration,
    study_root: PathBuf,
    paths: Arc<BTreeMap<String, PathBuf>>,
}

pub fn load_glv_inputs(study_root: impl Into<PathBuf>) -> Result<GlvInputs, GlvInputsError> {
    GlvInputs::load(study_root)
}

impl GlvInputs {
    pub fn load(study_root: impl Into<PathBuf>) -> Result<Self, GlvInputsError> {
        let study_root = study_root.into();
        let configuration_directory = study_root.join("config");
        let configurations = ConfigurationSpace::load(&configuration_directory)?;
        let paths_file = configuration_directory.join("paths.json");
        let source = fs::read(&paths_file).map_err(|source| GlvInputsError::ReadPaths {
            path: paths_file.clone(),
            source,
        })?;
        let raw: BTreeMap<String, String> =
            serde_json::from_slice(&source).map_err(|source| GlvInputsError::ParsePaths {
                path: paths_file,
                source,
            })?;
        let paths = Arc::new(
            raw.into_iter()
                .map(|(key, value)| (key, PathBuf::from(value)))
                .collect(),
        );
        Ok(Self {
            study_root,
            configurations,
            paths,
        })
    }

    pub fn study_root(&self) -> &Path {
        &self.study_root
    }

    pub fn configuration_directory(&self) -> &Path {
        self.configurations.directory()
    }

    pub fn configurations(&self) -> &ConfigurationSpace {
        &self.configurations
    }

    pub fn combination_count(&self) -> u64 {
        self.configurations.combination_count()
    }

    pub fn combinations(&self) -> impl Iterator<Item = GlvConfiguration> + '_ {
        self.configurations
            .combinations()
            .map(|configuration| self.attach(configuration))
    }

    pub fn combination(&self, ordinal: u64) -> Result<GlvConfiguration, ConfigurationError> {
        self.configurations
            .combination(ordinal)
            .map(|configuration| self.attach(configuration))
    }

    pub fn resolve_path(&self, key: &str) -> Result<PathBuf, GlvInputsError> {
        resolve_path(&self.study_root, &self.paths, key)
    }

    pub fn paths(&self) -> impl Iterator<Item = (&str, &Path)> {
        self.paths
            .iter()
            .map(|(key, path)| (key.as_str(), path.as_path()))
    }

    fn attach(&self, configuration: ResolvedConfiguration) -> GlvConfiguration {
        GlvConfiguration {
            configuration,
            study_root: self.study_root.clone(),
            paths: Arc::clone(&self.paths),
        }
    }
}

impl GlvConfiguration {
    pub fn ordinal(&self) -> u64 {
        self.configuration.ordinal()
    }

    pub fn configuration(&self) -> &ResolvedConfiguration {
        &self.configuration
    }

    pub fn value(&self, key: &str) -> Option<&Value> {
        self.configuration.value(key)
    }

    pub fn decode_value<T>(&self, key: &str) -> Result<T, ConfigurationError>
    where
        T: DeserializeOwned,
    {
        self.configuration.decode_value(key)
    }

    pub fn decode_values<Values, Keys>(&self, keys: Keys) -> Result<Values, ConfigurationError>
    where
        Keys: ParameterKeyTuple<Values>,
    {
        self.configuration.decode_values(keys)
    }

    pub fn resolve_path(&self, key: &str) -> Result<PathBuf, GlvInputsError> {
        resolve_path(&self.study_root, &self.paths, key)
    }
}

fn resolve_path(
    study_root: &Path,
    paths: &BTreeMap<String, PathBuf>,
    key: &str,
) -> Result<PathBuf, GlvInputsError> {
    let path = paths.get(key).ok_or_else(|| GlvInputsError::UnknownPath {
        key: key.to_owned(),
    })?;
    Ok(if path.is_absolute() {
        path.clone()
    } else {
        study_root.join(path)
    })
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvInputsError {
    #[error(transparent)]
    Configuration(#[from] ConfigurationError),
    #[error("failed to read GLV study paths `{path}`")]
    ReadPaths {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse GLV study paths `{path}`")]
    ParsePaths {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("GLV study paths do not contain `{key}`")]
    UnknownPath { key: String },
}
