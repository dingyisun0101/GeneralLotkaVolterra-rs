//! GLV-owned loading of configurations and named study paths.

use std::path::{Path, PathBuf};

use scientific_workflow::configuration::{
    ConfigurationError, ConfigurationSpace, ParameterKeyTuple, ProjectPaths, ResolvedConfiguration,
};
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

#[derive(Clone)]
pub struct GlvInputs {
    configurations: ConfigurationSpace,
    paths: ProjectPaths,
}

#[derive(Clone)]
pub struct GlvConfiguration {
    configuration: ResolvedConfiguration,
    paths: ProjectPaths,
}

impl GlvInputs {
    pub fn load(study_root: impl Into<PathBuf>) -> Result<Self, GlvInputsError> {
        let study_root = study_root.into();
        let configuration_directory = study_root.join("config");
        let configurations = ConfigurationSpace::load(&configuration_directory)?;
        let paths = ProjectPaths::load(study_root)?;
        Ok(Self {
            configurations,
            paths,
        })
    }

    pub fn study_root(&self) -> &Path {
        self.paths.project_root()
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
        Ok(self.paths.resolve_path(key)?)
    }

    pub fn paths(&self) -> impl Iterator<Item = (&str, &Path)> {
        self.paths.iter()
    }

    pub fn project_paths(&self) -> &ProjectPaths {
        &self.paths
    }

    fn attach(&self, configuration: ResolvedConfiguration) -> GlvConfiguration {
        GlvConfiguration {
            configuration,
            paths: self.paths.clone(),
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
        Ok(self.paths.resolve_path(key)?)
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvInputsError {
    #[error(transparent)]
    Configuration(#[from] ConfigurationError),
}
