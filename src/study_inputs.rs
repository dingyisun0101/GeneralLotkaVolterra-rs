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

    /// Expands `{parameter}` placeholders from this resolved configuration and
    /// resolves the resulting project-path key.
    pub fn resolve_path_template(&self, template: &str) -> Result<PathBuf, GlvInputsError> {
        let key = expand_template(template, &self.configuration).map_err(|reason| {
            GlvInputsError::InvalidPathKeyTemplate {
                template: template.to_owned(),
                reason,
            }
        })?;
        if self.paths.contains(&key) {
            return self.resolve_path(&key);
        }
        let Some(path_template) = self.paths.path(template) else {
            return self.resolve_path(&key);
        };
        let path_template =
            path_template
                .to_str()
                .ok_or_else(|| GlvInputsError::InvalidPathKeyTemplate {
                    template: template.to_owned(),
                    reason: "templated project path is not valid UTF-8".to_owned(),
                })?;
        let rendered = expand_template(path_template, &self.configuration).map_err(|reason| {
            GlvInputsError::InvalidPathKeyTemplate {
                template: path_template.to_owned(),
                reason,
            }
        })?;
        let rendered = PathBuf::from(rendered);
        Ok(if rendered.is_absolute() {
            rendered
        } else {
            self.paths.project_root().join(rendered)
        })
    }

    /// Expands scalar `{parameter}` placeholders from this configuration.
    pub fn expand_template(&self, template: &str) -> Result<String, GlvInputsError> {
        expand_template(template, &self.configuration).map_err(|reason| {
            GlvInputsError::InvalidPathKeyTemplate {
                template: template.to_owned(),
                reason,
            }
        })
    }
}

fn expand_template(
    template: &str,
    configuration: &ResolvedConfiguration,
) -> Result<String, String> {
    let mut output = String::with_capacity(template.len());
    let mut remainder = template;
    while let Some(open) = remainder.find('{') {
        output.push_str(&remainder[..open]);
        let after_open = &remainder[open + 1..];
        let close = after_open
            .find('}')
            .ok_or_else(|| "unclosed placeholder".to_owned())?;
        let name = &after_open[..close];
        if name.is_empty() || name.contains('{') || name.contains('}') {
            return Err("placeholders must contain one nonempty parameter path".to_owned());
        }
        let path = if name.starts_with('/') {
            name.to_owned()
        } else {
            format!("/{name}")
        };
        let value = configuration
            .value(&path)
            .ok_or_else(|| format!("configuration does not contain `{path}`"))?;
        match value {
            Value::String(value) => output.push_str(value),
            Value::Number(value) => output.push_str(&value.to_string()),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::Null | Value::Array(_) | Value::Object(_) => {
                return Err(format!("configuration value `{path}` is not scalar"));
            }
        }
        remainder = &after_open[close + 1..];
    }
    if remainder.contains('}') {
        return Err("closing brace without an opening placeholder".to_owned());
    }
    output.push_str(remainder);
    Ok(output)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvInputsError {
    #[error(transparent)]
    Configuration(#[from] ConfigurationError),
    #[error("invalid path-key template `{template}`: {reason}")]
    InvalidPathKeyTemplate { template: String, reason: String },
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::GlvInputs;

    #[test]
    fn configuration_templates_expand_scalar_sweep_values() {
        let root = std::env::temp_dir().join(format!("glv-path-template-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("config")).unwrap();
        fs::write(root.join("config/fixed.json"), b"{}").unwrap();
        fs::write(
            root.join("config/sweep.json"),
            br#"{"mode":"cartesian","axes":{"energy":{"values":[0.2]},"sys_idx":{"values":[1]}}}"#,
        )
        .unwrap();
        fs::write(
            root.join("config/paths.json"),
            br#"{"matrix_E{energy}_sys_{sys_idx}":"matrices/E={energy}/sys={sys_idx}.json"}"#,
        )
        .unwrap();

        let configuration = GlvInputs::load(&root).unwrap().combination(0).unwrap();
        assert_eq!(
            configuration
                .resolve_path_template("matrix_E{energy}_sys_{sys_idx}")
                .unwrap(),
            root.join("matrices/E=0.2/sys=1.json")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
