use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

#[derive(Debug, PartialEq, Eq)]
enum TokenizerSource {
    Local(PathBuf),
    Hub(String),
}

fn resolve_tokenizer_source(model_or_path: &str) -> Result<TokenizerSource> {
    let path = Path::new(model_or_path);
    if path.is_dir() {
        let tokenizer_json = path.join("tokenizer.json");
        if !tokenizer_json.is_file() {
            return Err(anyhow!(
                "local tokenizer directory {} does not contain tokenizer.json",
                path.display()
            ));
        }
        return Ok(TokenizerSource::Local(tokenizer_json));
    }
    if path.is_file() {
        return Ok(TokenizerSource::Local(path.to_path_buf()));
    }
    Ok(TokenizerSource::Hub(model_or_path.to_string()))
}

pub(crate) fn load_tokenizer(model_or_path: &str) -> Result<Tokenizer> {
    match resolve_tokenizer_source(model_or_path)? {
        TokenizerSource::Local(path) => Tokenizer::from_file(&path).map_err(|error| {
            anyhow!(
                "failed to load local tokenizer {}: {}",
                path.display(),
                error
            )
        }),
        TokenizerSource::Hub(model) => Tokenizer::from_pretrained(&model, None)
            .map_err(|error| anyhow!("failed to load tokenizer {}: {}", model, error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn local_directory_resolves_to_tokenizer_json() {
        let directory = std::env::temp_dir().join(format!(
            "batchbench-tokenizer-loader-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let tokenizer_json = directory.join("tokenizer.json");
        fs::write(&tokenizer_json, "{}").unwrap();

        assert_eq!(
            resolve_tokenizer_source(directory.to_str().unwrap()).unwrap(),
            TokenizerSource::Local(tokenizer_json)
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn hub_identifier_remains_remote() {
        assert_eq!(
            resolve_tokenizer_source("zai-org/GLM-5.2-FP8").unwrap(),
            TokenizerSource::Hub("zai-org/GLM-5.2-FP8".to_string())
        );
    }
}
