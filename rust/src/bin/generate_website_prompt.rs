use std::fs;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let tokenizer_path = std::env::args().nth(1).context("tokenizer path required")?;
    let output_path = std::env::args().nth(2).context("output path required")?;
    let target: usize = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "34400".to_string())
        .parse()
        .context("invalid target token count")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| anyhow::anyhow!("failed to load tokenizer: {error}"))?;
    let mut prompt = String::from(
        "Sustained readiness test marker: glm-5.2-120-agent-12m-tpm-2026-08-21.\n\n\
You are an AI website and application builder working iteratively with a project environment. \
Help users create, inspect, edit, preview, validate, and improve websites and web applications. \
Use the available tools deliberately, preserve project conventions, make focused changes, and verify results.\n\n",
    );
    let policy = "Inspect the current application state, preserve useful existing work, apply only the requested website or app change, maintain accessible responsive behavior, and validate the result through the environment before continuing.";
    while tokenizer
        .encode(prompt.clone(), false)
        .map_err(|error| anyhow::anyhow!("tokenization failed: {error}"))?
        .len()
        < target
    {
        prompt.push_str(policy);
        prompt.push('\n');
    }
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow::anyhow!("tokenization failed: {error}"))?;
    let ids = &encoding.get_ids()[..target.min(encoding.len())];
    let prompt = tokenizer
        .decode(ids, true)
        .map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
    fs::write(&output_path, prompt).with_context(|| format!("failed to write {output_path}"))?;
    println!("wrote {output_path} with {} tokens", ids.len());
    Ok(())
}
