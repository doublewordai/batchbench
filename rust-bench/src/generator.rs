use anyhow::{anyhow, Context, Result};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::LogNormal;
use tokenizers::Tokenizer;

use crate::config::RequestEntry;

#[derive(Clone, Debug)]
pub enum DistMode {
    Fixed,
    LogNormal,
}

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub count: usize,
    pub prefix_overlap: f64,
    pub target_tokens: Option<usize>,
    pub token_tolerance: Option<usize>,
    pub tokenizer_model: String,
    pub dist_mode: DistMode,
    pub dist_median: Option<f64>,
    pub dist_sigma: f64,
    pub dist_max: Option<usize>,
    pub seed: Option<u64>,
}

fn resolve_tolerance(target_tokens: usize, explicit: Option<usize>) -> usize {
    if target_tokens == 0 {
        return 0;
    }
    if let Some(explicit) = explicit {
        return explicit;
    }
    // Match Python behavior: max(5, 5% of target)
    std::cmp::max(5, (target_tokens as f64 * 0.05).round() as usize)
}

pub fn generate_requests(opts: &GenerateOptions, model: &str) -> Result<Vec<RequestEntry>> {
    if opts.count == 0 {
        return Err(anyhow!("generate count must be greater than zero"));
    }

    let mut rng: StdRng = match opts.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(rand::thread_rng())
            .context("failed to initialize rng from thread_rng")?,
    };

    let tokenizer = Tokenizer::from_pretrained(&opts.tokenizer_model, None)
        .map_err(|e| anyhow!("failed to load tokenizer {}: {}", opts.tokenizer_model, e))?;

    // Sample sequence lengths
    let mut sequence_lengths: Vec<usize> = Vec::with_capacity(opts.count);
    let tolerance = opts
        .target_tokens
        .map(|t| resolve_tolerance(t, opts.token_tolerance));

    match opts.dist_mode {
        DistMode::LogNormal => {
            let median = opts
                .dist_median
                .ok_or_else(|| anyhow!("--gen-dist-median is required for lognormal mode"))?;
            let mu = median.ln();
            let sigma = opts.dist_sigma;
            if sigma <= 0.0 {
                return Err(anyhow!("lognormal sigma must be > 0"));
            }
            let lognorm = LogNormal::new(mu, sigma)
                .map_err(|e| anyhow!("failed to create lognormal distribution: {}", e))?;
            for _ in 0..opts.count {
                let sample = lognorm.sample(&mut rng);
                let mut length = sample.round().max(1.0) as usize;
                if let Some(max_val) = opts.dist_max {
                    length = length.min(max_val);
                }
                sequence_lengths.push(length);
            }
        }
        DistMode::Fixed => {
            for _ in 0..opts.count {
                let length = if let Some(target) = opts.target_tokens {
                    let tol = tolerance.unwrap_or(0);
                    let lower = std::cmp::max(1, target.saturating_sub(tol));
                    let upper = target.saturating_add(tol);
                    if lower == upper {
                        lower
                    } else {
                        rng.gen_range(lower..=upper)
                    }
                } else {
                    1
                };
                sequence_lengths.push(length);
            }
        }
    }

    // Compute shared prefix
    let prefix_ratio = opts.prefix_overlap.clamp(0.0, 1.0);
    let min_length = sequence_lengths.iter().cloned().min().unwrap_or(0);
    let mut prefix_length = if min_length > 0 {
        (min_length as f64 * prefix_ratio).floor() as usize
    } else {
        0
    };
    if prefix_ratio > 0.0 && prefix_length == 0 && min_length > 0 {
        prefix_length = 1;
    }

    let prefix_ids: Vec<u32> = if prefix_length > 0 {
        (0..prefix_length)
            .map(|_| rng.gen_range(1u32..=10000u32))
            .collect()
    } else {
        Vec::new()
    };

    let mut requests: Vec<RequestEntry> = Vec::with_capacity(opts.count);
    for (idx, &seq_len) in sequence_lengths.iter().enumerate() {
        // Build token ids with shared prefix + unique tail
        let mut token_ids: Vec<u32> = (0..seq_len)
            .map(|_| rng.gen_range(1u32..=10000u32))
            .collect();

        let unique_tail = if prefix_length >= seq_len {
            Vec::new()
        } else {
            token_ids.split_off(prefix_length)
        };

        let mut final_ids = prefix_ids.clone();
        final_ids.extend_from_slice(&unique_tail);

        let mut prompt = tokenizer
            .decode(&final_ids, true)
            .unwrap_or_default();

        if prompt.trim().is_empty() {
            // Fallback: join ids when decoding yields empty text
            prompt = final_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(" ");
        }

        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "model": model,
        });

        requests.push(RequestEntry {
            body,
            line_idx: idx,
            input_tokens: seq_len,
        });
    }

    Ok(requests)
}
