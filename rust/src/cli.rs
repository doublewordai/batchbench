use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Duration;

use crate::{
    generate_requests, run_benchmark, BenchmarkConfig, BenchmarkReport, DistMode, GenerateOptions,
    RequestEntry, RunMode,
};
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use std::cmp;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct CsvResult {
    timestamp: String,
    model: String,
    dataset_path: String,
    dataset_size: usize,
    users: usize,
    requests_per_user: usize,
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    total_prompt_tokens: u64,
    total_completion_tokens: u64,
    total_duration_seconds: f64,
    requests_per_second: f64,
    prompt_tokens_per_second: f64,
    completion_tokens_per_second: f64,
    latency_p50_ms: Option<f64>,
    latency_p90_ms: Option<f64>,
    latency_p99_ms: Option<f64>,
    output_tokens: Option<usize>,
    output_vary: Option<usize>,
    output_lognorm_mu: Option<f64>,
    output_lognorm_sigma: Option<f64>,
    output_lognorm_max: Option<usize>,
    sglang: bool,
    request_timeout_secs: u64,
    max_retries: usize,
    retry_delay_ms: u64,
    host: String,
    endpoint: String,
}

#[derive(Parser, Debug)]
#[command(
    name = "batchbench",
    about = "Drive batchbench-rs benchmarks from the CLI"
)]
struct Args {
    /// Fraction of tokens shared as prefix across generated prompts (0.0-1.0)
    #[arg(
        long = "input-prefix-overlap",
        alias = "gen-prefix-overlap",
        default_value_t = 0.0
    )]
    input_prefix_overlap: f64,

    /// Target number of input tokens per prompt (0 = disable)
    #[arg(
        long = "input-tokens",
        alias = "gen-approx-input-tokens",
        default_value_t = 0
    )]
    input_tokens: usize,

    /// Apply a +/- uniform variation when --input-tokens is provided
    #[arg(
        long = "input-vary",
        alias = "gen-token-tolerance",
        default_value_t = 0
    )]
    input_vary: usize,

    /// Sample input length from log-normal distribution with this mu parameter (mean of underlying normal)
    #[arg(long = "input-lognorm-mu")]
    input_lognorm_mu: Option<f64>,

    /// Sample input length from log-normal distribution with this median (preferred over mu)
    #[arg(long = "input-lognorm-median", alias = "gen-dist-median")]
    input_lognorm_median: Option<f64>,

    /// Sample input length from log-normal distribution with this sigma parameter (std dev of underlying normal)
    #[arg(long = "input-lognorm-sigma", alias = "gen-dist-sigma")]
    input_lognorm_sigma: Option<f64>,

    /// Maximum input tokens when using lognormal sampling (values above this are truncated)
    #[arg(long = "input-lognorm-max", alias = "gen-dist-max")]
    input_lognorm_max: Option<usize>,

    /// Deprecated legacy input distribution mode selector
    #[arg(long = "gen-dist-mode", hide = true)]
    legacy_gen_dist_mode: Option<String>,

    /// Path to a JSONL dataset. Each line may contain {"text": "..."},
    /// {"body": {...}}, or a full request body object.
    #[arg(long, alias = "jsonl")]
    dataset_jsonl: Option<PathBuf>,

    /// Number of concurrent users to spawn (default: 1)
    #[arg(long)]
    users: Option<usize>,

    /// OpenAI-style model identifier to embed in each request body
    #[arg(long, default_value = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")]
    model: String,

    /// Host to target (e.g. https://api.openai.com)
    #[arg(long, default_value = "https://api.openai.com")]
    host: String,

    /// Endpoint path or full URL (e.g. /v1/chat/completions)
    #[arg(long, default_value = "/v1/chat/completions")]
    endpoint: String,

    /// Requests per user (defaults to 1)
    #[arg(long)]
    requests_per_user: Option<usize>,

    /// API key to use; if omitted an environment variable is read
    #[arg(long)]
    api_key: Option<String>,

    /// Environment variable name to read the API key from when --api-key is not supplied
    #[arg(long, default_value = "OPENAI_API_KEY")]
    api_key_env: String,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 60)]
    request_timeout_secs: u64,

    /// Maximum retries per request
    #[arg(long, default_value_t = 2)]
    max_retries: usize,

    /// Base retry delay in milliseconds
    #[arg(long, default_value_t = 250)]
    retry_delay_ms: u64,

    /// Force the model to emit exactly this many new tokens
    #[arg(long)]
    output_tokens: Option<usize>,

    /// Apply a +/- uniform variation when --output-tokens is provided
    #[arg(long, default_value_t = 0)]
    output_vary: usize,

    /// Sample output length from log-normal distribution with this mu parameter (mean of underlying normal)
    #[arg(long)]
    output_lognorm_mu: Option<f64>,

    /// Sample output length from log-normal distribution with this median (preferred over mu)
    #[arg(long)]
    output_lognorm_median: Option<f64>,

    /// Sample output length from log-normal distribution with this sigma parameter (std dev of underlying normal)
    #[arg(long)]
    output_lognorm_sigma: Option<f64>,

    /// Maximum output tokens when using lognormal sampling (values above this are truncated)
    #[arg(long)]
    output_lognorm_max: Option<usize>,

    /// Use SGLang token parameters (min_new_tokens/max_new_tokens) instead of min_tokens/max_tokens
    #[arg(long)]
    sglang: bool,

    /// Enable verbose mode to print request/response details
    #[arg(long, short)]
    verbose: bool,

    /// Dry run (do not send HTTP requests; log request selection and token settings)
    #[arg(long)]
    dry_run: bool,

    // Tokenizers reuse the request model; no separate flags.
    /// Optional path to write CSV summary output
    #[arg(long)]
    results_csv: Option<PathBuf>,

    /// Random seed for reproducible benchmarking (default: None)
    #[arg(long)]
    seed: Option<u64>,
}

enum ParsedArgs {
    Ready(Args),
    Displayed,
}

fn parse_args<I, T>(argv: I) -> Result<ParsedArgs>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Args::try_parse_from(argv) {
        Ok(args) => Ok(ParsedArgs::Ready(args)),
        Err(err) => match err.kind() {
            clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
                err.print()
                    .map_err(|print_err| anyhow!(print_err.to_string()))?;
                Ok(ParsedArgs::Displayed)
            }
            _ => Err(anyhow!(err.to_string())),
        },
    }
}

pub fn run_from_env() -> Result<()> {
    match parse_args(std::env::args_os()) {
        Ok(ParsedArgs::Ready(args)) => run_with_runtime(args),
        Ok(ParsedArgs::Displayed) => Ok(()),
        Err(err) => Err(err),
    }
}

pub fn run_from_argv(argv: Vec<String>) -> Result<()> {
    let mut full_argv = Vec::with_capacity(argv.len() + 1);
    full_argv.push("batchbench".to_string());
    full_argv.extend(argv);
    match parse_args(full_argv) {
        Ok(ParsedArgs::Ready(args)) => run_with_runtime(args),
        Ok(ParsedArgs::Displayed) => Ok(()),
        Err(err) => Err(err),
    }
}

fn run_with_runtime(args: Args) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime for CLI")?;
    runtime.block_on(run(args))
}

async fn run(args: Args) -> Result<()> {
    let legacy_input_dist_mode = if let Some(mode) = args.legacy_gen_dist_mode.as_deref() {
        Some(parse_dist_mode(mode)?)
    } else {
        None
    };

    let input_lognorm_requested = args.input_lognorm_mu.is_some()
        || args.input_lognorm_median.is_some()
        || args.input_lognorm_sigma.is_some()
        || args.input_lognorm_max.is_some();
    let dataset_jsonl = args.dataset_jsonl.clone();
    let input_generation_requested = args.input_tokens > 0
        || args.input_vary > 0
        || input_lognorm_requested
        || args.input_prefix_overlap != 0.0
        || legacy_input_dist_mode.is_some();

    if dataset_jsonl.is_some() && input_generation_requested {
        return Err(anyhow!(
            "--dataset-jsonl cannot be combined with input generation flags"
        ));
    }

    if args.input_tokens == 0 && args.input_vary > 0 {
        return Err(anyhow!("input-vary requires --input-tokens to be set"));
    }

    if args.input_lognorm_mu.is_some() && args.input_lognorm_median.is_some() {
        return Err(anyhow!(
            "provide either --input-lognorm-mu or --input-lognorm-median, not both"
        ));
    }

    let input_dist_mode = match legacy_input_dist_mode {
        Some(DistMode::Fixed) => {
            if input_lognorm_requested {
                return Err(anyhow!(
                    "--gen-dist-mode fixed cannot be combined with --input-lognorm-* flags"
                ));
            }
            DistMode::Fixed
        }
        Some(DistMode::LogNormal) => DistMode::LogNormal,
        None => {
            if input_lognorm_requested {
                DistMode::LogNormal
            } else {
                DistMode::Fixed
            }
        }
    };

    if args.input_tokens > 0 && input_dist_mode == DistMode::LogNormal {
        return Err(anyhow!(
            "lognormal input sampling cannot be combined with --input-tokens"
        ));
    }

    let (input_dist_mu, input_dist_median, input_dist_sigma, input_dist_max) = if input_dist_mode
        == DistMode::LogNormal
    {
        let sigma = if let Some(sigma) = args.input_lognorm_sigma {
            if sigma <= 0.0 {
                return Err(anyhow!("input-lognorm-sigma must be greater than zero"));
            }
            sigma
        } else if legacy_input_dist_mode == Some(DistMode::LogNormal) {
            0.5
        } else {
            return Err(anyhow!(
                "--input-lognorm-sigma is required when using lognormal input generation"
            ));
        };

        if let Some(median) = args.input_lognorm_median {
            if median <= 0.0 {
                return Err(anyhow!("--input-lognorm-median must be greater than zero"));
            }
            (None, Some(median), sigma, args.input_lognorm_max)
        } else if let Some(mu) = args.input_lognorm_mu {
            (Some(mu), None, sigma, args.input_lognorm_max)
        } else {
            return Err(anyhow!(
                    "--input-lognorm-median or --input-lognorm-mu is required when using lognormal input generation"
                ));
        }
    } else {
        (None, None, 0.5, None)
    };

    if let Some(tokens) = args.output_tokens {
        if tokens == 0 {
            return Err(anyhow!("output-tokens must be greater than zero"));
        }
        if tokens > i64::MAX as usize {
            return Err(anyhow!(
                "output-tokens must be less than or equal to {}",
                i64::MAX
            ));
        }
    }

    if args.output_tokens.is_none() {
        if args.output_vary > 0 {
            return Err(anyhow!("output-vary requires --output-tokens to be set"));
        }
    } else if args.output_vary > i64::MAX as usize {
        return Err(anyhow!(
            "output-vary must be less than or equal to {}",
            i64::MAX
        ));
    }

    // Validate lognormal parameters (accept median -> mu conversion)
    let output_lognorm = {
        if args.output_tokens.is_some()
            && (args.output_lognorm_mu.is_some() || args.output_lognorm_median.is_some())
        {
            return Err(anyhow!(
                "lognormal output sampling cannot be combined with --output-tokens"
            ));
        }

        // Reject mixing mu and median simultaneously
        if args.output_lognorm_mu.is_some() && args.output_lognorm_median.is_some() {
            return Err(anyhow!(
                "provide either --output-lognorm-mu or --output-lognorm-median, not both"
            ));
        }

        // Resolve mu
        let mu = if let Some(median) = args.output_lognorm_median {
            if median <= 0.0 {
                return Err(anyhow!("--output-lognorm-median must be greater than zero"));
            }
            median.ln()
        } else if let Some(mu) = args.output_lognorm_mu {
            mu
        } else {
            // no lognorm
            f64::NAN
        };

        if mu.is_nan() {
            None
        } else if let Some(sigma) = args.output_lognorm_sigma {
            if sigma <= 0.0 {
                return Err(anyhow!("output-lognorm-sigma must be greater than zero"));
            }
            Some((mu, sigma, args.output_lognorm_max))
        } else {
            return Err(anyhow!(
                "--output-lognorm-sigma is required when using lognormal output"
            ));
        }
    };

    let api_key = args
        .api_key
        .or_else(|| std::env::var(&args.api_key_env).ok());

    // Resolve request multiplicity up front
    let requests_per_user = args.requests_per_user.unwrap_or(1);
    if requests_per_user == 0 {
        return Err(anyhow!("requests_per_user must be greater than zero"));
    }

    let (mut request_bodies, dataset_label, dataset_is_generated) =
        if let Some(dataset_path) = dataset_jsonl.as_ref() {
            let requests = load_dataset_jsonl(dataset_path, &args.model)?;
            let label = dataset_path.display().to_string();
            (requests, label, false)
        } else {
            let user_count = args.users.unwrap_or(1);
            if user_count == 0 {
                return Err(anyhow!("users must be greater than zero"));
            }
            let total_requests = user_count
                .checked_mul(requests_per_user)
                .ok_or_else(|| anyhow!("users * requests_per_user overflowed"))?;

            let target_tokens = if args.input_tokens > 0 {
                Some(args.input_tokens)
            } else {
                None
            };

            let gen_opts = GenerateOptions {
                count: total_requests,
                prefix_overlap: args.input_prefix_overlap,
                target_tokens,
                token_tolerance: if args.input_tokens > 0 {
                    Some(args.input_vary)
                } else {
                    None
                },
                tokenizer_model: args.model.clone(),
                dist_mode: input_dist_mode,
                dist_mu: input_dist_mu,
                dist_median: input_dist_median,
                dist_sigma: input_dist_sigma,
                dist_max: input_dist_max,
                seed: args.seed,
            };

            let requests = generate_requests(&gen_opts, &args.model)?;
            let label = format!(
                "generated (count={}, tokenizer={})",
                total_requests, args.model
            );
            (requests, label, true)
        };

    let dataset_size = request_bodies.len();
    let user_count = if dataset_is_generated {
        args.users.unwrap_or(1)
    } else if let Some(users) = args.users {
        users
    } else {
        dataset_size / requests_per_user
    };
    if user_count == 0 {
        return Err(anyhow!("users must be greater than zero"));
    }

    let total_requests = user_count
        .checked_mul(requests_per_user)
        .ok_or_else(|| anyhow!("users * requests_per_user overflowed"))?;
    if total_requests > dataset_size {
        return Err(anyhow!(
            "dataset contains {} request entries but benchmark needs {} (users {} * requests-per-user {})",
            dataset_size,
            total_requests,
            user_count,
            requests_per_user
        ));
    }

    let output_vary = if args.output_tokens.is_some() {
        Some(args.output_vary)
    } else {
        None
    };
    apply_output_tokens(
        &mut request_bodies,
        args.output_tokens,
        output_vary,
        args.seed,
        args.sglang,
    )?;

    // Print input token histogram
    let input_tokens: Vec<usize> = request_bodies.iter().map(|r| r.input_tokens).collect();
    if !input_tokens.is_empty() {
        print_histogram("Input tokens", &input_tokens, 20, 50);
    }

    let endpoint = resolve_endpoint(&args.host, &args.endpoint);
    let endpoint_for_config = endpoint.clone(); // Clone for later use in config JSON

    // Print benchmark configuration summary
    println!("=== Benchmark Configuration ===");
    println!("Endpoint: {}", endpoint);
    println!("Model: {}", args.model);
    println!("Dataset: {}", dataset_label);
    println!("Dataset size: {}", dataset_size);
    println!("Users: {}", user_count);
    println!(
        "Mode: Deterministic mapping m*N+n into {}",
        if dataset_is_generated {
            "generated dataset"
        } else {
            "dataset"
        }
    );
    println!("Requests per user: {}", requests_per_user);
    println!("Total requests: {}", total_requests);
    if let Some(tokens) = args.output_tokens {
        let vary = args.output_vary;
        if vary == 0 {
            println!("Output tokens: {} (no variation)", tokens);
        } else {
            println!("Output tokens: {} ±{}", tokens, vary);
        }
    }
    if let Some((mu, sigma, max)) = output_lognorm {
        // Calculate expected mean and median for user info
        let expected_mean = (mu + sigma * sigma / 2.0).exp();
        let expected_median = mu.exp();
        if let Some(max_val) = max {
            println!(
                "Output tokens: lognormal(median≈{:.0}, σ={}, max={}) [expected mean≈{:.0}, μ={:.3}]",
                expected_median, sigma, max_val, expected_mean, mu
            );
        } else {
            println!(
                "Output tokens: lognormal(median≈{:.0}, σ={}) [expected mean≈{:.0}, μ={:.3}]",
                expected_median, sigma, expected_mean, mu
            );
        }
    }
    println!("Request timeout: {}s", args.request_timeout_secs);
    println!(
        "Output token params: {}",
        if args.sglang {
            "sglang (min_new_tokens/max_new_tokens)"
        } else {
            "default (min_tokens/max_tokens)"
        }
    );
    println!("Max retries: {}", args.max_retries);
    println!("Retry delay: {}ms", args.retry_delay_ms);
    println!("===============================\n");

    let mode = RunMode::Finite { requests_per_user };

    let mut config = BenchmarkConfig::try_new(
        endpoint,
        api_key,
        user_count,
        mode,
        request_bodies
            .first()
            .cloned()
            .expect("non-empty request bodies"),
    )?
    .with_request_timeout(Duration::from_secs(args.request_timeout_secs))
    .with_retry(args.max_retries, Duration::from_millis(args.retry_delay_ms))
    .with_verbose(args.verbose)
    .with_dry_run(args.dry_run)
    .with_sglang(args.sglang);

    if let Some(seed) = args.seed {
        config = config.with_seed(seed);
    }

    config = config.with_request_list(request_bodies)?;

    if let Some((mu, sigma, max)) = output_lognorm {
        config = config.with_output_lognorm(mu, sigma, max);
    }

    let start_time = chrono::Utc::now();
    let report = run_benchmark(config).await?;

    print_summary(&report)?;

    if let Some(csv_path) = args.results_csv.as_ref() {
        let record = CsvResult {
            timestamp: start_time.to_rfc3339(),
            model: args.model.clone(),
            dataset_path: dataset_label.clone(),
            dataset_size,
            users: user_count,
            requests_per_user,
            total_requests: report.total_requests,
            successful_requests: report.successful_requests,
            failed_requests: report.failed_requests,
            total_prompt_tokens: report.total_prompt_tokens,
            total_completion_tokens: report.total_completion_tokens,
            total_duration_seconds: report.total_duration.as_secs_f64(),
            requests_per_second: report.requests_per_second,
            prompt_tokens_per_second: report.prompt_tokens_per_second,
            completion_tokens_per_second: report.completion_tokens_per_second,
            latency_p50_ms: report.latency_p50.map(|d| d.as_secs_f64() * 1000.0),
            latency_p90_ms: report.latency_p90.map(|d| d.as_secs_f64() * 1000.0),
            latency_p99_ms: report.latency_p99.map(|d| d.as_secs_f64() * 1000.0),
            output_tokens: args.output_tokens,
            output_vary: if args.output_tokens.is_some() {
                Some(args.output_vary)
            } else {
                None
            },
            output_lognorm_mu: args.output_lognorm_mu,
            output_lognorm_sigma: args.output_lognorm_sigma,
            output_lognorm_max: args.output_lognorm_max,
            sglang: args.sglang,
            request_timeout_secs: args.request_timeout_secs,
            max_retries: args.max_retries,
            retry_delay_ms: args.retry_delay_ms,
            host: args.host.clone(),
            endpoint: endpoint_for_config.clone(),
        };

        match write_results_csv(csv_path.as_path(), &record) {
            Ok(_) => println!("Wrote benchmark summary to {}", csv_path.display()),
            Err(err) => eprintln!(
                "Failed to write benchmark summary to {}: {}",
                csv_path.display(),
                err
            ),
        }
    }

    Ok(())
}

fn parse_dist_mode(mode: &str) -> Result<DistMode> {
    match mode.to_ascii_lowercase().as_str() {
        "fixed" => Ok(DistMode::Fixed),
        "lognormal" => Ok(DistMode::LogNormal),
        other => Err(anyhow!(
            "invalid --gen-dist-mode '{}'; use fixed or lognormal",
            other
        )),
    }
}

fn load_dataset_jsonl(path: &Path, model: &str) -> Result<Vec<RequestEntry>> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open dataset JSONL {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut requests = Vec::new();

    for (line_number, line) in reader.lines().enumerate() {
        let line_number = line_number + 1;
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from dataset JSONL {}",
                line_number,
                path.display()
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value: serde_json::Value = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "failed to parse dataset JSONL {} line {} as JSON",
                path.display(),
                line_number
            )
        })?;
        requests.push(dataset_value_to_request_entry(
            value,
            requests.len(),
            line_number,
            model,
        )?);
    }

    if requests.is_empty() {
        return Err(anyhow!("dataset JSONL {} is empty", path.display()));
    }

    Ok(requests)
}

fn dataset_value_to_request_entry(
    value: serde_json::Value,
    line_idx: usize,
    line_number: usize,
    model: &str,
) -> Result<RequestEntry> {
    let input_tokens = value
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(0);

    if let Some(body) = value.get("body").cloned() {
        if !body.is_object() {
            return Err(anyhow!(
                "dataset line {} has a body field, but body must be a JSON object",
                line_number
            ));
        }
        return Ok(RequestEntry {
            body,
            line_idx,
            input_tokens,
        });
    }

    if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
        return Ok(RequestEntry {
            body: json!({
                "messages": [
                    {"role": "user", "content": text}
                ],
                "model": model,
            }),
            line_idx,
            input_tokens,
        });
    }

    let mut body = value;
    let Some(map) = body.as_object_mut() else {
        return Err(anyhow!(
            "dataset line {} must be a JSON object, or contain a body object or text string",
            line_number
        ));
    };
    map.remove("input_tokens");
    map.remove("line_idx");

    Ok(RequestEntry {
        body,
        line_idx,
        input_tokens,
    })
}

/// Apply output token settings to generated request bodies, honoring optional variation and seed.
fn apply_output_tokens(
    bodies: &mut [RequestEntry],
    output_tokens: Option<usize>,
    output_vary: Option<usize>,
    seed: Option<u64>,
    use_sglang: bool,
) -> Result<()> {
    if output_tokens.is_none() {
        return Ok(());
    }

    let tokens = output_tokens.unwrap();
    if tokens == 0 {
        return Err(anyhow!("output-tokens must be greater than zero"));
    }

    for (idx, entry) in bodies.iter_mut().enumerate() {
        let mut final_tokens = tokens;
        if let Some(vary) = output_vary {
            if vary > 0 {
                // Deterministic per-entry variation when seed is provided
                let mut rng = if let Some(seed) = seed {
                    let mixed_seed = seed.wrapping_add((idx as u64).wrapping_mul(65537));
                    rand::rngs::StdRng::seed_from_u64(mixed_seed)
                } else {
                    rand::rngs::StdRng::from_rng(rand::thread_rng())
                        .map_err(|e| anyhow!("failed to initialize rng: {}", e))?
                };
                let vary_i64 = vary as i64;
                let base_i64 = tokens as i64;
                let delta = rng.gen_range(-vary_i64..=vary_i64);
                let adjusted = (base_i64 + delta).max(1);
                final_tokens = adjusted as usize;
            }
        }

        if let Some(map) = entry.body.as_object_mut() {
            let (max_key, min_key) = output_token_field_names(use_sglang);
            map.insert(max_key.to_string(), json!(final_tokens));
            map.insert(min_key.to_string(), json!(final_tokens));
        }
    }

    Ok(())
}

fn output_token_field_names(use_sglang: bool) -> (&'static str, &'static str) {
    if use_sglang {
        ("max_new_tokens", "min_new_tokens")
    } else {
        ("max_tokens", "min_tokens")
    }
}

fn resolve_endpoint(host: &str, endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        return endpoint.to_string();
    }

    let normalized_host = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("https://{}", host.trim_end_matches('/'))
    };

    format!("{}/{}", normalized_host, endpoint.trim_start_matches('/'))
}

fn print_summary(report: &BenchmarkReport) -> Result<()> {
    println!(
        "Total requests: {} (success {}, failure {})",
        report.total_requests, report.successful_requests, report.failed_requests
    );
    println!(
        "Token totals: prompt {} completion {}",
        report.total_prompt_tokens, report.total_completion_tokens
    );
    println!(
        "Total duration: {:.2}s",
        report.total_duration.as_secs_f64()
    );
    println!(
        "Throughput: prompt {:.2} tok/s, completion {:.2} tok/s, requests {:.2} req/s",
        report.prompt_tokens_per_second,
        report.completion_tokens_per_second,
        report.requests_per_second
    );

    let p50 = format_latency(report.latency_p50);
    let p90 = format_latency(report.latency_p90);
    let p99 = format_latency(report.latency_p99);
    println!("Latency (ms): p50={} p90={} p99={}", p50, p90, p99);

    if !report.failures.is_empty() {
        println!("Failures: {}", report.failures.len());
        for failure in &report.failures {
            println!("  user {}: {}", failure.user_id, failure.error);
        }
    }

    Ok(())
}

fn write_results_csv(path: &Path, record: &CsvResult) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }

    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    writer
        .serialize(record)
        .with_context(|| format!("failed to serialize results to {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn print_histogram(label: &str, data: &[usize], bins: usize, bar_width: usize) {
    if data.is_empty() || bins == 0 {
        return;
    }
    let min = *data.iter().min().unwrap_or(&0);
    let max = *data.iter().max().unwrap_or(&0);
    let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / data.len() as f64;
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[((sorted.len() as f64 * 0.95).round() as usize).min(sorted.len() - 1)];
    let p99 = sorted[((sorted.len() as f64 * 0.99).round() as usize).min(sorted.len() - 1)];

    let span = if max > min { max - min } else { 1 };
    let bin_width = cmp::max(1, (span as f64 / bins as f64).ceil() as usize);
    let mut counts = vec![0usize; bins];
    for &v in data {
        let idx = cmp::min((v - min) / bin_width, bins - 1);
        counts[idx] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&1);

    eprintln!("\n== {} (n={}) ==", label, data.len());
    eprintln!(
        "min={} mean={:.1} median={} p95={} p99={} max={}",
        min, mean, median, p95, p99, max
    );
    for (i, count) in counts.iter().enumerate() {
        let start = min + i * bin_width;
        let end = start + bin_width;
        let bar_len = if max_count > 0 {
            ((count * bar_width) as f64 / max_count as f64).round() as usize
        } else {
            0
        };
        let bar = "#".repeat(bar_len);
        eprintln!(
            "{:>6}-{:>6} | {:<bar_width$} {}",
            start,
            end,
            bar,
            count,
            bar_width = bar_width
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_text_line_becomes_chat_request() {
        let entry =
            dataset_value_to_request_entry(json!({"text": "hello", "input_tokens": 3}), 0, 1, "m")
                .unwrap();

        assert_eq!(entry.line_idx, 0);
        assert_eq!(entry.input_tokens, 3);
        assert_eq!(entry.body["model"], "m");
        assert_eq!(entry.body["messages"][0]["content"], "hello");
    }

    #[test]
    fn dataset_body_line_uses_body_object() {
        let entry = dataset_value_to_request_entry(
            json!({"body": {"model": "x", "prompt": "hello"}, "input_tokens": 5}),
            4,
            5,
            "ignored",
        )
        .unwrap();

        assert_eq!(entry.line_idx, 4);
        assert_eq!(entry.input_tokens, 5);
        assert_eq!(entry.body, json!({"model": "x", "prompt": "hello"}));
    }

    #[test]
    fn dataset_full_request_line_drops_metadata_fields() {
        let entry = dataset_value_to_request_entry(
            json!({"model": "x", "messages": [], "input_tokens": 7, "line_idx": 99}),
            1,
            2,
            "ignored",
        )
        .unwrap();

        assert_eq!(entry.input_tokens, 7);
        assert_eq!(entry.body, json!({"model": "x", "messages": []}));
    }
}

fn format_latency(latency: Option<Duration>) -> String {
    match latency {
        Some(value) => format!("{:.2}", value.as_secs_f64() * 1000.0),
        None => "n/a".to_string(),
    }
}
