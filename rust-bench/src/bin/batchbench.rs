use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use batchbench_rs::{run_benchmark, BenchmarkConfig, BenchmarkReport, RunMode};
use clap::Parser;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::{json, Value};

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
    random_requests: bool,
    output_tokens: Option<usize>,
    output_vary: Option<usize>,
    output_lognorm_mu: Option<f64>,
    output_lognorm_sigma: Option<f64>,
    output_lognorm_max: Option<usize>,
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
    /// Path to the JSONL file whose objects contain a `text` field
    #[arg(long)]
    jsonl: PathBuf,

    /// Number of concurrent users to spawn (defaults to the number of JSONL rows)
    #[arg(long)]
    users: Option<usize>,

    /// OpenAI-style model identifier to embed in each request body
    #[arg(long, default_value = "gpt-4o-mini")]
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
    #[arg(long)]
    output_vary: Option<usize>,

    /// Sample output length from log-normal distribution with this mu parameter (mean of underlying normal)
    #[arg(long)]
    output_lognorm_mu: Option<f64>,

    /// Sample output length from log-normal distribution with this sigma parameter (std dev of underlying normal)
    #[arg(long)]
    output_lognorm_sigma: Option<f64>,

    /// Maximum output tokens when using lognormal sampling (values above this are truncated)
    #[arg(long)]
    output_lognorm_max: Option<usize>,

    /// Enable verbose mode to print request/response details
    #[arg(long, short)]
    verbose: bool,

    /// Enable random request selection mode (users select random requests from entire dataset)
    #[arg(long)]
    random_requests: bool,

    /// Optional path to write CSV summary output
    #[arg(long)]
    results_csv: Option<PathBuf>,

    /// Random seed for reproducible benchmarking (default: None)
    #[arg(long)]
    seed: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

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

    if let Some(vary) = args.output_vary {
        if args.output_tokens.is_none() {
            return Err(anyhow!("output-vary requires --output-tokens to be set"));
        }
        if vary == 0 {
            return Err(anyhow!("output-vary must be greater than zero"));
        }
        if vary > i64::MAX as usize {
            return Err(anyhow!(
                "output-vary must be less than or equal to {}",
                i64::MAX
            ));
        }
    }

    // Validate lognormal parameters
    let output_lognorm = match (args.output_lognorm_mu, args.output_lognorm_sigma) {
        (Some(mu), Some(sigma)) => {
            if sigma <= 0.0 {
                return Err(anyhow!("output-lognorm-sigma must be greater than zero"));
            }
            if args.output_tokens.is_some() {
                return Err(anyhow!("--output-lognorm-mu/--output-lognorm-sigma cannot be used with --output-tokens"));
            }
            Some((mu, sigma, args.output_lognorm_max))
        }
        (Some(_), None) => {
            return Err(anyhow!(
                "--output-lognorm-mu requires --output-lognorm-sigma to be set"
            ));
        }
        (None, Some(_)) => {
            return Err(anyhow!(
                "--output-lognorm-sigma requires --output-lognorm-mu to be set"
            ));
        }
        (None, None) => None,
    };

    let api_key = args
        .api_key
        .or_else(|| std::env::var(&args.api_key_env).ok());

    let mut request_bodies = load_requests(
        &args.jsonl,
        &args.model,
        args.output_tokens,
        args.output_vary,
        args.seed,
    )
    .with_context(|| format!("failed to load requests from {}", args.jsonl.display()))?;

    if request_bodies.is_empty() {
        return Err(anyhow!(
            "{} did not contain any valid JSON records with a `text` field",
            args.jsonl.display()
        ));
    }

    let user_count = args.users.unwrap_or(request_bodies.len());
    if user_count == 0 {
        return Err(anyhow!("users must be greater than zero"));
    }

    if !args.random_requests && request_bodies.len() < user_count {
        return Err(anyhow!(
            "requested {} users but JSONL only provided {} records (use --random-requests to select randomly from dataset)",
            user_count,
            request_bodies.len()
        ));
    }

    let dataset_size = request_bodies.len();

    if !args.random_requests {
        request_bodies.truncate(user_count);
    }

    let endpoint = resolve_endpoint(&args.host, &args.endpoint);
    let endpoint_for_config = endpoint.clone(); // Clone for later use in config JSON
    let requests_per_user = args.requests_per_user.unwrap_or(1);
    if requests_per_user == 0 {
        return Err(anyhow!("requests_per_user must be greater than zero"));
    }

    // Print benchmark configuration summary
    println!("=== Benchmark Configuration ===");
    println!("Endpoint: {}", endpoint);
    println!("Model: {}", args.model);
    println!("Dataset: {}", args.jsonl.display());
    println!("Dataset size: {}", dataset_size);
    println!("Users: {}", user_count);
    if args.random_requests {
        println!("Mode: Random request selection (each user picks randomly from dataset)");
    } else {
        println!(
            "Mode: Fixed assignment (first {} dataset entries)",
            user_count
        );
    }
    println!("Requests per user: {}", requests_per_user);
    println!("Total requests: {}", user_count * requests_per_user);
    if let Some(tokens) = args.output_tokens {
        if let Some(vary) = args.output_vary {
            println!("Output tokens: {} ±{}", tokens, vary);
        } else {
            println!("Output tokens: {}", tokens);
        }
    }
    if let Some((mu, sigma, max)) = output_lognorm {
        // Calculate expected mean and median for user info
        let expected_mean = (mu + sigma * sigma / 2.0).exp();
        let expected_median = mu.exp();
        if let Some(max_val) = max {
            println!(
                "Output tokens: lognormal(μ={}, σ={}, max={}) [expected mean≈{:.0}, median≈{:.0}]",
                mu, sigma, max_val, expected_mean, expected_median
            );
        } else {
            println!(
                "Output tokens: lognormal(μ={}, σ={}) [expected mean≈{:.0}, median≈{:.0}]",
                mu, sigma, expected_mean, expected_median
            );
        }
    }
    println!("Request timeout: {}s", args.request_timeout_secs);
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
    .with_verbose(args.verbose);

    if let Some(seed) = args.seed {
        config = config.with_seed(seed);
    }

    if args.random_requests {
        config = config.with_random_request_pool(request_bodies)?;
    } else {
        config = config.with_per_user_bodies(request_bodies)?;
    }

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
            dataset_path: args.jsonl.to_string_lossy().into_owned(),
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
            random_requests: args.random_requests,
            output_tokens: args.output_tokens,
            output_vary: args.output_vary,
            output_lognorm_mu: args.output_lognorm_mu,
            output_lognorm_sigma: args.output_lognorm_sigma,
            output_lognorm_max: args.output_lognorm_max,
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

fn load_requests(
    path: &PathBuf,
    model: &str,
    output_tokens: Option<usize>,
    output_vary: Option<usize>,
    seed: Option<u64>,
) -> Result<Vec<Value>> {
    let file = File::open(path).with_context(|| format!("unable to open {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut bodies = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(trimmed)
            .with_context(|| format!("line {} is not valid JSON: {}", idx + 1, trimmed))?;

        // Try to parse as OpenAI Batch API format first (with "messages" field)
        let messages = if let Some(messages_array) = value.get("messages") {
            // OpenAI Batch API format: {"messages": [...], "model": "..."}
            if let Some(msgs) = messages_array.as_array() {
                // Validate and clone the messages array
                let mut validated_msgs = Vec::new();
                for (msg_idx, msg) in msgs.iter().enumerate() {
                    let content = msg.get("content").and_then(|v| v.as_str()).ok_or_else(|| {
                        anyhow!(
                            "line {} messages[{}] missing string field `content`",
                            idx + 1,
                            msg_idx
                        )
                    })?;
                    let role = msg.get("role").and_then(|v| v.as_str()).ok_or_else(|| {
                        anyhow!(
                            "line {} messages[{}] missing string field `role`",
                            idx + 1,
                            msg_idx
                        )
                    })?;
                    validated_msgs.push(json!({
                        "role": role,
                        "content": content,
                    }));
                }
                validated_msgs
            } else {
                return Err(anyhow!(
                    "line {} field `messages` must be an array",
                    idx + 1
                ));
            }
        } else if let Some(text_field) = value.get("text") {
            // Legacy format: {"text": "..."}
            if let Some(text_str) = text_field.as_str() {
                // Handle text as a string
                vec![json!({
                    "role": "user",
                    "content": text_str,
                })]
            } else if let Some(text_array) = text_field.as_array() {
                // Handle text as an array of maps with content and role fields
                let mut msgs = Vec::new();
                for (msg_idx, msg) in text_array.iter().enumerate() {
                    let content = msg.get("content").and_then(|v| v.as_str()).ok_or_else(|| {
                        anyhow!(
                            "line {} text[{}] missing string field `content`",
                            idx + 1,
                            msg_idx
                        )
                    })?;
                    let role = msg.get("role").and_then(|v| v.as_str()).ok_or_else(|| {
                        anyhow!(
                            "line {} text[{}] missing string field `role`",
                            idx + 1,
                            msg_idx
                        )
                    })?;
                    msgs.push(json!({
                        "role": role,
                        "content": content,
                    }));
                }
                msgs
            } else {
                return Err(anyhow!(
                    "line {} field `text` must be either a string or an array",
                    idx + 1
                ));
            }
        } else {
            return Err(anyhow!(
                "line {} missing required field `messages` or `text`",
                idx + 1
            ));
        };

        // Use the model from the JSONL if present, otherwise use the CLI argument
        let model_to_use = value.get("model").and_then(|v| v.as_str()).unwrap_or(model);

        let mut body = json!({
            "model": model_to_use,
            "messages": messages,
        });

        if let Some(tokens) = output_tokens {
            let mut final_tokens = tokens;
            if let Some(vary) = output_vary {
                // This randomization is per-JSONL-record (not per-dispatch). If --seed is set,
                // it becomes deterministic based on (seed, line_index).
                let mut rng = if let Some(seed) = seed {
                    // Mix in line index to ensure different lines get different samples.
                    let mixed_seed = seed
                        .wrapping_add((idx as u64).wrapping_mul(65537)); // prime multiplier
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

            if let Some(map) = body.as_object_mut() {
                map.insert("max_tokens".to_string(), json!(final_tokens));
                map.insert("min_tokens".to_string(), json!(final_tokens));
                // map.insert("max_completion_tokens".to_string(), json!(final_tokens));
                // map.insert("nvext".to_string(), json!({ "ignore_eos": true }));
            }
        }

        bodies.push(body);
    }

    Ok(bodies)
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

fn format_latency(latency: Option<Duration>) -> String {
    match latency {
        Some(value) => format!("{:.2}", value.as_secs_f64() * 1000.0),
        None => "n/a".to_string(),
    }
}
