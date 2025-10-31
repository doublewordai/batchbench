use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use batchbench_rs::{run_benchmark, BenchmarkConfig, BenchmarkReport, RunMode};
use clap::Parser;
use rand::Rng;
use serde_json::{json, Value};

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

    let api_key = args
        .api_key
        .or_else(|| std::env::var(&args.api_key_env).ok());

    let mut request_bodies = load_requests(
        &args.jsonl,
        &args.model,
        args.output_tokens,
        args.output_vary,
    )
    .with_context(|| format!("failed to load requests from {}", args.jsonl.display()))?;

    if request_bodies.is_empty() {
        return Err(anyhow!(
            "{} did not contain any JSON records with a `text` field",
            args.jsonl.display()
        ));
    }

    let user_count = args.users.unwrap_or(request_bodies.len());
    if user_count == 0 {
        return Err(anyhow!("users must be greater than zero"));
    }

    if request_bodies.len() < user_count {
        return Err(anyhow!(
            "requested {} users but JSONL only provided {} records",
            user_count,
            request_bodies.len()
        ));
    }

    request_bodies.truncate(user_count);

    let endpoint = resolve_endpoint(&args.host, &args.endpoint);
    let requests_per_user = args.requests_per_user.unwrap_or(1);
    if requests_per_user == 0 {
        return Err(anyhow!("requests_per_user must be greater than zero"));
    }

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
    .with_retry(args.max_retries, Duration::from_millis(args.retry_delay_ms));

    config = config.with_per_user_bodies(request_bodies)?;

    let report = run_benchmark(config).await?;

    print_summary(&report)?;

    Ok(())
}

fn load_requests(
    path: &PathBuf,
    model: &str,
    output_tokens: Option<usize>,
    output_vary: Option<usize>,
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
        let text = value
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("line {} missing string field `text`", idx + 1))?;
        let mut body = json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": text,
                }
            ]
        });

        if let Some(tokens) = output_tokens {
            let mut final_tokens = tokens;
            if let Some(vary) = output_vary {
                let mut rng = rand::thread_rng();
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

fn format_latency(latency: Option<Duration>) -> String {
    match latency {
        Some(value) => format!("{:.2}", value.as_secs_f64() * 1000.0),
        None => "n/a".to_string(),
    }
}
