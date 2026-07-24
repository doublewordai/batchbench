use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;

use crate::{run_agent_benchmark, AgentBenchmarkReport, AgentLoopConfig, SampleSpec};

const DEFAULT_MODEL: &str = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8";

#[derive(Parser, Debug)]
#[command(
    name = "batchbench-agent",
    about = "Benchmark concurrent stateful agent loops and server-side prefix caching"
)]
struct Args {
    /// Number of independent agent loops to run concurrently
    #[arg(long, default_value_t = 1)]
    agents: usize,

    /// OpenAI-style model identifier
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Tokenizer model used to synthesize prompts and environment responses (defaults to --model)
    #[arg(long)]
    tokenizer_model: Option<String>,

    /// Host to target (e.g. https://api.openai.com)
    #[arg(long, default_value = "https://api.openai.com")]
    host: String,

    /// Endpoint path or full URL
    #[arg(long, default_value = "/v1/chat/completions")]
    endpoint: String,

    /// Fixed initial user-prompt length (default: 128)
    #[arg(long, alias = "initial-prompt-tokens")]
    input_tokens: Option<usize>,

    /// Initial prompt log-normal mu (mean of the underlying normal)
    #[arg(long, alias = "initial-prompt-lognorm-mu")]
    input_lognorm_mu: Option<f64>,

    /// Initial prompt log-normal median (preferred over mu)
    #[arg(long, alias = "initial-prompt-lognorm-median")]
    input_lognorm_median: Option<f64>,

    /// Initial prompt log-normal sigma
    #[arg(long, alias = "initial-prompt-lognorm-sigma")]
    input_lognorm_sigma: Option<f64>,

    /// Maximum initial prompt tokens for log-normal sampling
    #[arg(long, alias = "initial-prompt-lognorm-max")]
    input_lognorm_max: Option<usize>,

    /// Fixed model response length (default: 64)
    #[arg(long, alias = "model-response-tokens")]
    output_tokens: Option<usize>,

    /// Model response log-normal mu (mean of the underlying normal)
    #[arg(long, alias = "model-response-lognorm-mu")]
    output_lognorm_mu: Option<f64>,

    /// Model response log-normal median (preferred over mu)
    #[arg(long, alias = "model-response-lognorm-median")]
    output_lognorm_median: Option<f64>,

    /// Model response log-normal sigma
    #[arg(long, alias = "model-response-lognorm-sigma")]
    output_lognorm_sigma: Option<f64>,

    /// Maximum model response tokens for log-normal sampling
    #[arg(long, alias = "model-response-lognorm-max")]
    output_lognorm_max: Option<usize>,

    /// Fixed synthetic environment response length (default: 64)
    #[arg(long, alias = "environment-response-tokens")]
    environment_tokens: Option<usize>,

    /// Environment response log-normal mu (mean of the underlying normal)
    #[arg(long, alias = "environment-response-lognorm-mu")]
    environment_lognorm_mu: Option<f64>,

    /// Environment response log-normal median (preferred over mu)
    #[arg(long, alias = "environment-response-lognorm-median")]
    environment_lognorm_median: Option<f64>,

    /// Environment response log-normal sigma
    #[arg(long, alias = "environment-response-lognorm-sigma")]
    environment_lognorm_sigma: Option<f64>,

    /// Maximum environment response tokens for log-normal sampling
    #[arg(long, alias = "environment-response-lognorm-max")]
    environment_lognorm_max: Option<usize>,

    /// Fixed number of tool invocations per agent (default: 4)
    #[arg(long)]
    tool_invocations: Option<usize>,

    /// Tool-invocation-count log-normal mu (mean of the underlying normal)
    #[arg(long)]
    tool_invocations_lognorm_mu: Option<f64>,

    /// Tool-invocation-count log-normal median (preferred over mu)
    #[arg(long)]
    tool_invocations_lognorm_median: Option<f64>,

    /// Tool-invocation-count log-normal sigma
    #[arg(long)]
    tool_invocations_lognorm_sigma: Option<f64>,

    /// Maximum tool invocations for log-normal sampling
    #[arg(long)]
    tool_invocations_lognorm_max: Option<usize>,

    /// Fixed simulated tool-call latency in milliseconds (default: 0)
    #[arg(long, alias = "tool-latency-ms")]
    tool_call_latency_ms: Option<usize>,

    /// Tool-call latency log-normal mu for values in milliseconds
    #[arg(long)]
    tool_call_latency_lognorm_mu: Option<f64>,

    /// Tool-call latency log-normal median in milliseconds (preferred over mu)
    #[arg(
        long = "tool-call-latency-lognorm-median-ms",
        alias = "tool-call-latency-lognorm-median"
    )]
    tool_call_latency_lognorm_median_ms: Option<f64>,

    /// Tool-call latency log-normal sigma
    #[arg(long)]
    tool_call_latency_lognorm_sigma: Option<f64>,

    /// Maximum sampled tool-call latency in milliseconds
    #[arg(
        long = "tool-call-latency-lognorm-max-ms",
        alias = "tool-call-latency-lognorm-max"
    )]
    tool_call_latency_lognorm_max_ms: Option<usize>,

    /// API key; when omitted, read --api-key-env
    #[arg(long)]
    api_key: Option<String>,

    /// Environment variable containing the API key
    #[arg(long, default_value = "OPENAI_API_KEY")]
    api_key_env: String,

    /// Per-request timeout in seconds
    #[arg(long, default_value_t = 60)]
    request_timeout_secs: u64,

    /// Maximum retries after the first request attempt
    #[arg(long, default_value_t = 2)]
    max_retries: usize,

    /// Base retry delay in milliseconds
    #[arg(long, default_value_t = 250)]
    retry_delay_ms: u64,

    /// Use SGLang min_new_tokens/max_new_tokens output constraints
    #[arg(long)]
    sglang: bool,

    /// Print truncated request and response payloads
    #[arg(long, short)]
    verbose: bool,

    /// Sample and display the agent plans without sending HTTP requests
    #[arg(long)]
    dry_run: bool,

    /// Random seed for reproducible independent samples
    #[arg(long)]
    seed: Option<u64>,

    /// Optional path to write a CSV report
    #[arg(long)]
    results_csv: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct CsvResult {
    timestamp: String,
    model: String,
    agents: usize,
    completed_agents: usize,
    planned_tool_invocations: u64,
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    estimated_cached_input_tokens: u64,
    total_tool_call_latency_seconds: f64,
    total_duration_seconds: f64,
    requests_per_second: f64,
    input_tokens_per_second: f64,
    output_tokens_per_second: f64,
    latency_p50_ms: Option<f64>,
    latency_p90_ms: Option<f64>,
    latency_p99_ms: Option<f64>,
    host: String,
    endpoint: String,
    sglang: bool,
    seed: Option<u64>,
    tool_call_latency_ms: Option<usize>,
    tool_call_latency_lognorm_mu: Option<f64>,
    tool_call_latency_lognorm_median_ms: Option<f64>,
    tool_call_latency_lognorm_sigma: Option<f64>,
    tool_call_latency_lognorm_max_ms: Option<usize>,
}

enum ParsedArgs {
    Ready(Box<Args>),
    Displayed,
}

pub fn run_from_env() -> Result<()> {
    match parse_args(std::env::args_os())? {
        ParsedArgs::Ready(args) => run_with_runtime(*args),
        ParsedArgs::Displayed => Ok(()),
    }
}

pub fn run_from_argv(argv: Vec<String>) -> Result<()> {
    let mut full_argv = Vec::with_capacity(argv.len() + 1);
    full_argv.push("batchbench-agent".to_string());
    full_argv.extend(argv);
    match parse_args(full_argv)? {
        ParsedArgs::Ready(args) => run_with_runtime(*args),
        ParsedArgs::Displayed => Ok(()),
    }
}

fn parse_args<I, T>(argv: I) -> Result<ParsedArgs>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Args::try_parse_from(argv) {
        Ok(args) => Ok(ParsedArgs::Ready(Box::new(args))),
        Err(error) => match error.kind() {
            clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
                error
                    .print()
                    .map_err(|print_error| anyhow!(print_error.to_string()))?;
                Ok(ParsedArgs::Displayed)
            }
            _ => Err(anyhow!(error.to_string())),
        },
    }
}

fn run_with_runtime(args: Args) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime for agent CLI")?;
    runtime.block_on(run(args))
}

async fn run(args: Args) -> Result<()> {
    if args.agents == 0 {
        return Err(anyhow!("agents must be greater than zero"));
    }
    if args.request_timeout_secs == 0 {
        return Err(anyhow!("request-timeout-secs must be greater than zero"));
    }

    let input_tokens = resolve_sample_spec(
        "input",
        args.input_tokens,
        args.input_lognorm_mu,
        args.input_lognorm_median,
        args.input_lognorm_sigma,
        args.input_lognorm_max,
        128,
    )?;
    let output_tokens = resolve_sample_spec(
        "output",
        args.output_tokens,
        args.output_lognorm_mu,
        args.output_lognorm_median,
        args.output_lognorm_sigma,
        args.output_lognorm_max,
        64,
    )?;
    let environment_tokens = resolve_sample_spec(
        "environment",
        args.environment_tokens,
        args.environment_lognorm_mu,
        args.environment_lognorm_median,
        args.environment_lognorm_sigma,
        args.environment_lognorm_max,
        64,
    )?;
    let tool_invocations = resolve_sample_spec(
        "tool-invocations",
        args.tool_invocations,
        args.tool_invocations_lognorm_mu,
        args.tool_invocations_lognorm_median,
        args.tool_invocations_lognorm_sigma,
        args.tool_invocations_lognorm_max,
        4,
    )?;
    let tool_call_latency_ms = resolve_optional_latency_spec(
        args.tool_call_latency_ms,
        args.tool_call_latency_lognorm_mu,
        args.tool_call_latency_lognorm_median_ms,
        args.tool_call_latency_lognorm_sigma,
        args.tool_call_latency_lognorm_max_ms,
    )?;

    let endpoint = resolve_endpoint(&args.host, &args.endpoint);
    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var(&args.api_key_env).ok());
    let tokenizer_model = args
        .tokenizer_model
        .clone()
        .unwrap_or_else(|| args.model.clone());

    println!("=== Agent Loop Benchmark Configuration ===");
    println!("Endpoint: {}", endpoint);
    println!("Model: {}", args.model);
    println!("Tokenizer: {}", tokenizer_model);
    println!("Parallel agents: {}", args.agents);
    println!("Initial prompt: {}", describe_spec(&input_tokens));
    println!("Model response: {}", describe_spec(&output_tokens));
    println!(
        "Environment response: {}",
        describe_spec(&environment_tokens)
    );
    println!(
        "Tool invocations per agent: {}",
        describe_spec(&tool_invocations)
    );
    println!(
        "Tool-call latency: {}",
        describe_latency_spec(tool_call_latency_ms.as_ref())
    );
    println!(
        "Output token params: {}",
        if args.sglang {
            "sglang (min_new_tokens/max_new_tokens)"
        } else {
            "default (min_tokens/max_tokens)"
        }
    );
    println!("==========================================\n");

    let mut config = AgentLoopConfig::try_new(
        &endpoint,
        api_key,
        args.model.clone(),
        args.agents,
        input_tokens,
        output_tokens,
        environment_tokens,
        tool_invocations,
    )?
    .with_tokenizer_model(tokenizer_model)
    .with_request_timeout(Duration::from_secs(args.request_timeout_secs))
    .with_retry(args.max_retries, Duration::from_millis(args.retry_delay_ms))
    .with_sglang(args.sglang)
    .with_verbose(args.verbose)
    .with_dry_run(args.dry_run);
    if let Some(tool_call_latency_ms) = tool_call_latency_ms {
        config = config.with_tool_call_latency_ms(tool_call_latency_ms)?;
    }
    if let Some(seed) = args.seed {
        config = config.with_seed(seed);
    }

    let start_time = chrono::Utc::now();
    let report = run_agent_benchmark(config).await?;
    print_summary(&report);

    if let Some(csv_path) = args.results_csv.as_deref() {
        let record = CsvResult {
            timestamp: start_time.to_rfc3339(),
            model: args.model,
            agents: args.agents,
            completed_agents: report.completed_agents,
            planned_tool_invocations: report.planned_tool_invocations,
            total_requests: report.total_requests,
            successful_requests: report.successful_requests,
            failed_requests: report.failed_requests,
            total_input_tokens: report.total_input_tokens,
            total_output_tokens: report.total_output_tokens,
            estimated_cached_input_tokens: report.estimated_cached_input_tokens,
            total_tool_call_latency_seconds: report.total_tool_call_latency.as_secs_f64(),
            total_duration_seconds: report.total_duration.as_secs_f64(),
            requests_per_second: report.requests_per_second,
            input_tokens_per_second: report.input_tokens_per_second,
            output_tokens_per_second: report.output_tokens_per_second,
            latency_p50_ms: milliseconds(report.latency_p50),
            latency_p90_ms: milliseconds(report.latency_p90),
            latency_p99_ms: milliseconds(report.latency_p99),
            host: args.host,
            endpoint,
            sglang: args.sglang,
            seed: args.seed,
            tool_call_latency_ms: args.tool_call_latency_ms,
            tool_call_latency_lognorm_mu: args.tool_call_latency_lognorm_mu,
            tool_call_latency_lognorm_median_ms: args.tool_call_latency_lognorm_median_ms,
            tool_call_latency_lognorm_sigma: args.tool_call_latency_lognorm_sigma,
            tool_call_latency_lognorm_max_ms: args.tool_call_latency_lognorm_max_ms,
        };
        write_results_csv(csv_path, &record)?;
        println!("Wrote agent benchmark summary to {}", csv_path.display());
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resolve_sample_spec(
    label: &str,
    fixed: Option<usize>,
    mu: Option<f64>,
    median: Option<f64>,
    sigma: Option<f64>,
    max: Option<usize>,
    default_fixed: usize,
) -> Result<SampleSpec> {
    let lognormal_requested = mu.is_some() || median.is_some() || sigma.is_some() || max.is_some();
    if fixed.is_some() && lognormal_requested {
        let fixed_flag = if label == "tool-invocations" {
            "--tool-invocations".to_string()
        } else {
            format!("--{}-tokens", label)
        };
        return Err(anyhow!(
            "{} cannot be combined with --{}-lognorm-* flags",
            fixed_flag,
            label,
        ));
    }
    if mu.is_some() && median.is_some() {
        return Err(anyhow!(
            "provide either --{}-lognorm-mu or --{}-lognorm-median, not both",
            label,
            label
        ));
    }

    if let Some(value) = fixed {
        return SampleSpec::fixed(value).with_context(|| format!("invalid fixed {} value", label));
    }
    if !lognormal_requested {
        return SampleSpec::fixed(default_fixed);
    }

    let sigma = sigma.ok_or_else(|| {
        anyhow!(
            "--{}-lognorm-sigma is required for log-normal sampling",
            label
        )
    })?;
    match (mu, median) {
        (Some(mu), None) => SampleSpec::log_normal(mu, sigma, max),
        (None, Some(median)) => SampleSpec::log_normal_from_median(median, sigma, max),
        _ => Err(anyhow!(
            "--{}-lognorm-median or --{}-lognorm-mu is required for log-normal sampling",
            label,
            label
        )),
    }
}

fn resolve_optional_latency_spec(
    fixed_ms: Option<usize>,
    mu: Option<f64>,
    median_ms: Option<f64>,
    sigma: Option<f64>,
    max_ms: Option<usize>,
) -> Result<Option<SampleSpec>> {
    let lognormal_requested =
        mu.is_some() || median_ms.is_some() || sigma.is_some() || max_ms.is_some();
    if fixed_ms.is_some() && lognormal_requested {
        return Err(anyhow!(
            "--tool-call-latency-ms cannot be combined with --tool-call-latency-lognorm-* flags"
        ));
    }
    if mu.is_some() && median_ms.is_some() {
        return Err(anyhow!(
            "provide either --tool-call-latency-lognorm-mu or --tool-call-latency-lognorm-median-ms, not both"
        ));
    }

    if let Some(fixed_ms) = fixed_ms {
        if fixed_ms == 0 {
            return Ok(None);
        }
        return SampleSpec::fixed(fixed_ms).map(Some);
    }
    if !lognormal_requested {
        return Ok(None);
    }

    let sigma = sigma.ok_or_else(|| {
        anyhow!("--tool-call-latency-lognorm-sigma is required for log-normal latency sampling")
    })?;
    let spec = match (mu, median_ms) {
        (Some(mu), None) => SampleSpec::log_normal(mu, sigma, max_ms),
        (None, Some(median_ms)) => {
            SampleSpec::log_normal_from_median(median_ms, sigma, max_ms)
        }
        _ => {
            return Err(anyhow!(
                "--tool-call-latency-lognorm-median-ms or --tool-call-latency-lognorm-mu is required for log-normal latency sampling"
            ))
        }
    }?;
    Ok(Some(spec))
}

fn describe_spec(spec: &SampleSpec) -> String {
    match spec {
        SampleSpec::Fixed(value) => format!("{} (fixed)", value),
        SampleSpec::LogNormal { mu, sigma, max } => {
            let max_text = max
                .map(|value| format!(", max={}", value))
                .unwrap_or_default();
            format!(
                "lognormal(median≈{:.0}, σ={}{}; μ={:.3})",
                mu.exp(),
                sigma,
                max_text,
                mu
            )
        }
    }
}

fn describe_latency_spec(spec: Option<&SampleSpec>) -> String {
    match spec {
        None => "0 ms (fixed)".to_string(),
        Some(SampleSpec::Fixed(value)) => format!("{} ms (fixed)", value),
        Some(SampleSpec::LogNormal { mu, sigma, max }) => {
            let max_text = max
                .map(|value| format!(", max={} ms", value))
                .unwrap_or_default();
            format!(
                "lognormal(median≈{:.0} ms, σ={}{}; μ={:.3})",
                mu.exp(),
                sigma,
                max_text,
                mu
            )
        }
    }
}

fn resolve_endpoint(host: &str, endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        return endpoint.to_string();
    }
    let host = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("https://{}", host.trim_end_matches('/'))
    };
    format!("{}/{}", host, endpoint.trim_start_matches('/'))
}

fn print_summary(report: &AgentBenchmarkReport) {
    println!(
        "Agents: {} completed / {} total",
        report.completed_agents, report.total_agents
    );
    println!(
        "Tool invocations: {} planned; requests {} (success {}, failure {})",
        report.planned_tool_invocations,
        report.total_requests,
        report.successful_requests,
        report.failed_requests
    );
    println!("Total input tokens sent: {}", report.total_input_tokens);
    println!(
        "Total output tokens generated: {}",
        report.total_output_tokens
    );
    println!(
        "Estimated cached input tokens (perfect server-side caching): {}",
        report.estimated_cached_input_tokens
    );
    println!(
        "Total simulated tool-call latency: {:.2}s",
        report.total_tool_call_latency.as_secs_f64()
    );
    println!(
        "Total duration: {:.2}s; throughput: input {:.2} tok/s, output {:.2} tok/s, requests {:.2} req/s",
        report.total_duration.as_secs_f64(),
        report.input_tokens_per_second,
        report.output_tokens_per_second,
        report.requests_per_second
    );
    println!(
        "Latency (ms): p50={} p90={} p99={}",
        format_latency(report.latency_p50),
        format_latency(report.latency_p90),
        format_latency(report.latency_p99)
    );
    for failure in &report.failures {
        println!(
            "Failure: agent {} invocation {}: {}",
            failure.agent_id, failure.invocation, failure.error
        );
    }
}

fn milliseconds(duration: Option<Duration>) -> Option<f64> {
    duration.map(|value| value.as_secs_f64() * 1000.0)
}

fn format_latency(duration: Option<Duration>) -> String {
    milliseconds(duration)
        .map(|value| format!("{:.2}", value))
        .unwrap_or_else(|| "n/a".to_string())
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
        .with_context(|| format!("failed to serialize {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_and_lognormal_options_are_mutually_exclusive() {
        let error = resolve_sample_spec("output", Some(8), None, Some(16.0), Some(0.5), None, 64)
            .unwrap_err();
        assert!(error.to_string().contains("cannot be combined"));
    }

    #[test]
    fn median_is_converted_to_mu() {
        let spec = resolve_sample_spec(
            "environment",
            None,
            None,
            Some(32.0),
            Some(0.5),
            Some(128),
            64,
        )
        .unwrap();
        match spec {
            SampleSpec::LogNormal { mu, sigma, max } => {
                assert!((mu - 32.0f64.ln()).abs() < f64::EPSILON);
                assert_eq!(sigma, 0.5);
                assert_eq!(max, Some(128));
            }
            SampleSpec::Fixed(_) => panic!("expected log-normal spec"),
        }
    }

    #[test]
    fn defaults_are_fixed() {
        let spec = resolve_sample_spec("input", None, None, None, None, None, 128).unwrap();
        assert!(matches!(spec, SampleSpec::Fixed(128)));
    }

    #[test]
    fn tool_call_latency_defaults_to_zero_and_accepts_fixed_ms() {
        let default = resolve_optional_latency_spec(None, None, None, None, None).unwrap();
        assert!(default.is_none());

        let zero = resolve_optional_latency_spec(Some(0), None, None, None, None).unwrap();
        assert!(zero.is_none());

        let fixed = resolve_optional_latency_spec(Some(125), None, None, None, None).unwrap();
        assert!(matches!(fixed, Some(SampleSpec::Fixed(125))));
    }

    #[test]
    fn tool_call_latency_accepts_log_normal_milliseconds() {
        let spec =
            resolve_optional_latency_spec(None, None, Some(250.0), Some(0.4), Some(1_000)).unwrap();
        match spec {
            Some(SampleSpec::LogNormal { mu, sigma, max }) => {
                assert!((mu - 250.0f64.ln()).abs() < f64::EPSILON);
                assert_eq!(sigma, 0.4);
                assert_eq!(max, Some(1_000));
            }
            _ => panic!("expected log-normal latency spec"),
        }
    }

    #[test]
    fn fixed_and_lognormal_tool_call_latency_are_mutually_exclusive() {
        let error = resolve_optional_latency_spec(Some(100), None, Some(200.0), Some(0.5), None)
            .unwrap_err();
        assert!(error.to_string().contains("cannot be combined"));
    }
}
