use std::collections::HashMap;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use reqwest::header::{HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    run_benchmark, run_from_argv, BenchmarkConfig, BenchmarkReport, DistMode, FailureRecord,
    GenerateOptions, RequestEntry, RunMode,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PyRequestEntry {
    body: Value,
    line_idx: usize,
    input_tokens: usize,
}

impl From<RequestEntry> for PyRequestEntry {
    fn from(value: RequestEntry) -> Self {
        Self {
            body: value.body,
            line_idx: value.line_idx,
            input_tokens: value.input_tokens,
        }
    }
}

impl From<PyRequestEntry> for RequestEntry {
    fn from(value: PyRequestEntry) -> Self {
        Self {
            body: value.body,
            line_idx: value.line_idx,
            input_tokens: value.input_tokens,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PyDistMode {
    Fixed,
    #[serde(alias = "lognormal")]
    LogNormal,
}

impl Default for PyDistMode {
    fn default() -> Self {
        Self::Fixed
    }
}

impl From<PyDistMode> for DistMode {
    fn from(value: PyDistMode) -> Self {
        match value {
            PyDistMode::Fixed => DistMode::Fixed,
            PyDistMode::LogNormal => DistMode::LogNormal,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct PyGenerateOptions {
    count: usize,
    #[serde(default)]
    prefix_overlap: f64,
    #[serde(default)]
    target_tokens: Option<usize>,
    #[serde(default)]
    token_tolerance: Option<usize>,
    tokenizer_model: String,
    #[serde(default)]
    dist_mode: PyDistMode,
    #[serde(default)]
    dist_median: Option<f64>,
    #[serde(default = "default_dist_sigma")]
    dist_sigma: f64,
    #[serde(default)]
    dist_max: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

fn default_dist_sigma() -> f64 {
    0.5
}

impl From<PyGenerateOptions> for GenerateOptions {
    fn from(value: PyGenerateOptions) -> Self {
        Self {
            count: value.count,
            prefix_overlap: value.prefix_overlap,
            target_tokens: value.target_tokens,
            token_tolerance: value.token_tolerance,
            tokenizer_model: value.tokenizer_model,
            dist_mode: value.dist_mode.into(),
            dist_median: value.dist_median,
            dist_sigma: value.dist_sigma,
            dist_max: value.dist_max,
            seed: value.seed,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PyRunMode {
    Finite { requests_per_user: usize },
    LongRunning { duration_secs: f64 },
}

impl TryFrom<PyRunMode> for RunMode {
    type Error = anyhow::Error;

    fn try_from(value: PyRunMode) -> Result<Self> {
        match value {
            PyRunMode::Finite { requests_per_user } => {
                if requests_per_user == 0 {
                    return Err(anyhow!("requests_per_user must be greater than zero"));
                }
                Ok(RunMode::Finite { requests_per_user })
            }
            PyRunMode::LongRunning { duration_secs } => {
                if duration_secs <= 0.0 {
                    return Err(anyhow!("duration_secs must be greater than zero"));
                }
                Ok(RunMode::LongRunning {
                    duration: Duration::from_secs_f64(duration_secs),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct PyOutputLognorm {
    mu: f64,
    sigma: f64,
    #[serde(default)]
    max: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct PyBenchmarkConfig {
    endpoint: String,
    #[serde(default)]
    api_key: Option<String>,
    user_count: usize,
    mode: PyRunMode,
    request_body: PyRequestEntry,
    #[serde(default)]
    requests: Vec<PyRequestEntry>,
    #[serde(default)]
    request_timeout_secs: Option<f64>,
    #[serde(default)]
    max_retries: Option<usize>,
    #[serde(default)]
    retry_delay_ms: Option<u64>,
    #[serde(default)]
    headers: HashMap<String, String>,
    #[serde(default)]
    verbose: bool,
    #[serde(default)]
    output_lognorm: Option<PyOutputLognorm>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    dry_run: bool,
}

fn to_benchmark_config(value: PyBenchmarkConfig) -> Result<BenchmarkConfig> {
    let mode: RunMode = value.mode.try_into()?;
    let request_body: RequestEntry = value.request_body.into();

    let mut config = BenchmarkConfig::try_new(
        value.endpoint,
        value.api_key,
        value.user_count,
        mode,
        request_body,
    )?;

    if let Some(request_timeout_secs) = value.request_timeout_secs {
        if request_timeout_secs <= 0.0 {
            return Err(anyhow!("request_timeout_secs must be greater than zero"));
        }
        config = config.with_request_timeout(Duration::from_secs_f64(request_timeout_secs));
    }

    if let Some(max_retries) = value.max_retries {
        config = config.with_retry(
            max_retries,
            Duration::from_millis(value.retry_delay_ms.unwrap_or(250)),
        );
    } else if let Some(retry_delay_ms) = value.retry_delay_ms {
        let max_retries = config.max_retries;
        config = config.with_retry(max_retries, Duration::from_millis(retry_delay_ms));
    }

    if value.verbose {
        config = config.with_verbose(true);
    }

    if let Some(output_lognorm) = value.output_lognorm {
        config =
            config.with_output_lognorm(output_lognorm.mu, output_lognorm.sigma, output_lognorm.max);
    }

    if let Some(seed) = value.seed {
        config = config.with_seed(seed);
    }

    if value.dry_run {
        config = config.with_dry_run(true);
    }

    for (name, value) in value.headers {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .with_context(|| format!("invalid header name: {}", name))?;
        let header_value = HeaderValue::from_str(&value)
            .with_context(|| format!("invalid value for header {}", name))?;
        config = config.add_header(header_name, header_value);
    }

    if !value.requests.is_empty() {
        let requests = value.requests.into_iter().map(RequestEntry::from).collect();
        config = config.with_request_list(requests)?;
    }

    Ok(config)
}

#[derive(Debug, Clone, Serialize)]
struct PyFailureRecord {
    user_id: usize,
    error: String,
}

impl From<FailureRecord> for PyFailureRecord {
    fn from(value: FailureRecord) -> Self {
        Self {
            user_id: value.user_id,
            error: value.error,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct PyBenchmarkReport {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    total_prompt_tokens: u64,
    total_completion_tokens: u64,
    total_duration_secs: f64,
    prompt_tokens_per_second: f64,
    completion_tokens_per_second: f64,
    requests_per_second: f64,
    total_token_throughput: f64,
    latency_p50_ms: Option<f64>,
    latency_p90_ms: Option<f64>,
    latency_p99_ms: Option<f64>,
    failures: Vec<PyFailureRecord>,
}

impl From<BenchmarkReport> for PyBenchmarkReport {
    fn from(value: BenchmarkReport) -> Self {
        Self {
            total_requests: value.total_requests,
            successful_requests: value.successful_requests,
            failed_requests: value.failed_requests,
            total_prompt_tokens: value.total_prompt_tokens,
            total_completion_tokens: value.total_completion_tokens,
            total_duration_secs: value.total_duration.as_secs_f64(),
            prompt_tokens_per_second: value.prompt_tokens_per_second,
            completion_tokens_per_second: value.completion_tokens_per_second,
            requests_per_second: value.requests_per_second,
            total_token_throughput: value.total_token_throughput(),
            latency_p50_ms: value
                .latency_p50
                .map(|latency| latency.as_secs_f64() * 1000.0),
            latency_p90_ms: value
                .latency_p90
                .map(|latency| latency.as_secs_f64() * 1000.0),
            latency_p99_ms: value
                .latency_p99
                .map(|latency| latency.as_secs_f64() * 1000.0),
            failures: value
                .failures
                .into_iter()
                .map(PyFailureRecord::from)
                .collect(),
        }
    }
}

fn to_py_error(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

#[pyfunction]
fn generate_requests_json(options_json: &str, model: &str) -> PyResult<String> {
    let options: PyGenerateOptions =
        serde_json::from_str(options_json).map_err(|err| to_py_error(anyhow!(err)))?;
    let requests =
        crate::generate_requests(&GenerateOptions::from(options), model).map_err(to_py_error)?;
    let py_requests: Vec<PyRequestEntry> = requests.into_iter().map(PyRequestEntry::from).collect();
    serde_json::to_string(&py_requests).map_err(|err| to_py_error(anyhow!(err)))
}

#[pyfunction]
fn run_benchmark_json(py: Python<'_>, config_json: &str) -> PyResult<String> {
    let config: PyBenchmarkConfig =
        serde_json::from_str(config_json).map_err(|err| to_py_error(anyhow!(err)))?;
    let config = to_benchmark_config(config).map_err(to_py_error)?;

    let result = py.allow_threads(move || -> Result<String> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(anyhow::Error::from)?;
        let report = runtime.block_on(run_benchmark(config))?;
        serde_json::to_string(&PyBenchmarkReport::from(report)).map_err(anyhow::Error::from)
    });
    result.map_err(to_py_error)
}

#[pyfunction]
fn run_cli(py: Python<'_>, argv: Vec<String>) -> PyResult<()> {
    py.allow_threads(move || run_from_argv(argv))
        .map_err(to_py_error)
}

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(generate_requests_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_benchmark_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_cli, module)?)?;
    Ok(())
}
