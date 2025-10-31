use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde_json::Value;
use tokio::fs;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

use crate::config::{BenchmarkConfig, RunMode};
use crate::report::{BenchmarkReport, FailureRecord};

pub async fn run_benchmark(config: BenchmarkConfig) -> Result<BenchmarkReport> {
    let start = Instant::now();
    let client = Client::builder()
        .timeout(config.request_timeout)
        .build()
        .context("failed to construct HTTP client")?;

    let user_count = config.user_count;
    let planned_total_requests = match &config.mode {
        RunMode::Finite { requests_per_user } => Some((user_count * *requests_per_user) as u64),
        RunMode::LongRunning { .. } => None,
    };

    let config = Arc::new(config);

    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (status_tx, status_rx) = mpsc::unbounded_channel();

    let metrics_status_tx = status_tx.clone();
    let metrics_handle = tokio::spawn(async move {
        collect_metrics(event_rx, planned_total_requests, metrics_status_tx).await
    });

    let tracker_handle = tokio::spawn(track_status(status_rx, start));

    let mut join_set = JoinSet::new();
    for user_id in 0..user_count {
        let client = client.clone();
        let config = Arc::clone(&config);
        let tx = event_tx.clone();
        let status_tx = status_tx.clone();
        join_set.spawn(async move { run_user(user_id, client, config, tx, status_tx).await });
    }
    drop(event_tx);
    drop(status_tx);

    while let Some(join_result) = join_set.join_next().await {
        join_result??;
    }

    let aggregator = metrics_handle.await??;
    let total_duration = start.elapsed();

    tracker_handle
        .await
        .map_err(|err| anyhow!("status tracker task failed: {}", err))?;

    Ok(aggregator.finalize(total_duration))
}

async fn collect_metrics(
    mut rx: mpsc::UnboundedReceiver<WorkerEvent>,
    planned_total_requests: Option<u64>,
    status_tx: mpsc::UnboundedSender<StatusEvent>,
) -> Result<MetricsAggregator> {
    let mut aggregator = MetricsAggregator::new(planned_total_requests);
    let _ = status_tx.send(StatusEvent::Requests {
        successes: aggregator.successful_requests,
        failures: aggregator.failed_requests,
        planned: aggregator.planned_total_requests,
    });
    while let Some(event) = rx.recv().await {
        aggregator.process(event)?;
        let _ = status_tx.send(StatusEvent::Requests {
            successes: aggregator.successful_requests,
            failures: aggregator.failed_requests,
            planned: aggregator.planned_total_requests,
        });
    }
    Ok(aggregator)
}

async fn run_user(
    user_id: usize,
    client: Client,
    config: Arc<BenchmarkConfig>,
    event_tx: mpsc::UnboundedSender<WorkerEvent>,
    status_tx: mpsc::UnboundedSender<StatusEvent>,
) -> Result<()> {
    let mode = config.mode.clone();

    match mode {
        RunMode::Finite { requests_per_user } => {
            for _ in 0..requests_per_user {
                dispatch_request(user_id, &client, &config, &event_tx, &status_tx).await?;
            }
        }
        RunMode::LongRunning { duration } => {
            let deadline = Instant::now() + duration;
            while Instant::now() < deadline {
                dispatch_request(user_id, &client, &config, &event_tx, &status_tx).await?;
            }
        }
    };

    Ok(())
}

async fn dispatch_request(
    user_id: usize,
    client: &Client,
    config: &BenchmarkConfig,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    status_tx: &mpsc::UnboundedSender<StatusEvent>,
) -> Result<()> {
    let request_body = config.request_body_for(user_id)?.clone();
    match single_attempt(client, config, &request_body).await {
        Ok(stats) => {
            event_tx
                .send(WorkerEvent::Success {
                    _user_id: user_id,
                    prompt_tokens: stats.prompt_tokens,
                    completion_tokens: stats.completion_tokens,
                    latency: stats.latency,
                })
                .map_err(|_| anyhow!("metrics channel closed before success event"))?;
            let _ = status_tx.send(StatusEvent::Tokens(stats.completion_tokens));
        }
        Err(err) => {
            let message = err.to_string();
            println!("Runner {} error: {}", user_id, message);
            event_tx
                .send(WorkerEvent::Failure {
                    user_id,
                    error: message,
                })
                .map_err(|_| anyhow!("metrics channel closed before failure event"))?;
        }
    }

    Ok(())
}

async fn single_attempt(
    client: &Client,
    config: &BenchmarkConfig,
    body: &Value,
) -> Result<RequestStats> {
    let start = Instant::now();
    let mut request = client.post(config.endpoint.clone());
    for (name, value) in config.headers.iter() {
        request = request.header(name, value);
    }
    let response = request.json(body).send().await?;
    let status = response.status();
    let bytes = response.bytes().await?;
    let log_path = log_response_bytes(&bytes).await?;

    if !status.is_success() {
        let snippet = String::from_utf8_lossy(&bytes);
        return Err(anyhow!(
            "request failed ({}) {} (logged to {})",
            status,
            snippet,
            log_path.display()
        ));
    }

    let payload: Value = serde_json::from_slice(&bytes)?;
    let (prompt_tokens, completion_tokens) = extract_usage(&payload)?;
    let latency = start.elapsed();

    Ok(RequestStats {
        prompt_tokens,
        completion_tokens,
        latency,
    })
}

fn extract_usage(payload: &Value) -> Result<(u64, u64)> {
    let usage = payload
        .get("usage")
        .ok_or_else(|| anyhow!("response missing usage field"))?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| anyhow!("usage.prompt_tokens missing or not an integer"))?;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    Ok((prompt_tokens, completion_tokens))
}

#[derive(Debug)]
struct RequestStats {
    prompt_tokens: u64,
    completion_tokens: u64,
    latency: Duration,
}

static LOG_SEQUENCE: AtomicU64 = AtomicU64::new(0);

async fn log_response_bytes(bytes: &[u8]) -> Result<PathBuf> {
    let current_dir = env::current_dir().context("failed to resolve current directory")?;
    let logs_dir = current_dir.join("logs");
    fs::create_dir_all(&logs_dir)
        .await
        .with_context(|| format!("failed to create logs directory at {}", logs_dir.display()))?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock set before UNIX_EPOCH")?
        .as_millis();
    let sequence = LOG_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let filename = format!("response-{}-{}.json", timestamp, sequence);
    let path = logs_dir.join(filename);

    fs::write(&path, bytes)
        .await
        .with_context(|| format!("failed to write response log {}", path.display()))?;

    Ok(path)
}

struct MetricsAggregator {
    total_prompt_tokens: u64,
    total_completion_tokens: u64,
    successful_requests: u64,
    failed_requests: u64,
    failures: Vec<FailureRecord>,
    latencies: Vec<Duration>,
    planned_total_requests: Option<u64>,
}

impl MetricsAggregator {
    fn new(planned_total_requests: Option<u64>) -> Self {
        Self {
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            successful_requests: 0,
            failed_requests: 0,
            failures: Vec::new(),
            latencies: Vec::new(),
            planned_total_requests,
        }
    }

    fn process(&mut self, event: WorkerEvent) -> Result<()> {
        match event {
            WorkerEvent::Success {
                _user_id: _,
                prompt_tokens,
                completion_tokens,
                latency,
            } => {
                self.total_prompt_tokens += prompt_tokens;
                self.total_completion_tokens += completion_tokens;
                self.successful_requests += 1;
                self.latencies.push(latency);
            }
            WorkerEvent::Failure { user_id, error } => {
                self.failed_requests += 1;
                self.failures.push(FailureRecord { user_id, error });
            }
        }
        Ok(())
    }

    fn finalize(self, total_duration: Duration) -> BenchmarkReport {
        let total_requests = self.successful_requests + self.failed_requests;
        let duration_secs = total_duration.as_secs_f64();
        let prompt_tokens_per_second = if duration_secs > 0.0 {
            self.total_prompt_tokens as f64 / duration_secs
        } else {
            0.0
        };
        let completion_tokens_per_second = if duration_secs > 0.0 {
            self.total_completion_tokens as f64 / duration_secs
        } else {
            0.0
        };
        let requests_per_second = if duration_secs > 0.0 {
            total_requests as f64 / duration_secs
        } else {
            0.0
        };

        let mut latencies = self.latencies;
        latencies.sort();
        let latency_p50 = percentile(&latencies, 0.50);
        let latency_p90 = percentile(&latencies, 0.90);
        let latency_p99 = percentile(&latencies, 0.99);

        BenchmarkReport {
            total_requests,
            successful_requests: self.successful_requests,
            failed_requests: self.failed_requests,
            total_prompt_tokens: self.total_prompt_tokens,
            total_completion_tokens: self.total_completion_tokens,
            total_duration,
            prompt_tokens_per_second,
            completion_tokens_per_second,
            requests_per_second,
            latency_p50,
            latency_p90,
            latency_p99,
            failures: self.failures,
        }
    }
}

fn percentile(sorted_latencies: &[Duration], quantile: f64) -> Option<Duration> {
    if sorted_latencies.is_empty() {
        return None;
    }

    let clamped = quantile.clamp(0.0, 1.0);
    let idx = ((sorted_latencies.len() - 1) as f64 * clamped).round() as usize;
    sorted_latencies.get(idx).cloned()
}

#[derive(Debug, Clone)]
enum StatusEvent {
    Tokens(u64),
    Requests {
        successes: u64,
        failures: u64,
        planned: Option<u64>,
    },
}

struct StatusSnapshot {
    total_tokens: u64,
    successes: u64,
    failures: u64,
    planned: Option<u64>,
}

async fn track_status(mut updates: mpsc::UnboundedReceiver<StatusEvent>, start: Instant) {
    let mut snapshot = StatusSnapshot {
        total_tokens: 0,
        successes: 0,
        failures: 0,
        planned: None,
    };

    while let Some(event) = updates.recv().await {
        match event {
            StatusEvent::Tokens(delta) => {
                snapshot.total_tokens = snapshot.total_tokens.saturating_add(delta);
            }
            StatusEvent::Requests {
                successes,
                failures,
                planned,
            } => {
                snapshot.successes = successes;
                snapshot.failures = failures;
                snapshot.planned = planned;
            }
        }

        render_status(&snapshot, start, true);
    }

    render_status(&snapshot, start, false);
}

fn render_status(snapshot: &StatusSnapshot, start: Instant, stay: bool) {
    let elapsed = start.elapsed().as_secs_f64();
    let throughput = if elapsed > 0.0 {
        snapshot.total_tokens as f64 / elapsed
    } else {
        0.0
    };
    let completed = snapshot.successes + snapshot.failures;
    let planned_text = snapshot
        .planned
        .map(|total| format!(" / {}", total))
        .unwrap_or_default();

    let throughput_line = format!("Throughput: {:.2} tok/s", throughput);
    let tokens_line = format!("Completion tokens: {}", snapshot.total_tokens);
    let requests_line = format!("Requests: {}{}", completed, planned_text);
    let failures_line = format!("Failures: {}", snapshot.failures);

    print!(
        "\r\x1b[2K{}\n\x1b[2K{}\n\x1b[2K{}\n\x1b[2K{}\n",
        throughput_line, tokens_line, requests_line, failures_line
    );

    if stay {
        print!("\x1b[4A\r");
    }

    let _ = io::stdout().flush();
}

#[derive(Debug)]
enum WorkerEvent {
    Success {
        _user_id: usize,
        prompt_tokens: u64,
        completion_tokens: u64,
        latency: Duration,
    },
    Failure {
        user_id: usize,
        error: String,
    },
}
