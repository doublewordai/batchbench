use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde_json::Value;
use tokio::fs;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

use crate::config::{BenchmarkConfig, RequestEntry, RunMode};
use crate::report::{BenchmarkReport, FailureRecord};
use std::cmp;

#[derive(Debug, Clone)]
struct DryRunRecord {
    user_id: usize,
    request_id: usize,
    input_tokens: usize,
    tokens: Option<usize>,
}

fn output_token_field_names(use_sglang: bool) -> (&'static str, &'static str) {
    if use_sglang {
        ("max_new_tokens", "min_new_tokens")
    } else {
        ("max_tokens", "min_tokens")
    }
}

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

    let mut interrupted = false;
    let mut ctrl_c = pin!(tokio::signal::ctrl_c());
    while !join_set.is_empty() {
        tokio::select! {
            signal_result = &mut ctrl_c, if !interrupted => {
                match signal_result {
                    Ok(()) => {
                        interrupted = true;
                        eprintln!("\nReceived Ctrl+C, cancelling active requests...");
                        join_set.abort_all();
                    }
                    Err(err) => {
                        eprintln!("\nFailed to listen for Ctrl+C: {}", err);
                    }
                }
            }
            join_result = join_set.join_next() => {
                match join_result {
                    Some(Ok(worker_result)) => {
                        match worker_result {
                            Ok(()) => {}
                            Err(err) => {
                                if interrupted {
                                    eprintln!("worker exited with error during shutdown: {}", err);
                                } else {
                                    return Err(err);
                                }
                            }
                        }
                    }
                    Some(Err(err)) => {
                        if interrupted && err.is_cancelled() {
                            continue;
                        }
                        return Err(anyhow!("worker task failed: {}", err));
                    }
                    None => break,
                }
            }
        }
    }

    let aggregator = metrics_handle.await??;
    let total_duration = start.elapsed();

    tracker_handle
        .await
        .map_err(|err| anyhow!("status tracker task failed: {}", err))?;

    let (report, dry_run_events) = aggregator.finalize(total_duration);

    if config.dry_run && !dry_run_events.is_empty() {
        print_sorted_dry_run_events(&dry_run_events);
    }

    Ok(report)
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
            for request_id in 0..requests_per_user {
                dispatch_request(user_id, request_id, &client, &config, &event_tx, &status_tx)
                    .await?;
            }
        }
        RunMode::LongRunning { duration } => {
            let deadline = Instant::now() + duration;
            let mut request_id = 0usize;
            while Instant::now() < deadline {
                dispatch_request(user_id, request_id, &client, &config, &event_tx, &status_tx)
                    .await?;
                request_id = request_id.wrapping_add(1);
            }
        }
    };

    Ok(())
}

async fn dispatch_request(
    user_id: usize,
    request_id: usize,
    client: &Client,
    config: &BenchmarkConfig,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    status_tx: &mpsc::UnboundedSender<StatusEvent>,
) -> Result<()> {
    // Deterministic mapping: request m from user n uses index m * user_count + n
    let idx = request_id
        .checked_mul(config.user_count)
        .and_then(|v| v.checked_add(user_id))
        .ok_or_else(|| anyhow!("request index overflowed"))?;

    let request_entry: RequestEntry = config.requests.get(idx).cloned().ok_or_else(|| {
        anyhow!(
            "no request entry for user {} request {} (index {})",
            user_id,
            request_id,
            idx
        )
    })?;

    let mut request_body = request_entry.body.clone();

    // If lognormal output sampling is configured, sample and inject tokens
    let mut lognorm_tokens: Option<usize> = None;
    if let Some((mu, sigma, max)) = config.output_lognorm {
        use rand::SeedableRng;
        use rand_distr::{Distribution, LogNormal};

        let log_normal = LogNormal::new(mu, sigma)
            .map_err(|e| anyhow!("failed to create lognormal distribution: {}", e))?;

        let sampled: f64 = if let Some(seed) = config.seed {
            // Use seeded RNG for reproducibility
            // Mix in user_id and request_id to ensure different requests get different samples
            let mixed_seed = seed
                .wrapping_add(user_id as u64)
                .wrapping_add((request_id as u64).wrapping_mul(65537)); // Use prime multiplier for better distribution
            let mut rng = rand::rngs::StdRng::seed_from_u64(mixed_seed);
            log_normal.sample(&mut rng)
        } else {
            let mut rng = rand::thread_rng();
            log_normal.sample(&mut rng)
        };

        let mut tokens = (sampled.round() as usize).max(1); // Ensure at least 1 token

        // Apply max truncation if specified
        if let Some(max_val) = max {
            tokens = tokens.min(max_val);
        }

        if let Some(map) = request_body.as_object_mut() {
            let (max_key, min_key) = output_token_field_names(config.sglang);
            map.insert(max_key.to_string(), serde_json::json!(tokens));
            map.insert(min_key.to_string(), serde_json::json!(tokens));
            lognorm_tokens = Some(tokens);
        }
    }

    if config.dry_run {
        let (max_key, min_key) = output_token_field_names(config.sglang);
        let token_field = request_body
            .get(max_key)
            .or_else(|| request_body.get(min_key))
            .or_else(|| request_body.get("max_tokens"))
            .or_else(|| request_body.get("min_tokens"))
            .or_else(|| request_body.get("max_new_tokens"))
            .or_else(|| request_body.get("min_new_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or(lognorm_tokens);

        event_tx
            .send(WorkerEvent::DryRun {
                user_id,
                request_id,
                input_tokens: request_entry.input_tokens,
                tokens: token_field,
            })
            .map_err(|_| anyhow!("metrics channel closed before dry-run event"))?;

        return Ok(());
    }

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
            let _ = status_tx.send(StatusEvent::Tokens {
                prompt_tokens: stats.prompt_tokens,
                completion_tokens: stats.completion_tokens,
            });
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
    if config.verbose {
        let sanitized_body = sanitize_request_body(body);
        println!("[REQUEST] {}", serde_json::to_string(&sanitized_body)?);
    }

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

    if config.verbose {
        let sanitized_response = sanitize_response(&payload);
        println!("[RESPONSE] {}", serde_json::to_string(&sanitized_response)?);
    }

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

fn sanitize_request_body(body: &Value) -> Value {
    let mut sanitized = body.clone();

    if let Some(obj) = sanitized.as_object_mut() {
        if let Some(messages) = obj.get_mut("messages") {
            if let Some(arr) = messages.as_array_mut() {
                for msg in arr.iter_mut() {
                    if let Some(msg_obj) = msg.as_object_mut() {
                        if let Some(content) = msg_obj.get_mut("content") {
                            if let Some(text) = content.as_str() {
                                let preview = truncate_text(text, 50);
                                *content = Value::String(preview);
                            }
                        }
                    }
                }
            }
        }
    }

    sanitized
}

fn sanitize_response(response: &Value) -> Value {
    let mut sanitized = response.clone();

    if let Some(obj) = sanitized.as_object_mut() {
        if let Some(choices) = obj.get_mut("choices") {
            if let Some(arr) = choices.as_array_mut() {
                for choice in arr.iter_mut() {
                    if let Some(choice_obj) = choice.as_object_mut() {
                        if let Some(message) = choice_obj.get_mut("message") {
                            if let Some(msg_obj) = message.as_object_mut() {
                                if let Some(content) = msg_obj.get_mut("content") {
                                    if let Some(text) = content.as_str() {
                                        let preview = truncate_text(text, 50);
                                        *content = Value::String(preview);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    sanitized
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        let truncated: String = trimmed.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
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
    dry_run_events: Vec<DryRunRecord>,
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
            dry_run_events: Vec::new(),
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
            WorkerEvent::DryRun {
                user_id,
                request_id,
                input_tokens,
                tokens,
            } => {
                self.dry_run_events.push(DryRunRecord {
                    user_id,
                    request_id,
                    input_tokens,
                    tokens,
                });
            }
        }
        Ok(())
    }

    fn finalize(self, total_duration: Duration) -> (BenchmarkReport, Vec<DryRunRecord>) {
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

        let report = BenchmarkReport {
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
        };

        (report, self.dry_run_events)
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
    Tokens {
        prompt_tokens: u64,
        completion_tokens: u64,
    },
    Requests {
        successes: u64,
        failures: u64,
        planned: Option<u64>,
    },
}

struct StatusSnapshot {
    total_prompt_tokens: u64,
    total_completion_tokens: u64,
    successes: u64,
    failures: u64,
    planned: Option<u64>,
}

async fn track_status(mut updates: mpsc::UnboundedReceiver<StatusEvent>, start: Instant) {
    let mut snapshot = StatusSnapshot {
        total_prompt_tokens: 0,
        total_completion_tokens: 0,
        successes: 0,
        failures: 0,
        planned: None,
    };

    while let Some(event) = updates.recv().await {
        match event {
            StatusEvent::Tokens {
                prompt_tokens,
                completion_tokens,
            } => {
                snapshot.total_prompt_tokens =
                    snapshot.total_prompt_tokens.saturating_add(prompt_tokens);
                snapshot.total_completion_tokens = snapshot
                    .total_completion_tokens
                    .saturating_add(completion_tokens);
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
        snapshot.total_completion_tokens as f64 / elapsed
    } else {
        0.0
    };
    let completed = snapshot.successes + snapshot.failures;
    let planned_text = snapshot
        .planned
        .map(|total| format!(" / {}", total))
        .unwrap_or_default();

    let elapsed_line = format!("Elapsed: {:.1}s", elapsed);
    let throughput_line = format!("Throughput: {:.2} tok/s", throughput);
    let prompt_tokens_line = format!("Input tokens: {}", snapshot.total_prompt_tokens);
    let completion_tokens_line = format!("Output tokens: {}", snapshot.total_completion_tokens);
    let requests_line = format!("Requests: {}{}", completed, planned_text);
    let failures_line = format!("Failures: {}", snapshot.failures);

    print!(
        "\r\x1b[2K{}\n\x1b[2K{}\n\x1b[2K{}\n\x1b[2K{}\n\x1b[2K{}\n\x1b[2K{}\n",
        elapsed_line,
        throughput_line,
        prompt_tokens_line,
        completion_tokens_line,
        requests_line,
        failures_line
    );

    if stay {
        print!("\x1b[6A\r");
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
    DryRun {
        user_id: usize,
        request_id: usize,
        input_tokens: usize,
        tokens: Option<usize>,
    },
}

fn print_sorted_dry_run_events(events: &[DryRunRecord]) {
    let mut sorted = events.to_vec();
    sorted.sort_by(|a, b| {
        a.user_id
            .cmp(&b.user_id)
            .then_with(|| a.request_id.cmp(&b.request_id))
    });

    // Output token histogram (only if tokens present)
    let output_tokens: Vec<usize> = sorted.iter().filter_map(|ev| ev.tokens).collect();
    if !output_tokens.is_empty() {
        print_histogram("Output tokens (dry-run)", &output_tokens, 20, 50);
    }

    for ev in sorted {
        match ev.tokens {
            Some(t) => println!(
                "[DRY-RUN] user={} request={} input_tokens={} tokens={}",
                ev.user_id, ev.request_id, ev.input_tokens, t
            ),
            None => println!(
                "[DRY-RUN] user={} request={} input_tokens={} tokens=(not set)",
                ev.user_id, ev.request_id, ev.input_tokens
            ),
        }
    }
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
