use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

#[derive(Clone, Debug)]
pub struct MetricsConfig {
    pub endpoint: Url,
    pub output_dir: PathBuf,
    pub interval: Duration,
    pub timeout: Duration,
    pub fail_on_error: bool,
    pub metadata: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetricsRunReport {
    pub output_dir: String,
    pub scrape_count: u64,
    pub scrape_error_count: u64,
    pub first_timestamp: Option<String>,
    pub last_timestamp: Option<String>,
}

pub struct MetricsCollectorHandle {
    stop_tx: Option<oneshot::Sender<()>>,
    join_handle: JoinHandle<Result<MetricsRunReport>>,
}

impl MetricsCollectorHandle {
    pub async fn stop(mut self) -> Result<MetricsRunReport> {
        if let Some(stop_tx) = self.stop_tx.take() {
            let _ = stop_tx.send(());
        }
        self.join_handle
            .await
            .map_err(|err| anyhow!("metrics collector task failed: {}", err))?
    }
}

pub async fn spawn_metrics_collector(config: MetricsConfig) -> Result<MetricsCollectorHandle> {
    let (stop_tx, stop_rx) = oneshot::channel();
    let (ready_tx, ready_rx) = oneshot::channel();
    let join_handle = tokio::spawn(async move { collect_metrics(config, stop_rx, ready_tx).await });
    ready_rx
        .await
        .map_err(|_| anyhow!("metrics collector exited before initial scrape"))??;
    Ok(MetricsCollectorHandle {
        stop_tx: Some(stop_tx),
        join_handle,
    })
}

async fn collect_metrics(
    config: MetricsConfig,
    mut stop_rx: oneshot::Receiver<()>,
    ready_tx: oneshot::Sender<Result<()>>,
) -> Result<MetricsRunReport> {
    fs::create_dir_all(&config.output_dir).with_context(|| {
        format!(
            "failed to create metrics output directory {}",
            config.output_dir.display()
        )
    })?;

    let raw_path = config.output_dir.join("raw.promjsonl");
    let samples_path = config.output_dir.join("samples.csv");
    let metadata_path = config.output_dir.join("metadata.json");
    let summary_path = config.output_dir.join("summary.json");

    let mut raw_writer = BufWriter::new(
        File::create(&raw_path)
            .with_context(|| format!("failed to create metrics raw log {}", raw_path.display()))?,
    );
    let mut sample_writer = csv::Writer::from_path(&samples_path).with_context(|| {
        format!(
            "failed to create parsed metrics samples CSV {}",
            samples_path.display()
        )
    })?;

    let client = Client::builder()
        .timeout(config.timeout)
        .build()
        .context("failed to construct metrics HTTP client")?;
    let started_at = Utc::now();
    let start = Instant::now();
    let mut state = MetricsSummaryBuilder::new(config.output_dir.display().to_string());
    let mut seq = 0u64;

    if let Err(err) = scrape_once(
        &client,
        &config,
        &mut raw_writer,
        &mut sample_writer,
        &mut state,
        seq,
        start,
    )
    .await
    {
        let message = err.to_string();
        let _ = ready_tx.send(Err(anyhow!(message)));
        return Err(err);
    }
    let _ = ready_tx.send(Ok(()));
    seq += 1;

    loop {
        tokio::select! {
            _ = tokio::time::sleep(config.interval) => {
                scrape_once(
                    &client,
                    &config,
                    &mut raw_writer,
                    &mut sample_writer,
                    &mut state,
                    seq,
                    start,
                )
                .await?;
                seq += 1;
            }
            _ = &mut stop_rx => {
                break;
            }
        }
    }

    scrape_once(
        &client,
        &config,
        &mut raw_writer,
        &mut sample_writer,
        &mut state,
        seq,
        start,
    )
    .await?;

    raw_writer.flush()?;
    sample_writer.flush()?;

    let ended_at = Utc::now();
    let report = state.report();
    let summary = state.summary();
    let metadata = json!({
        "schema_version": 1,
        "benchmark_start_utc": started_at.to_rfc3339(),
        "benchmark_end_utc": ended_at.to_rfc3339(),
        "metrics_url": config.endpoint.as_str(),
        "interval_ms": config.interval.as_millis(),
        "timeout_ms": config.timeout.as_millis(),
        "fail_on_error": config.fail_on_error,
        "run": config.metadata,
    });
    fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)
        .with_context(|| format!("failed to write {}", metadata_path.display()))?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;

    if config.fail_on_error && report.scrape_error_count > 0 {
        return Err(anyhow!(
            "metrics scraping recorded {} error(s); see {}",
            report.scrape_error_count,
            raw_path.display()
        ));
    }

    Ok(report)
}

async fn scrape_once(
    client: &Client,
    config: &MetricsConfig,
    raw_writer: &mut BufWriter<File>,
    sample_writer: &mut csv::Writer<File>,
    state: &mut MetricsSummaryBuilder,
    seq: u64,
    start: Instant,
) -> Result<()> {
    let timestamp = Utc::now();
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let scrape_start = Instant::now();
    let result = client.get(config.endpoint.clone()).send().await;
    let scrape_duration_ms = scrape_start.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|err| err.to_string());
            if status.is_success() {
                match parse_prometheus_text(&body) {
                    Ok(samples) => {
                        write_raw_record(
                            raw_writer,
                            RawScrapeRecord::success(
                                seq,
                                timestamp,
                                elapsed_ms,
                                scrape_duration_ms,
                                status,
                                &body,
                            ),
                        )?;
                        state.record_success(timestamp, &samples);
                        for sample in samples {
                            sample_writer.serialize(SampleCsvRecord {
                                seq,
                                timestamp_utc: timestamp.to_rfc3339(),
                                elapsed_ms,
                                name: sample.name,
                                metric_type: sample.metric_type,
                                labels_json: serde_json::to_string(&sample.labels)?,
                                value: sample.value.to_string(),
                            })?;
                        }
                    }
                    Err(err) => {
                        state.record_error();
                        write_raw_record(
                            raw_writer,
                            RawScrapeRecord::parse_error(
                                seq,
                                timestamp,
                                elapsed_ms,
                                scrape_duration_ms,
                                status,
                                body,
                                err.to_string(),
                            ),
                        )?;
                    }
                }
            } else {
                state.record_error();
                write_raw_record(
                    raw_writer,
                    RawScrapeRecord::http_error(
                        seq,
                        timestamp,
                        elapsed_ms,
                        scrape_duration_ms,
                        status,
                        body,
                    ),
                )?;
            }
        }
        Err(err) => {
            state.record_error();
            write_raw_record(
                raw_writer,
                RawScrapeRecord::request_error(
                    seq,
                    timestamp,
                    elapsed_ms,
                    scrape_duration_ms,
                    err.to_string(),
                ),
            )?;
        }
    }

    raw_writer.flush()?;
    sample_writer.flush()?;
    Ok(())
}

fn write_raw_record(writer: &mut BufWriter<File>, record: RawScrapeRecord) -> Result<()> {
    serde_json::to_writer(&mut *writer, &record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

#[derive(Serialize)]
struct RawScrapeRecord {
    seq: u64,
    timestamp_utc: String,
    elapsed_ms: u64,
    status: String,
    http_status: Option<u16>,
    scrape_duration_ms: u64,
    body: Option<String>,
    error: Option<String>,
}

impl RawScrapeRecord {
    fn success(
        seq: u64,
        timestamp: DateTime<Utc>,
        elapsed_ms: u64,
        scrape_duration_ms: u64,
        status: StatusCode,
        body: &str,
    ) -> Self {
        Self {
            seq,
            timestamp_utc: timestamp.to_rfc3339(),
            elapsed_ms,
            status: "ok".to_string(),
            http_status: Some(status.as_u16()),
            scrape_duration_ms,
            body: Some(body.to_string()),
            error: None,
        }
    }

    fn parse_error(
        seq: u64,
        timestamp: DateTime<Utc>,
        elapsed_ms: u64,
        scrape_duration_ms: u64,
        status: StatusCode,
        body: String,
        error: String,
    ) -> Self {
        Self {
            seq,
            timestamp_utc: timestamp.to_rfc3339(),
            elapsed_ms,
            status: "parse_error".to_string(),
            http_status: Some(status.as_u16()),
            scrape_duration_ms,
            body: Some(body),
            error: Some(error),
        }
    }

    fn http_error(
        seq: u64,
        timestamp: DateTime<Utc>,
        elapsed_ms: u64,
        scrape_duration_ms: u64,
        status: StatusCode,
        body: String,
    ) -> Self {
        Self {
            seq,
            timestamp_utc: timestamp.to_rfc3339(),
            elapsed_ms,
            status: "http_error".to_string(),
            http_status: Some(status.as_u16()),
            scrape_duration_ms,
            body: Some(body),
            error: None,
        }
    }

    fn request_error(
        seq: u64,
        timestamp: DateTime<Utc>,
        elapsed_ms: u64,
        scrape_duration_ms: u64,
        error: String,
    ) -> Self {
        Self {
            seq,
            timestamp_utc: timestamp.to_rfc3339(),
            elapsed_ms,
            status: "request_error".to_string(),
            http_status: None,
            scrape_duration_ms,
            body: None,
            error: Some(error),
        }
    }
}

#[derive(Serialize)]
struct SampleCsvRecord {
    seq: u64,
    timestamp_utc: String,
    elapsed_ms: u64,
    name: String,
    metric_type: String,
    labels_json: String,
    value: String,
}

#[derive(Clone, Debug)]
struct ParsedSample {
    name: String,
    metric_type: String,
    labels: BTreeMap<String, String>,
    value: f64,
}

fn parse_prometheus_text(text: &str) -> Result<Vec<ParsedSample>> {
    let mut metric_types = HashMap::new();
    let mut samples = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let mut parts = rest.split_whitespace();
            if let (Some(name), Some(metric_type)) = (parts.next(), parts.next()) {
                metric_types.insert(name.to_string(), metric_type.to_string());
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }

        let sample = parse_sample_line(line, &metric_types)
            .with_context(|| format!("failed to parse Prometheus sample line: {}", line))?;
        samples.push(sample);
    }

    Ok(samples)
}

fn parse_sample_line(line: &str, metric_types: &HashMap<String, String>) -> Result<ParsedSample> {
    let (name_and_labels, rest) =
        split_sample_name_and_rest(line).ok_or_else(|| anyhow!("sample line missing value"))?;
    let value_text = rest
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow!("sample line missing value"))?;
    let value = parse_prometheus_float(value_text)?;

    let (name, labels) = if let Some(label_start) = name_and_labels.find('{') {
        let label_end = name_and_labels
            .rfind('}')
            .ok_or_else(|| anyhow!("metric labels missing closing brace"))?;
        let name = &name_and_labels[..label_start];
        let labels_text = &name_and_labels[label_start + 1..label_end];
        (name.to_string(), parse_labels(labels_text)?)
    } else {
        (name_and_labels.to_string(), BTreeMap::new())
    };

    let metric_type = infer_metric_type(&name, metric_types);
    Ok(ParsedSample {
        name,
        metric_type,
        labels,
        value,
    })
}

fn split_sample_name_and_rest(line: &str) -> Option<(&str, &str)> {
    let mut in_labels = false;
    let mut in_quotes = false;
    let mut escaped = false;

    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_quotes && ch == '\\' {
            escaped = true;
            continue;
        }
        if in_labels && ch == '"' {
            in_quotes = !in_quotes;
            continue;
        }
        if !in_quotes {
            if ch == '{' {
                in_labels = true;
                continue;
            }
            if ch == '}' {
                in_labels = false;
                continue;
            }
            if ch.is_whitespace() {
                return Some((&line[..idx], line[idx..].trim_start()));
            }
        }
    }

    None
}

fn infer_metric_type(name: &str, metric_types: &HashMap<String, String>) -> String {
    if let Some(metric_type) = metric_types.get(name) {
        return metric_type.clone();
    }
    for suffix in ["_bucket", "_sum", "_count"] {
        if let Some(base) = name.strip_suffix(suffix) {
            if let Some(metric_type) = metric_types.get(base) {
                return metric_type.clone();
            }
        }
    }
    "untyped".to_string()
}

fn parse_labels(labels_text: &str) -> Result<BTreeMap<String, String>> {
    let mut labels = BTreeMap::new();
    let mut key = String::new();
    let mut value = String::new();
    let mut in_key = true;
    let mut in_quotes = false;
    let mut escaped = false;

    for ch in labels_text.chars().chain(std::iter::once(',')) {
        if escaped {
            value.push(match ch {
                'n' => '\n',
                't' => '\t',
                '\\' => '\\',
                '"' => '"',
                other => other,
            });
            escaped = false;
            continue;
        }
        if in_quotes && ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_quotes = !in_quotes;
            continue;
        }
        if !in_quotes && in_key && ch == '=' {
            in_key = false;
            continue;
        }
        if !in_quotes && ch == ',' {
            let label_key = key.trim();
            if !label_key.is_empty() {
                labels.insert(label_key.to_string(), value.clone());
            }
            key.clear();
            value.clear();
            in_key = true;
            continue;
        }
        if in_key {
            key.push(ch);
        } else {
            value.push(ch);
        }
    }

    if in_quotes {
        return Err(anyhow!("unterminated label quote"));
    }

    Ok(labels)
}

fn parse_prometheus_float(value: &str) -> Result<f64> {
    match value {
        "NaN" | "nan" => Ok(f64::NAN),
        "Inf" | "+Inf" | "inf" | "+inf" => Ok(f64::INFINITY),
        "-Inf" | "-inf" => Ok(f64::NEG_INFINITY),
        other => other
            .parse::<f64>()
            .with_context(|| format!("invalid Prometheus float: {}", value)),
    }
}

#[derive(Default)]
struct MetricsSummaryBuilder {
    output_dir: String,
    scrape_count: u64,
    scrape_error_count: u64,
    first_timestamp: Option<DateTime<Utc>>,
    last_timestamp: Option<DateTime<Utc>>,
    scalar_stats: BTreeMap<String, ScalarMetricStats>,
    histogram_buckets: BTreeMap<String, BTreeMap<String, BucketStats>>,
}

impl MetricsSummaryBuilder {
    fn new(output_dir: String) -> Self {
        Self {
            output_dir,
            ..Self::default()
        }
    }

    fn record_success(&mut self, timestamp: DateTime<Utc>, samples: &[ParsedSample]) {
        self.scrape_count += 1;
        self.first_timestamp.get_or_insert(timestamp);
        self.last_timestamp = Some(timestamp);

        let mut aggregated = BTreeMap::<(String, String), f64>::new();
        let mut buckets = BTreeMap::<(String, String), f64>::new();

        for sample in samples {
            if !sample.value.is_finite() {
                continue;
            }
            if sample.name.ends_with("_bucket") {
                if let Some(le) = sample.labels.get("le") {
                    let base = sample.name.trim_end_matches("_bucket").to_string();
                    *buckets.entry((base, le.clone())).or_default() += sample.value;
                }
                continue;
            }
            if sample.name.ends_with("_sum") || sample.name.ends_with("_count") {
                continue;
            }
            *aggregated
                .entry((sample.name.clone(), sample.metric_type.clone()))
                .or_default() += sample.value;
        }

        for ((name, metric_type), value) in aggregated {
            self.scalar_stats
                .entry(name)
                .or_insert_with(|| ScalarMetricStats::new(metric_type))
                .record(value);
        }
        for ((name, le), value) in buckets {
            self.histogram_buckets
                .entry(name)
                .or_default()
                .entry(le)
                .or_default()
                .record(value);
        }
    }

    fn record_error(&mut self) {
        self.scrape_error_count += 1;
    }

    fn report(&self) -> MetricsRunReport {
        MetricsRunReport {
            output_dir: self.output_dir.clone(),
            scrape_count: self.scrape_count,
            scrape_error_count: self.scrape_error_count,
            first_timestamp: self.first_timestamp.map(|ts| ts.to_rfc3339()),
            last_timestamp: self.last_timestamp.map(|ts| ts.to_rfc3339()),
        }
    }

    fn summary(&self) -> Value {
        let mut counters = serde_json::Map::new();
        let mut gauges = serde_json::Map::new();
        for (name, stats) in &self.scalar_stats {
            match stats.metric_type.as_str() {
                "counter" => {
                    counters.insert(name.clone(), stats.counter_json());
                }
                "gauge" => {
                    gauges.insert(name.clone(), stats.gauge_json());
                }
                _ => {}
            }
        }

        let mut histograms = serde_json::Map::new();
        for (name, buckets) in &self.histogram_buckets {
            histograms.insert(name.clone(), histogram_json(buckets));
        }

        json!({
            "schema_version": 1,
            "scrape_count": self.scrape_count,
            "scrape_error_count": self.scrape_error_count,
            "first_timestamp": self.first_timestamp.map(|ts| ts.to_rfc3339()),
            "last_timestamp": self.last_timestamp.map(|ts| ts.to_rfc3339()),
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
        })
    }
}

#[derive(Default)]
struct ScalarMetricStats {
    metric_type: String,
    first: Option<f64>,
    last: Option<f64>,
    min: f64,
    max: f64,
    sum: f64,
    count: u64,
}

impl ScalarMetricStats {
    fn new(metric_type: String) -> Self {
        Self {
            metric_type,
            ..Self::default()
        }
    }

    fn record(&mut self, value: f64) {
        self.first.get_or_insert(value);
        self.last = Some(value);
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.sum += value;
        self.count += 1;
    }

    fn counter_json(&self) -> Value {
        let first = self.first.unwrap_or(0.0);
        let last = self.last.unwrap_or(first);
        json!({
            "first": first,
            "last": last,
            "delta": if last >= first { last - first } else { last },
        })
    }

    fn gauge_json(&self) -> Value {
        json!({
            "first": self.first,
            "last": self.last,
            "min": self.min,
            "max": self.max,
            "mean": if self.count > 0 { Some(self.sum / self.count as f64) } else { None },
        })
    }
}

#[derive(Default)]
struct BucketStats {
    first: Option<f64>,
    last: Option<f64>,
}

impl BucketStats {
    fn record(&mut self, value: f64) {
        self.first.get_or_insert(value);
        self.last = Some(value);
    }

    fn delta(&self) -> f64 {
        let first = self.first.unwrap_or(0.0);
        let last = self.last.unwrap_or(first);
        if last >= first {
            last - first
        } else {
            last
        }
    }
}

fn histogram_json(buckets: &BTreeMap<String, BucketStats>) -> Value {
    let mut bucket_values = Vec::new();
    for (le, stats) in buckets {
        bucket_values.push(json!({
            "le": le,
            "delta": stats.delta(),
        }));
    }

    json!({
        "buckets": bucket_values,
        "p50": approximate_histogram_quantile(buckets, 0.50),
        "p90": approximate_histogram_quantile(buckets, 0.90),
        "p99": approximate_histogram_quantile(buckets, 0.99),
    })
}

fn approximate_histogram_quantile(
    buckets: &BTreeMap<String, BucketStats>,
    quantile: f64,
) -> Option<f64> {
    let mut parsed = buckets
        .iter()
        .filter_map(|(le, stats)| parse_bucket_bound(le).map(|bound| (bound, stats.delta())))
        .collect::<Vec<_>>();
    parsed.sort_by(|a, b| a.0.total_cmp(&b.0));

    let total = parsed
        .iter()
        .find(|(bound, _)| bound.is_infinite())
        .map(|(_, count)| *count)
        .or_else(|| parsed.last().map(|(_, count)| *count))?;
    if total <= 0.0 {
        return None;
    }

    let target = total * quantile;
    let mut previous_bound = 0.0;
    let mut previous_count = 0.0;
    for (bound, cumulative_count) in parsed {
        if cumulative_count >= target {
            if bound.is_infinite() {
                return Some(previous_bound);
            }
            let bucket_count = cumulative_count - previous_count;
            if bucket_count <= 0.0 {
                return Some(bound);
            }
            let within_bucket = (target - previous_count) / bucket_count;
            return Some(previous_bound + (bound - previous_bound) * within_bucket);
        }
        previous_bound = bound;
        previous_count = cumulative_count;
    }
    None
}

fn parse_bucket_bound(value: &str) -> Option<f64> {
    match value {
        "+Inf" | "Inf" | "inf" | "+inf" => Some(f64::INFINITY),
        other => other.parse::<f64>().ok(),
    }
}

pub fn resolve_metrics_endpoint(host: &str, endpoint: &str) -> Result<Url> {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        return Url::parse(endpoint)
            .with_context(|| format!("invalid metrics endpoint: {}", endpoint));
    }

    let normalized_host = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("https://{}", host.trim_end_matches('/'))
    };
    let resolved = format!("{}/{}", normalized_host, endpoint.trim_start_matches('/'));
    Url::parse(&resolved).with_context(|| format!("invalid metrics endpoint: {}", resolved))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_metrics_endpoint_against_host() {
        let url = resolve_metrics_endpoint("http://127.0.0.1:3000/", "/metrics").unwrap();
        assert_eq!(url.as_str(), "http://127.0.0.1:3000/metrics");
    }

    #[test]
    fn parses_colon_metric_names_and_labels() {
        let parsed = parse_prometheus_text(
            r#"
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="Qwen/Qwen3"} 12
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="Qwen/Qwen3"} 0.25
"#,
        )
        .unwrap();

        assert_eq!(parsed[0].name, "vllm:prompt_tokens_total");
        assert_eq!(parsed[0].metric_type, "counter");
        assert_eq!(parsed[0].labels.get("model_name").unwrap(), "Qwen/Qwen3");
        assert_eq!(parsed[1].name, "sglang:token_usage");
        assert_eq!(parsed[1].metric_type, "gauge");
    }

    #[test]
    fn summarizes_counter_deltas_and_gauge_stats() {
        let mut summary = MetricsSummaryBuilder::new("metrics".to_string());
        let first = parse_prometheus_text(
            r#"
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 10
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 1
"#,
        )
        .unwrap();
        let second = parse_prometheus_text(
            r#"
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 25
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
"#,
        )
        .unwrap();

        summary.record_success(Utc::now(), &first);
        summary.record_success(Utc::now(), &second);
        let payload = summary.summary();

        assert_eq!(
            payload["counters"]["vllm:prompt_tokens_total"]["delta"],
            json!(15.0)
        );
        assert_eq!(
            payload["gauges"]["vllm:num_requests_running"]["max"],
            json!(3.0)
        );
    }
}
