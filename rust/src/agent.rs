use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, Url};
use serde::Serialize;
use serde_json::{json, Map, Value};
use tokenizers::Tokenizer;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::{Barrier, Mutex};
use tokio::task::JoinSet;

const DYNAMO_SESSION_HEADER: &str = "x-dynamo-session-id";

/// A request header whose value is rendered independently for each agent.
///
/// The literal `{agent_id}` placeholder is replaced with the zero-based agent ID.
#[derive(Clone, Debug)]
pub struct AgentHeaderTemplate {
    name: HeaderName,
    value_template: String,
}

impl AgentHeaderTemplate {
    pub fn try_new(name: impl AsRef<str>, value_template: impl Into<String>) -> Result<Self> {
        let raw_name = name.as_ref();
        let name = HeaderName::from_bytes(raw_name.as_bytes())
            .with_context(|| format!("invalid agent header name {raw_name:?}"))?;
        let value_template = value_template.into();
        if !value_template.contains("{agent_id}") {
            return Err(anyhow!(
                "agent header template for {raw_name} must contain {{agent_id}}"
            ));
        }
        let template = Self {
            name,
            value_template,
        };
        template.render(0)?;
        Ok(template)
    }

    pub(crate) fn render(&self, agent_id: usize) -> Result<(HeaderName, HeaderValue)> {
        let value = self
            .value_template
            .replace("{agent_id}", &agent_id.to_string());
        let value = HeaderValue::from_str(&value).with_context(|| {
            format!(
                "agent header {} rendered an invalid value for agent {}",
                self.name, agent_id
            )
        })?;
        Ok((self.name.clone(), value))
    }
}

/// A fixed value or a value independently sampled from a log-normal distribution.
#[derive(Clone, Debug)]
pub enum SampleSpec {
    Fixed(usize),
    LogNormal {
        mu: f64,
        sigma: f64,
        max: Option<usize>,
    },
}

impl SampleSpec {
    pub fn fixed(value: usize) -> Result<Self> {
        let spec = Self::Fixed(value);
        spec.validate("sample")?;
        Ok(spec)
    }

    pub fn log_normal(mu: f64, sigma: f64, max: Option<usize>) -> Result<Self> {
        let spec = Self::LogNormal { mu, sigma, max };
        spec.validate("sample")?;
        Ok(spec)
    }

    pub fn log_normal_from_median(median: f64, sigma: f64, max: Option<usize>) -> Result<Self> {
        if !median.is_finite() || median <= 0.0 {
            return Err(anyhow!("log-normal median must be greater than zero"));
        }
        Self::log_normal(median.ln(), sigma, max)
    }

    pub fn validate(&self, label: &str) -> Result<()> {
        match self {
            Self::Fixed(value) => {
                if *value == 0 {
                    return Err(anyhow!("{} must be greater than zero", label));
                }
            }
            Self::LogNormal { mu, sigma, max } => {
                if !mu.is_finite() {
                    return Err(anyhow!("{} log-normal mu must be finite", label));
                }
                if !sigma.is_finite() || *sigma <= 0.0 {
                    return Err(anyhow!(
                        "{} log-normal sigma must be greater than zero",
                        label
                    ));
                }
                if max == &Some(0) {
                    return Err(anyhow!(
                        "{} log-normal max must be greater than zero",
                        label
                    ));
                }
                LogNormal::new(*mu, *sigma).with_context(|| {
                    format!("failed to create {} log-normal distribution", label)
                })?;
            }
        }
        Ok(())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, label: &str) -> Result<usize> {
        match self {
            Self::Fixed(value) => Ok(*value),
            Self::LogNormal { mu, sigma, max } => {
                let distribution = LogNormal::new(*mu, *sigma).with_context(|| {
                    format!("failed to create {} log-normal distribution", label)
                })?;
                let sampled = distribution.sample(rng);
                if !sampled.is_finite() {
                    return Err(anyhow!(
                        "{} log-normal distribution produced a non-finite sample",
                        label
                    ));
                }

                let mut rounded = sampled.round().max(1.0);
                if let Some(max) = max {
                    rounded = rounded.min(*max as f64);
                }
                if rounded > usize::MAX as f64 {
                    return Err(anyhow!("{} sample exceeds usize::MAX", label));
                }
                Ok(rounded as usize)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct AgentLoopConfig {
    pub endpoint: Url,
    pub model: String,
    pub tokenizer_model: String,
    pub agent_count: usize,
    pub input_tokens: SampleSpec,
    pub output_tokens: SampleSpec,
    pub environment_tokens: SampleSpec,
    pub tool_invocations: SampleSpec,
    pub tool_call_latency_ms: Option<SampleSpec>,
    pub temperature: Option<f64>,
    pub max_tokens_only: bool,
    pub user_prefix: Option<String>,
    pub agent_header_templates: Vec<AgentHeaderTemplate>,
    pub nvext_extra_fields: Vec<String>,
    pub events_jsonl: Option<PathBuf>,
    pub barrier_after_invocation: Option<usize>,
    pub request_timeout: Duration,
    pub max_retries: usize,
    pub retry_delay: Duration,
    pub headers: HeaderMap,
    pub verbose: bool,
    pub dry_run: bool,
    pub sglang: bool,
    pub seed: Option<u64>,
}

impl AgentLoopConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        endpoint: impl AsRef<str>,
        api_key: Option<String>,
        model: impl Into<String>,
        agent_count: usize,
        input_tokens: SampleSpec,
        output_tokens: SampleSpec,
        environment_tokens: SampleSpec,
        tool_invocations: SampleSpec,
    ) -> Result<Self> {
        if agent_count == 0 {
            return Err(anyhow!("agent_count must be greater than zero"));
        }
        input_tokens.validate("input tokens")?;
        output_tokens.validate("output tokens")?;
        environment_tokens.validate("environment tokens")?;
        tool_invocations.validate("tool invocations")?;

        let endpoint = Url::parse(endpoint.as_ref())
            .with_context(|| format!("invalid endpoint URL: {}", endpoint.as_ref()))?;
        let model = model.into();
        if model.is_empty() {
            return Err(anyhow!("model must not be empty"));
        }

        let mut headers = HeaderMap::new();
        if let Some(api_key) = api_key {
            if !api_key.is_empty() {
                let value = HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .context("failed to build Authorization header from api_key")?;
                headers.insert(AUTHORIZATION, value);
            }
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(Self {
            endpoint,
            tokenizer_model: model.clone(),
            model,
            agent_count,
            input_tokens,
            output_tokens,
            environment_tokens,
            tool_invocations,
            tool_call_latency_ms: None,
            temperature: None,
            max_tokens_only: false,
            user_prefix: None,
            agent_header_templates: Vec::new(),
            nvext_extra_fields: Vec::new(),
            events_jsonl: None,
            barrier_after_invocation: None,
            request_timeout: Duration::from_secs(60),
            max_retries: 2,
            retry_delay: Duration::from_millis(250),
            headers,
            verbose: false,
            dry_run: false,
            sglang: false,
            seed: None,
        })
    }

    pub fn with_tokenizer_model(mut self, tokenizer_model: impl Into<String>) -> Self {
        self.tokenizer_model = tokenizer_model.into();
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Result<Self> {
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err(anyhow!("temperature must be finite and between 0 and 2"));
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    pub fn with_max_tokens_only(mut self, enabled: bool) -> Self {
        self.max_tokens_only = enabled;
        self
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        if !timeout.is_zero() {
            self.request_timeout = timeout;
        }
        self
    }

    pub fn with_retry(mut self, max_retries: usize, retry_delay: Duration) -> Self {
        self.max_retries = max_retries;
        if !retry_delay.is_zero() {
            self.retry_delay = retry_delay;
        }
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    pub fn with_sglang(mut self, sglang: bool) -> Self {
        self.sglang = sglang;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_user_prefix(mut self, user_prefix: impl Into<String>) -> Self {
        self.user_prefix = Some(user_prefix.into());
        self
    }

    pub fn add_agent_header_template(mut self, template: AgentHeaderTemplate) -> Result<Self> {
        if self
            .agent_header_templates
            .iter()
            .any(|existing| existing.name == template.name)
        {
            return Err(anyhow!(
                "duplicate agent header template for {}",
                template.name
            ));
        }
        self.agent_header_templates.push(template);
        Ok(self)
    }

    pub fn with_nvext_extra_fields(mut self, fields: Vec<String>) -> Result<Self> {
        for field in &fields {
            if field.trim().is_empty() {
                return Err(anyhow!("nvext extra field must not be empty"));
            }
        }
        self.nvext_extra_fields = fields;
        Ok(self)
    }

    pub fn with_events_jsonl(mut self, path: impl Into<PathBuf>) -> Self {
        self.events_jsonl = Some(path.into());
        self
    }

    pub fn with_barrier_after_invocation(mut self, invocation: usize) -> Result<Self> {
        if invocation == 0 {
            return Err(anyhow!("barrier invocation must be greater than zero"));
        }
        match self.tool_invocations {
            SampleSpec::Fixed(turns) if invocation < turns => {}
            SampleSpec::Fixed(turns) => {
                return Err(anyhow!(
                    "barrier invocation {invocation} must be less than the fixed turn count {turns}"
                ));
            }
            SampleSpec::LogNormal { .. } => {
                return Err(anyhow!(
                    "barrier-after-invocation requires a fixed --tool-invocations value"
                ));
            }
        }
        self.barrier_after_invocation = Some(invocation);
        Ok(self)
    }

    pub fn with_tool_call_latency_ms(mut self, latency_ms: SampleSpec) -> Result<Self> {
        latency_ms.validate("tool call latency")?;
        self.tool_call_latency_ms = Some(latency_ms);
        Ok(self)
    }

    pub fn add_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.insert(name, value);
        self
    }
}

#[derive(Clone)]
struct AgentEventSink {
    writer: Arc<Mutex<BufWriter<tokio::fs::File>>>,
}

struct InvocationBarrier {
    barrier: Barrier,
    cancelled: AtomicBool,
    cancelled_notify: tokio::sync::Notify,
}

impl InvocationBarrier {
    fn new(participants: usize) -> Arc<Self> {
        Arc::new(Self {
            barrier: Barrier::new(participants),
            cancelled: AtomicBool::new(false),
            cancelled_notify: tokio::sync::Notify::new(),
        })
    }

    fn participant(self: &Arc<Self>) -> InvocationBarrierParticipant {
        InvocationBarrierParticipant {
            barrier: self.clone(),
            passed: false,
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.cancelled_notify.notify_waiters();
    }

    async fn wait(&self) {
        let cancelled = self.cancelled_notify.notified();
        tokio::pin!(cancelled);
        cancelled.as_mut().enable();
        if self.cancelled.load(Ordering::Acquire) {
            return;
        }
        tokio::select! {
            _ = self.barrier.wait() => {}
            _ = cancelled => {}
        }
    }
}

struct InvocationBarrierParticipant {
    barrier: Arc<InvocationBarrier>,
    passed: bool,
}

impl InvocationBarrierParticipant {
    async fn wait(&mut self) {
        self.barrier.wait().await;
        self.passed = true;
    }
}

impl Drop for InvocationBarrierParticipant {
    fn drop(&mut self) {
        if !self.passed {
            self.barrier.cancel();
        }
    }
}

impl AgentEventSink {
    async fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
        }
        let file = tokio::fs::File::create(path)
            .await
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self {
            writer: Arc::new(Mutex::new(BufWriter::new(file))),
        })
    }

    async fn write<T: Serialize>(&self, event: &T) -> Result<()> {
        let mut line = serde_json::to_vec(event).context("failed to serialize agent event")?;
        line.push(b'\n');
        let mut writer = self.writer.lock().await;
        writer
            .write_all(&line)
            .await
            .context("failed to write agent event")?;
        writer
            .flush()
            .await
            .context("failed to flush agent event")?;
        Ok(())
    }
}

#[derive(Serialize)]
struct AgentRequestEvent<'a> {
    event: &'static str,
    timestamp: String,
    agent_id: usize,
    invocation: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    worker_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dp_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

impl<'a> AgentRequestEvent<'a> {
    fn started(agent_id: usize, invocation: usize, session_id: Option<&'a str>) -> Self {
        Self {
            event: "request_start",
            timestamp: event_timestamp(),
            agent_id,
            invocation,
            session_id,
            status: None,
            latency_ms: None,
            prompt_tokens: None,
            cached_tokens: None,
            completion_tokens: None,
            worker_id: None,
            dp_rank: None,
            error: None,
        }
    }

    fn completed(
        agent_id: usize,
        invocation: usize,
        session_id: Option<&'a str>,
        result: &RequestResult,
    ) -> Self {
        Self {
            event: "request_complete",
            timestamp: event_timestamp(),
            agent_id,
            invocation,
            session_id,
            status: Some("success"),
            latency_ms: Some(result.latency.as_secs_f64() * 1000.0),
            prompt_tokens: Some(result.prompt_tokens),
            cached_tokens: result.cached_tokens,
            completion_tokens: Some(result.completion_tokens),
            worker_id: result.worker_id,
            dp_rank: result.dp_rank,
            error: None,
        }
    }

    fn failed(
        agent_id: usize,
        invocation: usize,
        session_id: Option<&'a str>,
        latency: Duration,
        error: &'a str,
    ) -> Self {
        Self {
            event: "request_complete",
            timestamp: event_timestamp(),
            agent_id,
            invocation,
            session_id,
            status: Some("failure"),
            latency_ms: Some(latency.as_secs_f64() * 1000.0),
            prompt_tokens: None,
            cached_tokens: None,
            completion_tokens: None,
            worker_id: None,
            dp_rank: None,
            error: Some(error),
        }
    }
}

fn event_timestamp() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[derive(Clone, Debug)]
struct AgentTurnPlan {
    output_tokens: usize,
    environment_tokens: usize,
    environment_content: String,
    tool_call_latency: Duration,
}

#[derive(Clone, Debug)]
struct AgentPlan {
    agent_id: usize,
    initial_prompt_tokens: usize,
    initial_prompt: String,
    turns: Vec<AgentTurnPlan>,
}

#[derive(Clone, Debug)]
pub struct AgentFailureRecord {
    pub agent_id: usize,
    pub invocation: usize,
    pub error: String,
}

#[derive(Clone, Debug)]
pub struct AgentBenchmarkReport {
    pub total_agents: usize,
    pub completed_agents: usize,
    pub planned_tool_invocations: u64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub estimated_cached_input_tokens: u64,
    pub reported_cached_input_tokens: u64,
    pub total_tool_call_latency: Duration,
    pub total_duration: Duration,
    pub input_tokens_per_second: f64,
    pub output_tokens_per_second: f64,
    pub requests_per_second: f64,
    pub latency_p50: Option<Duration>,
    pub latency_p90: Option<Duration>,
    pub latency_p99: Option<Duration>,
    pub failures: Vec<AgentFailureRecord>,
}

#[derive(Debug)]
struct AgentWorkerReport {
    completed: bool,
    successful_requests: u64,
    failed_requests: u64,
    input_tokens: u64,
    output_tokens: u64,
    estimated_cached_input_tokens: u64,
    reported_cached_input_tokens: u64,
    tool_call_latency: Duration,
    latencies: Vec<Duration>,
    failures: Vec<AgentFailureRecord>,
    dry_run_records: Vec<DryRunRecord>,
}

impl AgentWorkerReport {
    fn new() -> Self {
        Self {
            completed: false,
            successful_requests: 0,
            failed_requests: 0,
            input_tokens: 0,
            output_tokens: 0,
            estimated_cached_input_tokens: 0,
            reported_cached_input_tokens: 0,
            tool_call_latency: Duration::ZERO,
            latencies: Vec::new(),
            failures: Vec::new(),
            dry_run_records: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct DryRunRecord {
    agent_id: usize,
    invocation: usize,
    input_content_tokens: usize,
    output_tokens: usize,
    environment_tokens: usize,
    tool_call_latency_ms: usize,
}

#[derive(Debug)]
struct RequestResult {
    prompt_tokens: u64,
    cached_tokens: Option<u64>,
    completion_tokens: u64,
    worker_id: Option<u64>,
    dp_rank: Option<u32>,
    assistant_message: Value,
    latency: Duration,
}

/// Run independent, stateful agent loops concurrently.
///
/// Each agent begins with a separately generated user message. Every model response and synthetic
/// tool response is appended to that agent's messages before its next request, preserving the
/// previous request as a prefix for server-side KV-cache testing.
pub async fn run_agent_benchmark(config: AgentLoopConfig) -> Result<AgentBenchmarkReport> {
    let plans = build_agent_plans(&config)?;
    let planned_tool_invocations = plans
        .iter()
        .try_fold(0u64, |total, plan| {
            total.checked_add(plan.turns.len() as u64)
        })
        .ok_or_else(|| anyhow!("planned tool invocation count overflowed"))?;

    let client = Client::builder()
        .timeout(config.request_timeout)
        .build()
        .context("failed to construct HTTP client")?;
    let event_sink = match config.events_jsonl.as_deref() {
        Some(path) => Some(AgentEventSink::create(path).await?),
        None => None,
    };
    let invocation_barrier = config
        .barrier_after_invocation
        .filter(|_| !config.dry_run)
        .map(|_| InvocationBarrier::new(config.agent_count));
    let config = std::sync::Arc::new(config);
    let start = Instant::now();

    let mut join_set = JoinSet::new();
    for plan in plans {
        let client = client.clone();
        let config = std::sync::Arc::clone(&config);
        let event_sink = event_sink.clone();
        let invocation_barrier = invocation_barrier
            .as_ref()
            .map(InvocationBarrier::participant);
        join_set.spawn(async move {
            run_agent(plan, client, config, event_sink, invocation_barrier).await
        });
    }

    let mut completed_agents = 0usize;
    let mut successful_requests = 0u64;
    let mut failed_requests = 0u64;
    let mut total_input_tokens = 0u64;
    let mut total_output_tokens = 0u64;
    let mut estimated_cached_input_tokens = 0u64;
    let mut reported_cached_input_tokens = 0u64;
    let mut total_tool_call_latency = Duration::ZERO;
    let mut latencies = Vec::new();
    let mut failures = Vec::new();
    let mut dry_run_records = Vec::new();

    while let Some(joined) = join_set.join_next().await {
        let worker = joined.map_err(|err| anyhow!("agent worker task failed: {}", err))??;
        if worker.completed {
            completed_agents += 1;
        }
        successful_requests = successful_requests.saturating_add(worker.successful_requests);
        failed_requests = failed_requests.saturating_add(worker.failed_requests);
        total_input_tokens = total_input_tokens.saturating_add(worker.input_tokens);
        total_output_tokens = total_output_tokens.saturating_add(worker.output_tokens);
        estimated_cached_input_tokens =
            estimated_cached_input_tokens.saturating_add(worker.estimated_cached_input_tokens);
        reported_cached_input_tokens =
            reported_cached_input_tokens.saturating_add(worker.reported_cached_input_tokens);
        total_tool_call_latency = total_tool_call_latency.saturating_add(worker.tool_call_latency);
        latencies.extend(worker.latencies);
        failures.extend(worker.failures);
        dry_run_records.extend(worker.dry_run_records);
    }

    if config.dry_run {
        dry_run_records.sort_by_key(|record| (record.agent_id, record.invocation));
        for record in dry_run_records {
            println!(
                "[DRY-RUN] agent={} invocation={} input_content_tokens={} output_tokens={} environment_tokens={} tool_call_latency_ms={}",
                record.agent_id,
                record.invocation,
                record.input_content_tokens,
                record.output_tokens,
                record.environment_tokens,
                record.tool_call_latency_ms
            );
        }
    }

    let total_duration = start.elapsed();
    latencies.sort();
    failures.sort_by_key(|failure| (failure.agent_id, failure.invocation));
    let total_requests = successful_requests.saturating_add(failed_requests);
    let duration_secs = total_duration.as_secs_f64();

    Ok(AgentBenchmarkReport {
        total_agents: config.agent_count,
        completed_agents,
        planned_tool_invocations,
        total_requests,
        successful_requests,
        failed_requests,
        total_input_tokens,
        total_output_tokens,
        estimated_cached_input_tokens,
        reported_cached_input_tokens,
        total_tool_call_latency,
        total_duration,
        input_tokens_per_second: rate(total_input_tokens, duration_secs),
        output_tokens_per_second: rate(total_output_tokens, duration_secs),
        requests_per_second: rate(total_requests, duration_secs),
        latency_p50: percentile(&latencies, 0.50),
        latency_p90: percentile(&latencies, 0.90),
        latency_p99: percentile(&latencies, 0.99),
        failures,
    })
}

fn build_agent_plans(config: &AgentLoopConfig) -> Result<Vec<AgentPlan>> {
    let tokenizer = Tokenizer::from_pretrained(&config.tokenizer_model, None).map_err(|err| {
        anyhow!(
            "failed to load tokenizer {}: {}",
            config.tokenizer_model,
            err
        )
    })?;
    let root_seed = match config.seed {
        Some(seed) => seed,
        None => rand::thread_rng().gen(),
    };

    let mut plans = Vec::with_capacity(config.agent_count);
    for agent_id in 0..config.agent_count {
        let mut input_rng = stream_rng(root_seed, agent_id, 0);
        let mut invocation_rng = stream_rng(root_seed, agent_id, 1);
        let mut output_rng = stream_rng(root_seed, agent_id, 2);
        let mut environment_rng = stream_rng(root_seed, agent_id, 3);
        let mut text_rng = stream_rng(root_seed, agent_id, 4);
        let mut tool_call_latency_rng = stream_rng(root_seed, agent_id, 5);

        let initial_prompt_tokens = config.input_tokens.sample(&mut input_rng, "input tokens")?;
        let invocation_count = config
            .tool_invocations
            .sample(&mut invocation_rng, "tool invocations")?;
        let initial_prompt =
            generate_synthetic_text(&tokenizer, initial_prompt_tokens, &mut text_rng)?;

        let mut turns = Vec::with_capacity(invocation_count);
        for _ in 0..invocation_count {
            let output_tokens = config
                .output_tokens
                .sample(&mut output_rng, "output tokens")?;
            let environment_tokens = config
                .environment_tokens
                .sample(&mut environment_rng, "environment tokens")?;
            let tool_call_latency_ms = match &config.tool_call_latency_ms {
                Some(spec) => spec.sample(&mut tool_call_latency_rng, "tool call latency")?,
                None => 0,
            };
            let tool_call_latency_ms = u64::try_from(tool_call_latency_ms)
                .context("sampled tool call latency exceeds u64::MAX milliseconds")?;
            let environment_content =
                generate_synthetic_text(&tokenizer, environment_tokens, &mut text_rng)?;
            turns.push(AgentTurnPlan {
                output_tokens,
                environment_tokens,
                environment_content,
                tool_call_latency: Duration::from_millis(tool_call_latency_ms),
            });
        }

        plans.push(AgentPlan {
            agent_id,
            initial_prompt_tokens,
            initial_prompt,
            turns,
        });
    }
    Ok(plans)
}

fn stream_rng(root_seed: u64, agent_id: usize, stream_id: u64) -> rand::rngs::StdRng {
    let mixed = root_seed
        .wrapping_add((agent_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(stream_id.wrapping_mul(0xD1B5_4A32_D192_ED03));
    rand::rngs::StdRng::seed_from_u64(mixed)
}

fn generate_synthetic_text<R: Rng + ?Sized>(
    tokenizer: &Tokenizer,
    target_tokens: usize,
    rng: &mut R,
) -> Result<String> {
    let vocab_size = tokenizer.get_vocab_size(false);
    if vocab_size == 0 {
        return Err(anyhow!("tokenizer vocabulary is empty"));
    }

    for _ in 0..8 {
        let ids: Vec<u32> = (0..target_tokens)
            .map(|_| rng.gen_range(0..vocab_size) as u32)
            .collect();
        let decoded = tokenizer
            .decode(&ids, true)
            .map_err(|err| anyhow!("failed to decode synthetic tokens: {}", err))?;
        if !decoded.trim().is_empty() {
            return Ok(decoded);
        }
    }

    Ok((0..target_tokens)
        .map(|_| rng.gen_range(1u32..=10_000u32).to_string())
        .collect::<Vec<_>>()
        .join(" "))
}

async fn run_agent(
    plan: AgentPlan,
    client: Client,
    config: std::sync::Arc<AgentLoopConfig>,
    event_sink: Option<AgentEventSink>,
    mut invocation_barrier: Option<InvocationBarrierParticipant>,
) -> Result<AgentWorkerReport> {
    let mut report = AgentWorkerReport::new();
    let mut messages = vec![json!({
        "role": "user",
        "content": plan.initial_prompt,
    })];
    let mut previous_prompt_tokens = 0u64;
    let agent_headers = render_agent_headers(&config, plan.agent_id)?;
    let session_id = agent_headers
        .get(DYNAMO_SESSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);

    for (turn_index, turn) in plan.turns.iter().enumerate() {
        let invocation = turn_index + 1;
        if config.dry_run {
            report.dry_run_records.push(DryRunRecord {
                agent_id: plan.agent_id,
                invocation,
                input_content_tokens: plan.initial_prompt_tokens
                    + plan.turns[..turn_index]
                        .iter()
                        .map(|prior| prior.output_tokens + prior.environment_tokens)
                        .sum::<usize>(),
                output_tokens: turn.output_tokens,
                environment_tokens: turn.environment_tokens,
                tool_call_latency_ms: turn.tool_call_latency.as_millis() as usize,
            });
            append_synthetic_turn(
                &mut messages,
                plan.agent_id,
                invocation,
                &format!("synthetic model response ({} tokens)", turn.output_tokens),
                &turn.environment_content,
            );
            continue;
        }

        let body = build_request_body(&config, &messages, turn.output_tokens, plan.agent_id);
        if let Some(sink) = &event_sink {
            sink.write(&AgentRequestEvent::started(
                plan.agent_id,
                invocation,
                session_id.as_deref(),
            ))
            .await?;
        }
        let request_start = Instant::now();
        match request_with_retries(&client, &config, &body, &agent_headers).await {
            Ok(result) => {
                if let Some(sink) = &event_sink {
                    sink.write(&AgentRequestEvent::completed(
                        plan.agent_id,
                        invocation,
                        session_id.as_deref(),
                        &result,
                    ))
                    .await?;
                }
                if config.barrier_after_invocation == Some(invocation) {
                    invocation_barrier
                        .as_mut()
                        .expect("configured invocation barrier must exist")
                        .wait()
                        .await;
                }
                report.successful_requests += 1;
                report.input_tokens = report.input_tokens.saturating_add(result.prompt_tokens);
                report.output_tokens = report
                    .output_tokens
                    .saturating_add(result.completion_tokens);
                report.estimated_cached_input_tokens = report
                    .estimated_cached_input_tokens
                    .saturating_add(estimated_cache_hit(
                        previous_prompt_tokens,
                        result.prompt_tokens,
                    ));
                report.reported_cached_input_tokens = report
                    .reported_cached_input_tokens
                    .saturating_add(result.cached_tokens.unwrap_or_default());
                previous_prompt_tokens = result.prompt_tokens;
                report.latencies.push(result.latency);

                if !turn.tool_call_latency.is_zero() {
                    tokio::time::sleep(turn.tool_call_latency).await;
                    report.tool_call_latency = report
                        .tool_call_latency
                        .saturating_add(turn.tool_call_latency);
                }

                append_model_and_environment(
                    &mut messages,
                    result.assistant_message,
                    plan.agent_id,
                    invocation,
                    &turn.environment_content,
                )?;
            }
            Err(err) => {
                let error = err.to_string();
                if let Some(sink) = &event_sink {
                    sink.write(&AgentRequestEvent::failed(
                        plan.agent_id,
                        invocation,
                        session_id.as_deref(),
                        request_start.elapsed(),
                        &error,
                    ))
                    .await?;
                }
                report.failed_requests += 1;
                report.failures.push(AgentFailureRecord {
                    agent_id: plan.agent_id,
                    invocation,
                    error,
                });
                return Ok(report);
            }
        }
    }

    report.completed = true;
    Ok(report)
}

fn build_request_body(
    config: &AgentLoopConfig,
    messages: &[Value],
    output_tokens: usize,
    agent_id: usize,
) -> Value {
    let mut body = json!({
        "model": config.model,
        "messages": messages,
        "tools": [{
            "type": "function",
            "function": {
                "name": "environment",
                "description": "Interact with the benchmark's synthetic environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": "The environment operation to perform."
                        }
                    },
                    "required": ["request"]
                }
            }
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "environment"}
        },
        "parallel_tool_calls": false
    });

    let (max_key, min_key) = output_token_field_names(config.sglang);
    if let Some(map) = body.as_object_mut() {
        map.insert(max_key.to_string(), json!(output_tokens));
        if !config.max_tokens_only {
            map.insert(min_key.to_string(), json!(output_tokens));
        }
        if let Some(prefix) = &config.user_prefix {
            map.insert("user".to_string(), json!(format!("{prefix}-{agent_id}")));
        }
        if let Some(temperature) = config.temperature {
            map.insert("temperature".to_string(), json!(temperature));
        }
        if !config.nvext_extra_fields.is_empty() {
            map.insert(
                "nvext".to_string(),
                json!({"extra_fields": &config.nvext_extra_fields}),
            );
        }
    }
    body
}

fn output_token_field_names(use_sglang: bool) -> (&'static str, &'static str) {
    if use_sglang {
        ("max_new_tokens", "min_new_tokens")
    } else {
        ("max_tokens", "min_tokens")
    }
}

async fn request_with_retries(
    client: &Client,
    config: &AgentLoopConfig,
    body: &Value,
    agent_headers: &HeaderMap,
) -> Result<RequestResult> {
    let start = Instant::now();
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        match single_attempt(client, config, body, agent_headers).await {
            Ok(mut result) => {
                result.latency = start.elapsed();
                return Ok(result);
            }
            Err(err) => {
                last_error = Some(err);
                if attempt < config.max_retries {
                    let multiplier = 1u32.checked_shl(attempt.min(31) as u32).unwrap_or(u32::MAX);
                    tokio::time::sleep(config.retry_delay.saturating_mul(multiplier)).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("request failed without an error")))
}

async fn single_attempt(
    client: &Client,
    config: &AgentLoopConfig,
    body: &Value,
    agent_headers: &HeaderMap,
) -> Result<RequestResult> {
    if config.verbose {
        println!("[AGENT REQUEST] {}", sanitize_request(body));
    }

    let mut headers = config.headers.clone();
    for (name, value) in agent_headers {
        headers.insert(name.clone(), value.clone());
    }
    let request = client.post(config.endpoint.clone()).headers(headers);
    let response = request.json(body).send().await?;
    let status = response.status();
    let bytes = response.bytes().await?;
    if !status.is_success() {
        return Err(anyhow!(
            "request failed ({}) {}",
            status,
            truncate_text(&String::from_utf8_lossy(&bytes), 500)
        ));
    }

    let payload: Value = serde_json::from_slice(&bytes)?;
    if config.verbose {
        println!("[AGENT RESPONSE] {}", sanitize_response(&payload));
    }
    parse_response_payload(&payload)
}

fn parse_response_payload(payload: &Value) -> Result<RequestResult> {
    let usage = payload
        .get("usage")
        .ok_or_else(|| anyhow!("response missing usage field"))?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("usage.prompt_tokens missing or not an integer"))?;
    let cached_tokens = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64);
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let assistant_message = payload
        .pointer("/choices/0/message")
        .cloned()
        .ok_or_else(|| anyhow!("response missing choices[0].message"))?;
    let worker_info = payload.pointer("/nvext/worker_id");
    let worker_id = worker_info.and_then(|info| {
        info.get("decode_worker_id")
            .or_else(|| info.get("prefill_worker_id"))
            .and_then(Value::as_u64)
    });
    let dp_rank = worker_info
        .and_then(|info| {
            info.get("decode_dp_rank")
                .or_else(|| info.get("prefill_dp_rank"))
                .and_then(Value::as_u64)
        })
        .and_then(|rank| u32::try_from(rank).ok());

    Ok(RequestResult {
        prompt_tokens,
        cached_tokens,
        completion_tokens,
        worker_id,
        dp_rank,
        assistant_message,
        latency: Duration::ZERO,
    })
}

fn render_agent_headers(config: &AgentLoopConfig, agent_id: usize) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    for template in &config.agent_header_templates {
        let (name, value) = template.render(agent_id)?;
        headers.insert(name, value);
    }
    Ok(headers)
}

fn append_model_and_environment(
    messages: &mut Vec<Value>,
    assistant_message: Value,
    agent_id: usize,
    invocation: usize,
    environment_content: &str,
) -> Result<()> {
    let mut assistant = normalize_assistant_message(assistant_message)?;
    let mut tool_call_ids = assistant
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|call| call.get("id").and_then(Value::as_str))
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if tool_call_ids.is_empty() {
        let tool_call_id = synthetic_tool_call_id(agent_id, invocation);
        let tool_call = json!({
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": "environment",
                "arguments": "{\"request\":\"continue\"}"
            }
        });
        assistant
            .as_object_mut()
            .ok_or_else(|| anyhow!("normalized assistant message is not an object"))?
            .insert("tool_calls".to_string(), json!([tool_call]));
        tool_call_ids.push(synthetic_tool_call_id(agent_id, invocation));
    }

    messages.push(assistant);
    for (index, tool_call_id) in tool_call_ids.into_iter().enumerate() {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": if index == 0 { environment_content } else { "" },
        }));
    }
    Ok(())
}

fn append_synthetic_turn(
    messages: &mut Vec<Value>,
    agent_id: usize,
    invocation: usize,
    model_content: &str,
    environment_content: &str,
) {
    let tool_call_id = synthetic_tool_call_id(agent_id, invocation);
    messages.push(json!({
        "role": "assistant",
        "content": model_content,
        "tool_calls": [{
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": "environment",
                "arguments": "{\"request\":\"continue\"}"
            }
        }]
    }));
    messages.push(json!({
        "role": "tool",
        "tool_call_id": synthetic_tool_call_id(agent_id, invocation),
        "content": environment_content,
    }));
}

fn normalize_assistant_message(message: Value) -> Result<Value> {
    let source = message
        .as_object()
        .ok_or_else(|| anyhow!("choices[0].message is not an object"))?;
    let mut normalized = Map::new();
    normalized.insert("role".to_string(), Value::String("assistant".to_string()));
    for key in ["content", "name", "tool_calls", "function_call", "refusal"] {
        if let Some(value) = source.get(key) {
            normalized.insert(key.to_string(), value.clone());
        }
    }
    if !normalized.contains_key("content") && !normalized.contains_key("tool_calls") {
        normalized.insert("content".to_string(), Value::Null);
    }
    Ok(Value::Object(normalized))
}

fn synthetic_tool_call_id(agent_id: usize, invocation: usize) -> String {
    format!("call_batchbench_{}_{}", agent_id, invocation)
}

fn sanitize_request(body: &Value) -> Value {
    let mut sanitized = body.clone();
    if let Some(messages) = sanitized.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            if let Some(content) = message.get_mut("content") {
                if let Some(text) = content.as_str() {
                    *content = Value::String(truncate_text(text, 50));
                }
            }
        }
    }
    sanitized
}

fn sanitize_response(payload: &Value) -> Value {
    let mut sanitized = payload.clone();
    if let Some(content) = sanitized
        .pointer_mut("/choices/0/message/content")
        .and_then(|value| value.as_str().map(str::to_string))
    {
        if let Some(value) = sanitized.pointer_mut("/choices/0/message/content") {
            *value = Value::String(truncate_text(&content, 50));
        }
    }
    sanitized
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        format!("{}...", trimmed.chars().take(max_chars).collect::<String>())
    }
}

fn rate(value: u64, duration_secs: f64) -> f64 {
    if duration_secs > 0.0 {
        value as f64 / duration_secs
    } else {
        0.0
    }
}

fn estimated_cache_hit(previous_prompt_tokens: u64, current_prompt_tokens: u64) -> u64 {
    previous_prompt_tokens.min(current_prompt_tokens)
}

fn percentile(sorted_latencies: &[Duration], quantile: f64) -> Option<Duration> {
    if sorted_latencies.is_empty() {
        return None;
    }
    let index = ((sorted_latencies.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted_latencies.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_and_lognormal_specs_validate() {
        assert!(SampleSpec::fixed(8).is_ok());
        assert!(SampleSpec::fixed(0).is_err());
        assert!(SampleSpec::log_normal_from_median(32.0, 0.5, Some(64)).is_ok());
        assert!(SampleSpec::log_normal_from_median(0.0, 0.5, None).is_err());
        assert!(SampleSpec::log_normal(1.0, 0.0, None).is_err());
    }

    #[test]
    fn fixed_samples_are_exact() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let spec = SampleSpec::fixed(13).unwrap();
        for _ in 0..10 {
            assert_eq!(spec.sample(&mut rng, "test").unwrap(), 13);
        }
    }

    #[test]
    fn tool_call_latency_is_optional_and_configurable() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            1,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap();
        assert!(config.tool_call_latency_ms.is_none());

        let config = config
            .with_tool_call_latency_ms(SampleSpec::fixed(250).unwrap())
            .unwrap();
        assert!(matches!(
            config.tool_call_latency_ms,
            Some(SampleSpec::Fixed(250))
        ));
    }

    #[test]
    fn invocation_barrier_requires_a_reachable_fixed_turn() {
        let fixed = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            2,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(5).unwrap(),
        )
        .unwrap();
        assert!(fixed.clone().with_barrier_after_invocation(3).is_ok());
        assert!(fixed.clone().with_barrier_after_invocation(0).is_err());
        assert!(fixed.with_barrier_after_invocation(5).is_err());

        let sampled = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            2,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::log_normal_from_median(5.0, 0.2, Some(10)).unwrap(),
        )
        .unwrap();
        assert!(sampled.with_barrier_after_invocation(3).is_err());
    }

    #[tokio::test]
    async fn invocation_barrier_unblocks_if_a_participant_drops() {
        let barrier = InvocationBarrier::new(2);
        let mut waiting = barrier.participant();
        let abandoned = barrier.participant();
        let task = tokio::spawn(async move { waiting.wait().await });
        tokio::task::yield_now().await;
        drop(abandoned);
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("cancelled barrier did not release")
            .unwrap();
    }

    #[test]
    fn lognormal_samples_are_seeded_and_capped() {
        let spec = SampleSpec::log_normal_from_median(100.0, 0.9, Some(40)).unwrap();
        let mut first = rand::rngs::StdRng::seed_from_u64(9);
        let mut second = rand::rngs::StdRng::seed_from_u64(9);
        let first_samples: Vec<_> = (0..20)
            .map(|_| spec.sample(&mut first, "test").unwrap())
            .collect();
        let second_samples: Vec<_> = (0..20)
            .map(|_| spec.sample(&mut second, "test").unwrap())
            .collect();
        assert_eq!(first_samples, second_samples);
        assert!(first_samples
            .iter()
            .all(|sample| *sample >= 1 && *sample <= 40));
    }

    #[test]
    fn appended_turn_preserves_model_message_and_adds_tool_result() {
        let mut messages = vec![json!({"role": "user", "content": "start"})];
        let assistant = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "environment", "arguments": "{\"request\":\"x\"}"}
            }],
            "reasoning_content": "server-specific field"
        });
        append_model_and_environment(&mut messages, assistant, 0, 1, "result").unwrap();

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_123");
        assert!(messages[1].get("reasoning_content").is_none());
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_123");
        assert_eq!(messages[2]["content"], "result");
    }

    #[test]
    fn content_only_response_gets_a_valid_synthetic_tool_call() {
        let mut messages = vec![json!({"role": "user", "content": "start"})];
        append_model_and_environment(
            &mut messages,
            json!({"role": "assistant", "content": "generated text"}),
            2,
            3,
            "environment",
        )
        .unwrap();

        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_batchbench_2_3");
        assert_eq!(messages[1]["content"], "generated text");
        assert_eq!(messages[2]["tool_call_id"], "call_batchbench_2_3");
    }

    #[test]
    fn cache_estimate_uses_only_the_previous_prompt() {
        let prompts = [100u64, 180, 250, 400];
        let estimate = prompts
            .windows(2)
            .map(|pair| estimated_cache_hit(pair[0], pair[1]))
            .sum::<u64>();
        assert_eq!(estimate, 530);
        assert_eq!(estimated_cache_hit(500, 450), 450);
    }

    #[test]
    fn request_body_contains_the_growing_history_and_turn_limit() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            1,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap();
        let messages = vec![
            json!({"role": "user", "content": "start"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "environment", "arguments": "{}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "result"}),
        ];

        let body = build_request_body(&config, &messages, 17, 0);
        assert_eq!(body["messages"], json!(messages));
        assert_eq!(body["max_tokens"], 17);
        assert_eq!(body["min_tokens"], 17);
        assert_eq!(
            body["tool_choice"]["function"]["name"],
            Value::String("environment".to_string())
        );
        assert!(body.get("user").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn request_body_can_set_greedy_sampling() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            1,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap()
        .with_temperature(0.0)
        .unwrap();

        let body = build_request_body(
            &config,
            &[json!({"role": "user", "content": "start"})],
            5,
            0,
        );
        assert_eq!(body["temperature"], 0.0);
    }

    #[test]
    fn request_body_can_omit_the_minimum_output_limit() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            1,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap()
        .with_max_tokens_only(true);

        let body = build_request_body(
            &config,
            &[json!({"role": "user", "content": "start"})],
            5,
            0,
        );
        assert_eq!(body["max_tokens"], 5);
        assert!(body.get("min_tokens").is_none());
    }

    #[test]
    fn user_prefix_stamps_a_per_agent_user_field() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            2,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap()
        .with_user_prefix("loadtest");
        let messages = vec![json!({"role": "user", "content": "start"})];

        let body = build_request_body(&config, &messages, 5, 1);
        assert_eq!(body["user"], Value::String("loadtest-1".to_string()));
    }

    #[test]
    fn agent_header_template_renders_a_distinct_session_per_agent() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            2,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap()
        .add_agent_header_template(
            AgentHeaderTemplate::try_new(DYNAMO_SESSION_HEADER, "scale-agent-{agent_id}").unwrap(),
        )
        .unwrap();

        let first = render_agent_headers(&config, 0).unwrap();
        let second = render_agent_headers(&config, 19).unwrap();
        assert_eq!(first.get(DYNAMO_SESSION_HEADER).unwrap(), "scale-agent-0");
        assert_eq!(second.get(DYNAMO_SESSION_HEADER).unwrap(), "scale-agent-19");
    }

    #[test]
    fn agent_header_template_requires_placeholder_and_unique_name() {
        assert!(AgentHeaderTemplate::try_new(DYNAMO_SESSION_HEADER, "one-session").is_err());

        let template = AgentHeaderTemplate::try_new("x-test-agent", "agent-{agent_id}").unwrap();
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            1,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap()
        .add_agent_header_template(template.clone())
        .unwrap();
        assert!(config.add_agent_header_template(template).is_err());
    }

    #[test]
    fn request_body_can_request_dynamo_routing_metadata() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            1,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap()
        .with_nvext_extra_fields(vec!["worker_id".to_string()])
        .unwrap();

        let body = build_request_body(
            &config,
            &[json!({"role": "user", "content": "start"})],
            5,
            0,
        );
        assert_eq!(body["nvext"]["extra_fields"], json!(["worker_id"]));
    }

    #[test]
    fn response_parser_extracts_cache_and_dynamo_target_metadata() {
        let result = parse_response_payload(&json!({
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 96}
            },
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "nvext": {
                "worker_id": {
                    "prefill_worker_id": 7,
                    "prefill_dp_rank": 0,
                    "decode_worker_id": 8,
                    "decode_dp_rank": 1
                }
            }
        }))
        .unwrap();

        assert_eq!(result.prompt_tokens, 120);
        assert_eq!(result.cached_tokens, Some(96));
        assert_eq!(result.completion_tokens, 8);
        assert_eq!(result.worker_id, Some(8));
        assert_eq!(result.dp_rank, Some(1));
    }
}
