use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokenizers::Tokenizer;
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::tokenizer_loader::load_tokenizer;

const SMG_ROUTING_KEY: HeaderName = HeaderName::from_static("x-smg-routing-key");
const SMG_TARGET_WORKER: HeaderName = HeaderName::from_static("x-smg-target-worker");

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
    pub user_prefix: Option<String>,
    pub dp_rank_perfect_routing_num: Option<usize>,
    pub request_timeout: Duration,
    pub max_retries: usize,
    pub retry_delay: Duration,
    pub headers: HeaderMap,
    pub verbose: bool,
    pub dry_run: bool,
    pub sglang: bool,
    pub ignore_eos: bool,
    pub user_tagging: bool,
    pub seed: Option<u64>,
    pub agent_plans_jsonl: Option<PathBuf>,
    pub max_active_agents: Option<usize>,
    pub replay_initial_overhead_tokens: usize,
    pub replay_turn_overhead_tokens: usize,
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
            user_prefix: None,
            dp_rank_perfect_routing_num: None,
            request_timeout: Duration::from_secs(60),
            max_retries: 2,
            retry_delay: Duration::from_millis(250),
            headers,
            verbose: false,
            dry_run: false,
            sglang: false,
            ignore_eos: false,
            user_tagging: true,
            seed: None,
            agent_plans_jsonl: None,
            max_active_agents: None,
            replay_initial_overhead_tokens: 0,
            replay_turn_overhead_tokens: 0,
        })
    }

    pub fn with_tokenizer_model(mut self, tokenizer_model: impl Into<String>) -> Self {
        self.tokenizer_model = tokenizer_model.into();
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

    pub fn with_ignore_eos(mut self, ignore_eos: bool) -> Self {
        self.ignore_eos = ignore_eos;
        self
    }

    pub fn with_user_tagging(mut self, user_tagging: bool) -> Self {
        self.user_tagging = user_tagging;
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

    pub fn with_dp_rank_perfect_routing(mut self, num_ranks: usize) -> Result<Self> {
        if num_ranks == 0 {
            return Err(anyhow!(
                "dp rank perfect routing rank count must be greater than zero"
            ));
        }
        self.dp_rank_perfect_routing_num = Some(num_ranks);
        Ok(self)
    }

    pub fn with_tool_call_latency_ms(mut self, latency_ms: SampleSpec) -> Result<Self> {
        latency_ms.validate("tool call latency")?;
        self.tool_call_latency_ms = Some(latency_ms);
        Ok(self)
    }

    pub fn with_agent_plans_jsonl(mut self, path: impl Into<PathBuf>) -> Self {
        self.agent_plans_jsonl = Some(path.into());
        self
    }

    pub fn with_max_active_agents(mut self, max_active_agents: usize) -> Result<Self> {
        if max_active_agents == 0 {
            return Err(anyhow!("max active agents must be greater than zero"));
        }
        self.max_active_agents = Some(max_active_agents);
        Ok(self)
    }

    pub fn with_replay_prompt_overhead(
        mut self,
        initial_overhead_tokens: usize,
        turn_overhead_tokens: usize,
    ) -> Self {
        self.replay_initial_overhead_tokens = initial_overhead_tokens;
        self.replay_turn_overhead_tokens = turn_overhead_tokens;
        self
    }

    pub fn add_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.insert(name, value);
        self
    }
}

#[derive(Clone, Debug)]
struct AgentTurnPlan {
    input_content_tokens: usize,
    target_prompt_tokens: usize,
    output_tokens: usize,
    environment_tokens: usize,
    environment_content: String,
    tool_call_latency: Duration,
    reset_prompt: Option<String>,
}

#[derive(Clone, Debug)]
struct AgentPlan {
    agent_id: usize,
    trajectory_id: String,
    user_tag: Option<String>,
    initial_prompt: String,
    turns: Vec<AgentTurnPlan>,
}

#[derive(Clone, Debug)]
pub struct AgentFailureRecord {
    pub agent_id: usize,
    pub trajectory_id: String,
    pub invocation: usize,
    pub error: String,
}

#[derive(Clone, Debug)]
pub struct AgentLifecycleRecord {
    pub agent_id: usize,
    pub trajectory_id: String,
    pub routing_slot: usize,
    pub admitted_at: Duration,
    pub finished_at: Duration,
    pub completed: bool,
}

#[derive(Clone, Debug)]
pub struct AgentBenchmarkReport {
    pub total_agents: usize,
    pub max_active_agents: usize,
    pub completed_agents: usize,
    pub planned_tool_invocations: u64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub estimated_cached_input_tokens: u64,
    pub total_tool_call_latency: Duration,
    pub total_duration: Duration,
    pub last_agent_admitted_at: Duration,
    pub final_drain_duration: Duration,
    pub input_tokens_per_second: f64,
    pub output_tokens_per_second: f64,
    pub requests_per_second: f64,
    pub latency_p50: Option<Duration>,
    pub latency_p90: Option<Duration>,
    pub latency_p99: Option<Duration>,
    pub agent_end_to_end_latency_p50: Option<Duration>,
    pub agent_end_to_end_latency_p90: Option<Duration>,
    pub agent_end_to_end_latency_p99: Option<Duration>,
    pub failures: Vec<AgentFailureRecord>,
    pub agent_lifecycles: Vec<AgentLifecycleRecord>,
}

#[derive(Debug)]
struct AgentWorkerReport {
    agent_id: usize,
    trajectory_id: String,
    routing_slot: usize,
    admitted_at: Duration,
    completed: bool,
    end_to_end_latency: Duration,
    successful_requests: u64,
    failed_requests: u64,
    input_tokens: u64,
    output_tokens: u64,
    estimated_cached_input_tokens: u64,
    tool_call_latency: Duration,
    latencies: Vec<Duration>,
    failures: Vec<AgentFailureRecord>,
    dry_run_records: Vec<DryRunRecord>,
}

impl AgentWorkerReport {
    fn new(
        agent_id: usize,
        trajectory_id: String,
        routing_slot: usize,
        admitted_at: Duration,
    ) -> Self {
        Self {
            agent_id,
            trajectory_id,
            routing_slot,
            admitted_at,
            completed: false,
            end_to_end_latency: Duration::ZERO,
            successful_requests: 0,
            failed_requests: 0,
            input_tokens: 0,
            output_tokens: 0,
            estimated_cached_input_tokens: 0,
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
    trajectory_id: String,
    routing_slot: usize,
    admitted_at: Duration,
    invocation: usize,
    input_content_tokens: usize,
    target_prompt_tokens: usize,
    output_tokens: usize,
    environment_tokens: usize,
    tool_call_latency_ms: usize,
    reset_before: bool,
}

pub(crate) const TRAJECTORY_PLAN_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrajectoryPlanSpec {
    schema_version: u32,
    trajectory_id: String,
    requests: Vec<TrajectoryRequestSpec>,
    #[serde(default, rename = "metadata")]
    _metadata: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrajectoryRequestSpec {
    prompt_tokens: usize,
    output_tokens: usize,
    #[serde(default)]
    reset_before: bool,
    #[serde(default)]
    delay_after_ms: u64,
}

#[derive(Debug)]
struct RequestResult {
    prompt_tokens: u64,
    completion_tokens: u64,
    assistant_message: Value,
    latency: Duration,
}

#[derive(Debug)]
struct RollingAdmission<T> {
    pending: VecDeque<T>,
    max_active: usize,
}

impl<T> RollingAdmission<T> {
    fn new(plans: Vec<T>, max_active: usize) -> Result<Self> {
        if max_active == 0 {
            return Err(anyhow!("max active agents must be greater than zero"));
        }
        let effective_max_active = max_active.min(plans.len());
        Ok(Self {
            pending: VecDeque::from(plans),
            max_active: effective_max_active,
        })
    }

    fn initial_admissions(&mut self) -> Vec<(usize, T)> {
        (0..self.max_active)
            .map(|routing_slot| {
                let plan = self
                    .pending
                    .pop_front()
                    .expect("validated admission limit cannot exceed plan count");
                (routing_slot, plan)
            })
            .collect()
    }

    fn replacement(&mut self, routing_slot: usize) -> Option<(usize, T)> {
        self.pending.pop_front().map(|plan| (routing_slot, plan))
    }
}

/// Run independent, stateful agent loops concurrently.
///
/// Each agent begins with a separately generated user message. Every model response and synthetic
/// tool response is appended to that agent's messages before its next request, preserving the
/// previous request as a prefix for server-side KV-cache testing.
pub async fn run_agent_benchmark(config: AgentLoopConfig) -> Result<AgentBenchmarkReport> {
    let plans = build_agent_plans(&config)?;
    let total_agents = plans.len();
    let requested_max_active_agents = config.max_active_agents.unwrap_or(total_agents);
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
    let config = std::sync::Arc::new(config);
    let start = Instant::now();

    let mut join_set = JoinSet::new();
    let mut admission = RollingAdmission::new(plans, requested_max_active_agents)?;
    let max_active_agents = admission.max_active;
    for (routing_slot, plan) in admission.initial_admissions() {
        spawn_agent(
            &mut join_set,
            plan,
            routing_slot,
            Duration::ZERO,
            &client,
            &config,
        );
    }
    let mut last_agent_admitted_at = Duration::ZERO;

    let mut completed_agents = 0usize;
    let mut successful_requests = 0u64;
    let mut failed_requests = 0u64;
    let mut total_input_tokens = 0u64;
    let mut total_output_tokens = 0u64;
    let mut estimated_cached_input_tokens = 0u64;
    let mut total_tool_call_latency = Duration::ZERO;
    let mut latencies = Vec::new();
    let mut agent_end_to_end_latencies = Vec::new();
    let mut failures = Vec::new();
    let mut dry_run_records = Vec::new();
    let mut agent_lifecycles = Vec::with_capacity(total_agents);

    while let Some(joined) = join_set.join_next().await {
        let worker = joined.map_err(|err| anyhow!("agent worker task failed: {}", err))??;
        let finished_at = start.elapsed();
        let routing_slot = worker.routing_slot;
        agent_lifecycles.push(AgentLifecycleRecord {
            agent_id: worker.agent_id,
            trajectory_id: worker.trajectory_id.clone(),
            routing_slot,
            admitted_at: worker.admitted_at,
            finished_at,
            completed: worker.completed,
        });
        if worker.completed {
            completed_agents += 1;
            agent_end_to_end_latencies.push(worker.end_to_end_latency);
        }
        successful_requests = successful_requests.saturating_add(worker.successful_requests);
        failed_requests = failed_requests.saturating_add(worker.failed_requests);
        total_input_tokens = total_input_tokens.saturating_add(worker.input_tokens);
        total_output_tokens = total_output_tokens.saturating_add(worker.output_tokens);
        estimated_cached_input_tokens =
            estimated_cached_input_tokens.saturating_add(worker.estimated_cached_input_tokens);
        total_tool_call_latency = total_tool_call_latency.saturating_add(worker.tool_call_latency);
        latencies.extend(worker.latencies);
        failures.extend(worker.failures);
        dry_run_records.extend(worker.dry_run_records);

        if let Some((routing_slot, plan)) = admission.replacement(routing_slot) {
            let admitted_at = start.elapsed();
            last_agent_admitted_at = admitted_at;
            spawn_agent(
                &mut join_set,
                plan,
                routing_slot,
                admitted_at,
                &client,
                &config,
            );
        }
    }

    if config.dry_run {
        dry_run_records.sort_by_key(|record| (record.agent_id, record.invocation));
        for record in dry_run_records {
            println!(
                "[DRY-RUN] agent={} invocation={} input_content_tokens={} output_tokens={} environment_tokens={} tool_call_latency_ms={} trajectory_id={} routing_slot={} admitted_at_ms={} target_prompt_tokens={} reset_before={}",
                record.agent_id,
                record.invocation,
                record.input_content_tokens,
                record.output_tokens,
                record.environment_tokens,
                record.tool_call_latency_ms,
                record.trajectory_id,
                record.routing_slot,
                record.admitted_at.as_millis(),
                record.target_prompt_tokens,
                record.reset_before
            );
        }
    }

    let total_duration = start.elapsed();
    let final_drain_duration = total_duration.saturating_sub(last_agent_admitted_at);
    latencies.sort();
    agent_end_to_end_latencies.sort();
    failures.sort_by_key(|failure| (failure.agent_id, failure.invocation));
    agent_lifecycles.sort_by_key(|record| record.agent_id);
    let total_requests = successful_requests.saturating_add(failed_requests);
    let duration_secs = total_duration.as_secs_f64();

    Ok(AgentBenchmarkReport {
        total_agents,
        max_active_agents,
        completed_agents,
        planned_tool_invocations,
        total_requests,
        successful_requests,
        failed_requests,
        total_input_tokens,
        total_output_tokens,
        estimated_cached_input_tokens,
        total_tool_call_latency,
        total_duration,
        last_agent_admitted_at,
        final_drain_duration,
        input_tokens_per_second: rate(total_input_tokens, duration_secs),
        output_tokens_per_second: rate(total_output_tokens, duration_secs),
        requests_per_second: rate(total_requests, duration_secs),
        latency_p50: percentile(&latencies, 0.50),
        latency_p90: percentile(&latencies, 0.90),
        latency_p99: percentile(&latencies, 0.99),
        agent_end_to_end_latency_p50: percentile(&agent_end_to_end_latencies, 0.50),
        agent_end_to_end_latency_p90: percentile(&agent_end_to_end_latencies, 0.90),
        agent_end_to_end_latency_p99: percentile(&agent_end_to_end_latencies, 0.99),
        failures,
        agent_lifecycles,
    })
}

fn spawn_agent(
    join_set: &mut JoinSet<Result<AgentWorkerReport>>,
    plan: AgentPlan,
    routing_slot: usize,
    admitted_at: Duration,
    client: &Client,
    config: &std::sync::Arc<AgentLoopConfig>,
) {
    let client = client.clone();
    let config = std::sync::Arc::clone(config);
    join_set.spawn(async move { run_agent(plan, routing_slot, admitted_at, client, config).await });
}

fn build_agent_plans(config: &AgentLoopConfig) -> Result<Vec<AgentPlan>> {
    let tokenizer = load_tokenizer(&config.tokenizer_model)?;
    let root_seed = match config.seed {
        Some(seed) => seed,
        None => rand::thread_rng().gen(),
    };

    if let Some(path) = config.agent_plans_jsonl.as_deref() {
        let specs = load_trajectory_plan_specs(path)?;
        return build_replay_agent_plans(config, &tokenizer, root_seed, specs);
    }

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
        let mut target_prompt_tokens = initial_prompt_tokens;
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
                input_content_tokens: target_prompt_tokens,
                target_prompt_tokens,
                output_tokens,
                environment_tokens,
                environment_content,
                tool_call_latency: Duration::from_millis(tool_call_latency_ms),
                reset_prompt: None,
            });
            target_prompt_tokens = target_prompt_tokens
                .checked_add(output_tokens)
                .and_then(|tokens| tokens.checked_add(environment_tokens))
                .ok_or_else(|| anyhow!("sampled agent prompt-token count overflowed usize"))?;
        }

        plans.push(AgentPlan {
            agent_id,
            trajectory_id: format!("sampled-{agent_id}"),
            user_tag: generate_user_tag(
                config.user_tagging,
                config.user_prefix.as_deref(),
                agent_id,
            ),
            initial_prompt,
            turns,
        });
    }
    Ok(plans)
}

fn load_trajectory_plan_specs(path: &Path) -> Result<Vec<TrajectoryPlanSpec>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open trajectory plan file {}", path.display()))?;
    parse_trajectory_plan_specs(BufReader::new(file), &path.display().to_string())
}

fn parse_trajectory_plan_specs<R: BufRead>(
    reader: R,
    source: &str,
) -> Result<Vec<TrajectoryPlanSpec>> {
    let mut specs = Vec::new();
    let mut trajectory_ids = HashSet::new();

    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.with_context(|| format!("failed to read {source}:{line_number}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let spec: TrajectoryPlanSpec = serde_json::from_str(&line)
            .with_context(|| format!("invalid trajectory plan at {source}:{line_number}"))?;
        validate_trajectory_plan_spec(&spec)
            .with_context(|| format!("invalid trajectory plan at {source}:{line_number}"))?;
        if !trajectory_ids.insert(spec.trajectory_id.clone()) {
            return Err(anyhow!(
                "invalid trajectory plan at {source}:{line_number}: duplicate trajectory_id {:?}",
                spec.trajectory_id
            ));
        }
        specs.push(spec);
    }

    if specs.is_empty() {
        return Err(anyhow!(
            "trajectory plan file {source} contains no trajectory plans"
        ));
    }
    Ok(specs)
}

fn validate_trajectory_plan_spec(spec: &TrajectoryPlanSpec) -> Result<()> {
    if spec.schema_version != TRAJECTORY_PLAN_SCHEMA_VERSION {
        return Err(anyhow!(
            "unsupported schema_version {}; expected {}",
            spec.schema_version,
            TRAJECTORY_PLAN_SCHEMA_VERSION
        ));
    }
    if spec.trajectory_id.trim().is_empty() {
        return Err(anyhow!("trajectory_id must not be empty"));
    }
    if spec.requests.is_empty() {
        return Err(anyhow!("requests must contain at least one request"));
    }

    for (request_index, request) in spec.requests.iter().enumerate() {
        if request.prompt_tokens == 0 {
            return Err(anyhow!(
                "request {} prompt_tokens must be greater than zero",
                request_index + 1
            ));
        }
        if request.output_tokens == 0 {
            return Err(anyhow!(
                "request {} output_tokens must be greater than zero",
                request_index + 1
            ));
        }
        if request_index == 0 && request.reset_before {
            return Err(anyhow!("the first request cannot set reset_before"));
        }
        if request_index > 0 {
            let previous = &spec.requests[request_index - 1];
            inferred_environment_tokens(previous, request, 0).with_context(|| {
                format!(
                    "request {} prompt_tokens ({}) cannot follow the preceding prompt and output; set reset_before=true to model compaction or reset",
                    request_index + 1,
                    request.prompt_tokens
                )
            })?;
        }
    }
    Ok(())
}

fn inferred_environment_tokens(
    current: &TrajectoryRequestSpec,
    next: &TrajectoryRequestSpec,
    transition_overhead_tokens: usize,
) -> Result<usize> {
    if next.reset_before {
        return Ok(0);
    }
    let current_context = current
        .prompt_tokens
        .checked_add(current.output_tokens)
        .ok_or_else(|| anyhow!("trajectory prompt-token count overflowed usize"))?;
    let current_context = current_context
        .checked_add(transition_overhead_tokens)
        .ok_or_else(|| anyhow!("trajectory prompt-token count overflowed usize"))?;
    next.prompt_tokens
        .checked_sub(current_context)
        .ok_or_else(|| {
            anyhow!(
                "next prompt ({}) is smaller than current prompt plus output ({current_context})",
                next.prompt_tokens
            )
        })
}

fn build_replay_agent_plans(
    config: &AgentLoopConfig,
    tokenizer: &Tokenizer,
    root_seed: u64,
    specs: Vec<TrajectoryPlanSpec>,
) -> Result<Vec<AgentPlan>> {
    let mut plans = Vec::with_capacity(specs.len());
    for (agent_id, spec) in specs.into_iter().enumerate() {
        let mut text_rng = stream_rng(root_seed, agent_id, 4);
        let initial_prompt_tokens = spec.requests[0]
            .prompt_tokens
            .checked_sub(config.replay_initial_overhead_tokens)
            .filter(|tokens| *tokens > 0)
            .ok_or_else(|| {
                anyhow!(
                    "trajectory {:?} first prompt ({}) must exceed replay initial overhead ({})",
                    spec.trajectory_id,
                    spec.requests[0].prompt_tokens,
                    config.replay_initial_overhead_tokens
                )
            })?;
        let initial_prompt =
            generate_synthetic_text(tokenizer, initial_prompt_tokens, &mut text_rng)?;
        let mut turns = Vec::with_capacity(spec.requests.len());
        let mut current_input_content_tokens = initial_prompt_tokens;

        for (request_index, request) in spec.requests.iter().enumerate() {
            let reset_prompt = if request.reset_before {
                let reset_prompt_tokens = request
                    .prompt_tokens
                    .checked_sub(config.replay_initial_overhead_tokens)
                    .filter(|tokens| *tokens > 0)
                    .ok_or_else(|| {
                        anyhow!(
                            "trajectory {:?} request {} prompt ({}) must exceed replay initial overhead ({})",
                            spec.trajectory_id,
                            request_index + 1,
                            request.prompt_tokens,
                            config.replay_initial_overhead_tokens
                        )
                    })?;
                current_input_content_tokens = reset_prompt_tokens;
                Some(generate_synthetic_text(
                    tokenizer,
                    reset_prompt_tokens,
                    &mut text_rng,
                )?)
            } else {
                None
            };
            let environment_tokens = match spec.requests.get(request_index + 1) {
                Some(next) => inferred_environment_tokens(
                    request,
                    next,
                    config.replay_turn_overhead_tokens,
                )
                .with_context(|| {
                    format!(
                        "trajectory {:?} request {} cannot satisfy replay turn overhead {}; lower the overhead or set reset_before=true",
                        spec.trajectory_id,
                        request_index + 2,
                        config.replay_turn_overhead_tokens
                    )
                })?,
                None => 0,
            };
            let environment_content =
                generate_synthetic_text(tokenizer, environment_tokens, &mut text_rng)?;
            turns.push(AgentTurnPlan {
                input_content_tokens: current_input_content_tokens,
                target_prompt_tokens: request.prompt_tokens,
                output_tokens: request.output_tokens,
                environment_tokens,
                environment_content,
                tool_call_latency: Duration::from_millis(request.delay_after_ms),
                reset_prompt,
            });
            if spec
                .requests
                .get(request_index + 1)
                .is_some_and(|next| !next.reset_before)
            {
                current_input_content_tokens = current_input_content_tokens
                    .checked_add(request.output_tokens)
                    .and_then(|tokens| tokens.checked_add(environment_tokens))
                    .ok_or_else(|| {
                        anyhow!(
                            "trajectory {:?} generated content-token count overflowed usize",
                            spec.trajectory_id
                        )
                    })?;
            }
        }

        plans.push(AgentPlan {
            agent_id,
            trajectory_id: spec.trajectory_id,
            user_tag: generate_user_tag(
                config.user_tagging,
                config.user_prefix.as_deref(),
                agent_id,
            ),
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
    if target_tokens == 0 {
        return Ok(String::new());
    }
    let vocab_size = tokenizer.get_vocab_size(false);
    if vocab_size == 0 {
        return Err(anyhow!("tokenizer vocabulary is empty"));
    }

    let special_ids: HashSet<u32> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter_map(|(id, token)| token.special.then_some(id))
        .collect();
    let special_base_ids = special_ids
        .iter()
        .filter(|id| (**id as usize) < vocab_size)
        .count();
    if special_base_ids == vocab_size {
        return Err(anyhow!(
            "tokenizer vocabulary contains no non-special tokens"
        ));
    }

    let random_ids = |count: usize, rng: &mut R| -> Vec<u32> {
        let mut ids = Vec::with_capacity(count);
        while ids.len() < count {
            let id = rng.gen_range(0..vocab_size) as u32;
            if !special_ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    };
    let mut text = tokenizer
        .decode(&random_ids(target_tokens, rng), false)
        .map_err(|err| anyhow!("failed to decode synthetic tokens: {}", err))?;
    let mut actual_tokens = 0usize;

    for _ in 0..32 {
        let encoding = tokenizer
            .encode(text.as_str(), false)
            .map_err(|err| anyhow!("failed to re-encode synthetic text: {}", err))?;
        actual_tokens = encoding.len();
        let encoded_special = encoding.get_ids().iter().any(|id| special_ids.contains(id));
        if encoded_special {
            text = tokenizer
                .decode(&random_ids(target_tokens, rng), false)
                .map_err(|err| anyhow!("failed to replace synthetic special tokens: {}", err))?;
            continue;
        }
        if actual_tokens == target_tokens && !text.is_empty() {
            return Ok(text);
        }

        if actual_tokens > target_tokens {
            let end_offset = encoding.get_offsets()[target_tokens - 1].1;
            if end_offset > 0 && text.is_char_boundary(end_offset) {
                text.truncate(end_offset);
            } else {
                text = tokenizer
                    .decode(&encoding.get_ids()[..target_tokens], false)
                    .map_err(|err| anyhow!("failed to trim synthetic text: {}", err))?;
            }
        } else {
            let missing = target_tokens - actual_tokens;
            let extra = tokenizer
                .decode(&random_ids(missing.max(1), rng), false)
                .map_err(|err| anyhow!("failed to extend synthetic text: {}", err))?;
            if !text.ends_with(char::is_whitespace) {
                text.push(' ');
            }
            text.push_str(&extra);
        }
    }

    Err(anyhow!(
        "failed to generate synthetic text with exactly {target_tokens} tokens after 32 attempts (last count: {actual_tokens})"
    ))
}

async fn run_agent(
    plan: AgentPlan,
    routing_slot: usize,
    admitted_at: Duration,
    client: Client,
    config: std::sync::Arc<AgentLoopConfig>,
) -> Result<AgentWorkerReport> {
    let agent_start = Instant::now();
    let mut report = AgentWorkerReport::new(
        plan.agent_id,
        plan.trajectory_id.clone(),
        routing_slot,
        admitted_at,
    );
    let mut messages = vec![json!({
        "role": "user",
        "content": plan.initial_prompt,
    })];
    let mut previous_prompt_tokens = 0u64;

    for (turn_index, turn) in plan.turns.iter().enumerate() {
        let invocation = turn_index + 1;
        if let Some(reset_prompt) = turn.reset_prompt.as_deref() {
            messages = vec![json!({
                "role": "user",
                "content": reset_prompt,
            })];
            previous_prompt_tokens = 0;
        }
        if config.dry_run {
            report.dry_run_records.push(DryRunRecord {
                agent_id: plan.agent_id,
                trajectory_id: plan.trajectory_id.clone(),
                routing_slot,
                admitted_at,
                invocation,
                input_content_tokens: turn.input_content_tokens,
                target_prompt_tokens: turn.target_prompt_tokens,
                output_tokens: turn.output_tokens,
                environment_tokens: turn.environment_tokens,
                tool_call_latency_ms: turn.tool_call_latency.as_millis() as usize,
                reset_before: turn.reset_prompt.is_some(),
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

        let body = build_request_body(
            &config,
            &messages,
            turn.output_tokens,
            plan.user_tag.as_deref(),
        );
        match request_with_retries(
            &client,
            &config,
            &body,
            plan.user_tag.as_deref(),
            routing_slot,
        )
        .await
        {
            Ok(result) => {
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
                previous_prompt_tokens = result.prompt_tokens;
                report.latencies.push(result.latency);

                if !turn.tool_call_latency.is_zero() {
                    tokio::time::sleep(turn.tool_call_latency).await;
                    report.tool_call_latency = report
                        .tool_call_latency
                        .saturating_add(turn.tool_call_latency);
                }

                if let Err(err) = append_model_and_environment(
                    &mut messages,
                    result.assistant_message,
                    plan.agent_id,
                    invocation,
                    &turn.environment_content,
                ) {
                    report.failures.push(AgentFailureRecord {
                        agent_id: plan.agent_id,
                        trajectory_id: plan.trajectory_id.clone(),
                        invocation,
                        error: format!("failed to append model response: {err:#}"),
                    });
                    report.end_to_end_latency = agent_start.elapsed();
                    return Ok(report);
                }
            }
            Err(err) => {
                report.failed_requests += 1;
                report.failures.push(AgentFailureRecord {
                    agent_id: plan.agent_id,
                    trajectory_id: plan.trajectory_id.clone(),
                    invocation,
                    error: err.to_string(),
                });
                report.end_to_end_latency = agent_start.elapsed();
                return Ok(report);
            }
        }
    }

    report.completed = true;
    report.end_to_end_latency = agent_start.elapsed();
    Ok(report)
}

fn generate_user_tag(enabled: bool, prefix: Option<&str>, agent_id: usize) -> Option<String> {
    enabled.then(|| match prefix {
        Some(prefix) => format!("{prefix}-{agent_id}"),
        None => Uuid::new_v4().to_string(),
    })
}

fn build_request_body(
    config: &AgentLoopConfig,
    messages: &[Value],
    output_tokens: usize,
    user_tag: Option<&str>,
) -> Value {
    let mut body = json!({
        "model": config.model,
        "messages": messages
    });

    let (max_key, min_key) = output_token_field_names(config.sglang);
    if let Some(map) = body.as_object_mut() {
        map.insert(max_key.to_string(), json!(output_tokens));
        map.insert(min_key.to_string(), json!(output_tokens));
        if config.ignore_eos {
            map.insert("ignore_eos".to_string(), json!(true));
        }
        if let Some(user_tag) = user_tag {
            map.insert("user".to_string(), json!(user_tag));
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
    user_tag: Option<&str>,
    routing_slot: usize,
) -> Result<RequestResult> {
    let start = Instant::now();
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        match single_attempt(client, config, body, user_tag, routing_slot).await {
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
    user_tag: Option<&str>,
    routing_slot: usize,
) -> Result<RequestResult> {
    if config.verbose {
        println!("[AGENT REQUEST] {}", sanitize_request(body));
    }

    let request = client
        .post(config.endpoint.clone())
        .headers(build_request_headers(config, user_tag, routing_slot)?);
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
    let usage = payload
        .get("usage")
        .ok_or_else(|| anyhow!("response missing usage field"))?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("usage.prompt_tokens missing or not an integer"))?;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let assistant_message = payload
        .pointer("/choices/0/message")
        .cloned()
        .ok_or_else(|| anyhow!("response missing choices[0].message"))?;

    Ok(RequestResult {
        prompt_tokens,
        completion_tokens,
        assistant_message,
        latency: Duration::ZERO,
    })
}

fn build_request_headers(
    config: &AgentLoopConfig,
    user_tag: Option<&str>,
    routing_slot: usize,
) -> Result<HeaderMap> {
    let mut headers = config.headers.clone();
    if let Some(user_tag) = user_tag {
        let value = HeaderValue::from_str(user_tag)
            .context("failed to build X-SMG-Routing-Key header from user tag")?;
        headers.insert(SMG_ROUTING_KEY, value);
    }
    if let Some(num_ranks) = config.dp_rank_perfect_routing_num {
        let target_worker = (routing_slot % num_ranks).to_string();
        let value = HeaderValue::from_str(&target_worker)
            .context("failed to build X-SMG-Target-Worker header from agent rank")?;
        headers.insert(SMG_TARGET_WORKER, value);
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

    messages.push(assistant);
    messages.push(json!({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": environment_content,
    }));
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
    // Treat generated output as opaque assistant state. Provider-generated tool calls are
    // deliberately discarded; BatchBench adds its own valid synthetic tool-call envelope.
    for key in ["content", "reasoning_content", "name", "refusal"] {
        if let Some(value) = source.get(key) {
            normalized.insert(key.to_string(), value.clone());
        }
    }
    if !normalized.contains_key("content") {
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
    fn synthetic_text_reencodes_to_the_requested_token_count() {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::AddedToken;

        let vocab = [
            ("[UNK]".to_string(), 0),
            ("alpha".to_string(), 1),
            ("beta".to_string(), 2),
            ("gamma".to_string(), 3),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.add_special_tokens(&[AddedToken::from("beta", true)]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);

        let text = generate_synthetic_text(&tokenizer, 16, &mut rng).unwrap();
        let encoding = tokenizer.encode(text, false).unwrap();
        assert_eq!(encoding.len(), 16);
        let special_ids: HashSet<u32> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| token.special.then_some(id))
            .collect();
        assert!(encoding
            .get_ids()
            .iter()
            .all(|id| !special_ids.contains(id)));
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
    fn appended_turn_preserves_output_but_replaces_model_tool_calls() {
        let mut messages = vec![json!({"role": "user", "content": "start"})];
        let assistant = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "unexpected", "arguments": "{\"unterminated\":"}
            }],
            "reasoning_content": "server-specific field"
        });
        append_model_and_environment(&mut messages, assistant, 0, 1, "result").unwrap();

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_batchbench_0_1");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            "{\"request\":\"continue\"}"
        );
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["name"],
            "environment"
        );
        assert_eq!(messages[1]["reasoning_content"], "server-specific field");
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_batchbench_0_1");
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

        let body = build_request_body(&config, &messages, 17, Some("agent-user-tag"));
        assert_eq!(body["messages"], json!(messages));
        assert_eq!(body["max_tokens"], 17);
        assert_eq!(body["min_tokens"], 17);
        assert_eq!(body["user"], "agent-user-tag");
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn user_tags_are_uuid_v4_values_and_can_be_disabled() {
        let first = generate_user_tag(true, None, 0).unwrap();
        let second = generate_user_tag(true, None, 1).unwrap();
        assert_ne!(first, second);
        assert_eq!(Uuid::parse_str(&first).unwrap().get_version_num(), 4);
        assert_eq!(generate_user_tag(false, None, 0), None);
    }

    #[test]
    fn user_prefix_stamps_a_deterministic_per_agent_user_field() {
        assert_eq!(
            generate_user_tag(true, Some("loadtest"), 1).as_deref(),
            Some("loadtest-1")
        );
        assert_eq!(generate_user_tag(false, Some("loadtest"), 1), None);
    }

    #[test]
    fn user_tag_stamps_the_same_per_agent_routing_header() {
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
        let user_tag = generate_user_tag(true, Some("user"), 123).unwrap();
        let headers = build_request_headers(&config, Some(&user_tag), 123).unwrap();

        assert_eq!(user_tag, "user-123");
        assert_eq!(headers.get(SMG_ROUTING_KEY).unwrap(), user_tag.as_str());
    }

    #[test]
    fn routing_header_is_omitted_when_user_tagging_is_disabled() {
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
        let headers = build_request_headers(&config, None, 0).unwrap();

        assert!(!headers.contains_key(SMG_ROUTING_KEY));
    }

    #[test]
    fn perfect_routing_targets_routing_slot_modulo_rank_count() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            10,
            SampleSpec::fixed(8).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap()
        .with_user_tagging(false)
        .with_dp_rank_perfect_routing(8)
        .unwrap();

        for agent_id in 0..10 {
            let headers = build_request_headers(&config, None, agent_id).unwrap();
            assert_eq!(
                headers.get(SMG_TARGET_WORKER).unwrap(),
                (agent_id % 8).to_string().as_str()
            );
            assert!(!headers.contains_key(SMG_ROUTING_KEY));
        }
    }

    #[test]
    fn perfect_routing_rejects_zero_ranks() {
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

        assert!(config.with_dp_rank_perfect_routing(0).is_err());
    }

    #[test]
    fn request_body_omits_disabled_user_tag() {
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
        let messages = vec![json!({"role": "user", "content": "start"})];
        let body = build_request_body(&config, &messages, 4, None);
        assert!(body.get("user").is_none());
    }

    #[test]
    fn request_body_can_force_generation_past_eos() {
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
        .with_ignore_eos(true);
        let messages = vec![json!({"role": "user", "content": "start"})];
        let body = build_request_body(&config, &messages, 4, None);
        assert_eq!(body["ignore_eos"], true);
    }

    #[test]
    fn trajectory_jsonl_preserves_request_order_and_blank_lines() {
        let input = r#"
{"schema_version":1,"trajectory_id":"alpha","requests":[{"prompt_tokens":100,"output_tokens":20},{"prompt_tokens":150,"output_tokens":10,"delay_after_ms":25}]}

{"schema_version":1,"trajectory_id":"beta","requests":[{"prompt_tokens":80,"output_tokens":5}]}
"#;
        let specs = parse_trajectory_plan_specs(std::io::Cursor::new(input), "fixture").unwrap();

        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].trajectory_id, "alpha");
        assert_eq!(specs[0].requests[0].prompt_tokens, 100);
        assert_eq!(specs[0].requests[1].prompt_tokens, 150);
        assert_eq!(specs[0].requests[1].delay_after_ms, 25);
        assert_eq!(specs[1].trajectory_id, "beta");
    }

    #[test]
    fn trajectory_jsonl_accepts_explicit_compaction_reset() {
        let input = r#"{"schema_version":1,"trajectory_id":"reset","requests":[{"prompt_tokens":100,"output_tokens":20},{"prompt_tokens":60,"output_tokens":10,"reset_before":true}]}"#;
        let specs = parse_trajectory_plan_specs(std::io::Cursor::new(input), "fixture").unwrap();
        assert!(specs[0].requests[1].reset_before);
    }

    #[test]
    fn trajectory_environment_growth_is_derived_from_adjacent_requests() {
        let current = TrajectoryRequestSpec {
            prompt_tokens: 100,
            output_tokens: 20,
            reset_before: false,
            delay_after_ms: 0,
        };
        let next = TrajectoryRequestSpec {
            prompt_tokens: 150,
            output_tokens: 10,
            reset_before: false,
            delay_after_ms: 0,
        };
        assert_eq!(inferred_environment_tokens(&current, &next, 0).unwrap(), 30);
        assert_eq!(
            inferred_environment_tokens(&current, &next, 10).unwrap(),
            20
        );

        let reset = TrajectoryRequestSpec {
            prompt_tokens: 60,
            output_tokens: 10,
            reset_before: true,
            delay_after_ms: 0,
        };
        assert_eq!(
            inferred_environment_tokens(&current, &reset, 10).unwrap(),
            0
        );
    }

    #[test]
    fn replay_compiler_applies_overhead_and_reset_boundaries() {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        let vocab = [
            ("[UNK]".to_string(), 0),
            ("alpha".to_string(), 1),
            ("beta".to_string(), 2),
            ("gamma".to_string(), 3),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));

        let input = r#"{"schema_version":1,"trajectory_id":"shape","requests":[{"prompt_tokens":100,"output_tokens":20},{"prompt_tokens":150,"output_tokens":10,"delay_after_ms":25},{"prompt_tokens":60,"output_tokens":5,"reset_before":true}]}"#;
        let specs = parse_trajectory_plan_specs(std::io::Cursor::new(input), "fixture").unwrap();
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
        .with_replay_prompt_overhead(5, 10);

        let plans = build_replay_agent_plans(&config, &tokenizer, 42, specs).unwrap();
        let plan = &plans[0];
        assert_eq!(plan.trajectory_id, "shape");
        assert_eq!(
            tokenizer
                .encode(plan.initial_prompt.as_str(), false)
                .unwrap()
                .len(),
            95
        );
        assert_eq!(plan.turns.len(), 3);
        assert_eq!(plan.turns[0].input_content_tokens, 95);
        assert_eq!(plan.turns[0].target_prompt_tokens, 100);
        assert_eq!(plan.turns[0].environment_tokens, 20);
        assert_eq!(plan.turns[1].input_content_tokens, 135);
        assert_eq!(plan.turns[1].target_prompt_tokens, 150);
        assert_eq!(plan.turns[1].environment_tokens, 0);
        assert_eq!(plan.turns[1].tool_call_latency, Duration::from_millis(25));
        assert!(plan.turns[1].reset_prompt.is_none());
        assert_eq!(plan.turns[2].input_content_tokens, 55);
        assert_eq!(plan.turns[2].target_prompt_tokens, 60);
        let reset_prompt = plan.turns[2].reset_prompt.as_ref().unwrap();
        assert_eq!(
            tokenizer
                .encode(reset_prompt.as_str(), false)
                .unwrap()
                .len(),
            55
        );
    }

    #[test]
    fn trajectory_jsonl_rejects_implicit_prompt_reduction_with_line_context() {
        let input = r#"{"schema_version":1,"trajectory_id":"bad","requests":[{"prompt_tokens":100,"output_tokens":20},{"prompt_tokens":110,"output_tokens":10}]}"#;
        let error =
            parse_trajectory_plan_specs(std::io::Cursor::new(input), "fixture").unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("fixture:1"));
        assert!(message.contains("set reset_before=true"));
    }

    #[test]
    fn trajectory_jsonl_rejects_duplicate_ids_and_empty_input() {
        let duplicated = concat!(
            "{\"schema_version\":1,\"trajectory_id\":\"same\",\"requests\":[{\"prompt_tokens\":10,\"output_tokens\":1}]}\n",
            "{\"schema_version\":1,\"trajectory_id\":\"same\",\"requests\":[{\"prompt_tokens\":20,\"output_tokens\":1}]}\n"
        );
        let error =
            parse_trajectory_plan_specs(std::io::Cursor::new(duplicated), "fixture").unwrap_err();
        assert!(error.to_string().contains("duplicate trajectory_id"));

        let error = parse_trajectory_plan_specs(std::io::Cursor::new("\n"), "fixture").unwrap_err();
        assert!(error.to_string().contains("contains no trajectory plans"));
    }

    #[test]
    fn trajectory_jsonl_rejects_unknown_schema_and_fields() {
        let wrong_version = r#"{"schema_version":2,"trajectory_id":"v2","requests":[{"prompt_tokens":10,"output_tokens":1}]}"#;
        let error = parse_trajectory_plan_specs(std::io::Cursor::new(wrong_version), "fixture")
            .unwrap_err();
        assert!(format!("{error:#}").contains("unsupported schema_version"));

        let unknown = r#"{"schema_version":1,"trajectory_id":"extra","requests":[{"prompt_tokens":10,"output_tokens":1,"typo":true}]}"#;
        let error =
            parse_trajectory_plan_specs(std::io::Cursor::new(unknown), "fixture").unwrap_err();
        assert!(format!("{error:#}").contains("unknown field"));
    }

    #[test]
    fn rolling_admission_refills_the_freed_routing_slot_in_fifo_order() {
        let mut admission = RollingAdmission::new(vec!["a", "b", "c", "d"], 2).unwrap();
        assert_eq!(admission.initial_admissions(), vec![(0, "a"), (1, "b")]);
        assert_eq!(admission.replacement(1), Some((1, "c")));
        assert_eq!(admission.replacement(0), Some((0, "d")));
        assert_eq!(admission.replacement(1), None);
    }

    #[test]
    fn rolling_admission_validates_its_limit() {
        assert!(RollingAdmission::new(vec![1], 0).is_err());
        let mut admission = RollingAdmission::new(vec![1], 2).unwrap();
        assert_eq!(admission.initial_admissions(), vec![(0, 1)]);
    }
}
