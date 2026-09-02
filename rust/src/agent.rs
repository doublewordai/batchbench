use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
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
    pub admission: AdmissionMode,
    pub time_scale: f64,
}

/// How replayed trajectories enter the benchmark.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdmissionMode {
    /// Keep up to `max_active_agents` trajectories active; admit the next queued trajectory
    /// whenever one finishes (manifest order).
    #[default]
    ClosedLoop,
    /// Admit each trajectory at its `start_after_ms` offset regardless of free slots.
    /// `max_active_agents` becomes a hard cap that delays admission when reached.
    OpenLoop,
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
            admission: AdmissionMode::ClosedLoop,
            time_scale: 1.0,
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

    pub fn with_admission(mut self, admission: AdmissionMode) -> Self {
        self.admission = admission;
        self
    }

    /// Divide every manifest `start_after_ms` and `delay_after_ms` value by `time_scale`.
    pub fn with_time_scale(mut self, time_scale: f64) -> Result<Self> {
        if !time_scale.is_finite() || time_scale <= 0.0 {
            return Err(anyhow!("time scale must be a positive finite number"));
        }
        self.time_scale = time_scale;
        Ok(self)
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
    tool_call_latency: Duration,
    reset_prompt_tokens: Option<usize>,
    /// Schema v2: start a fresh conversation before this request (drops the block cache).
    reset_before: bool,
    /// Schema v2: ordered content blocks that fully define this request's prompt.
    blocks: Option<Vec<BlockSpec>>,
    /// Schema v2: per-request streaming override.
    stream: Option<bool>,
    /// Schema v2: per-request output cap override.
    max_tokens: Option<usize>,
}

impl AgentTurnPlan {
    fn uses_blocks(&self) -> bool {
        self.blocks.is_some()
    }
}

#[derive(Clone, Debug)]
struct AgentPlan {
    agent_id: usize,
    trajectory_id: String,
    user_tag: Option<String>,
    initial_prompt_tokens: usize,
    initial_content: Option<InitialContent>,
    turns: Vec<AgentTurnPlan>,
    /// Open-loop admission offset (already divided by the time scale).
    start_after: Duration,
}

impl AgentPlan {
    fn uses_blocks(&self) -> bool {
        self.turns.first().is_some_and(AgentTurnPlan::uses_blocks)
    }
}

/// Content prepared for a trajectory's first request before it is admitted.
#[derive(Clone, Debug)]
enum InitialContent {
    /// Schema v1 (and v2 without blocks): the generated initial user prompt.
    Prompt(String),
    /// Schema v2 with blocks: generated text for every non-live block of the first request.
    Blocks(BlockCache),
}

/// Per-trajectory cache of block content keyed by block seed. Equal seeds always resolve to the
/// exact bytes already sent earlier in the trajectory, including live assistant replies.
type BlockCache = HashMap<String, BlockContent>;

#[derive(Clone, Debug)]
enum BlockContent {
    Generated(std::sync::Arc<str>),
    /// The model's own normalized reply from an earlier request in this conversation.
    Live(Value),
}

#[derive(Clone)]
struct SyntheticTextGenerator {
    tokenizer: std::sync::Arc<Tokenizer>,
    eligible_token_ids: std::sync::Arc<Vec<u32>>,
    /// Lazily computed subset of `eligible_token_ids` whose text survives JSON string
    /// serialization unchanged (no quotes, backslashes, or control characters).
    json_safe_token_ids: std::sync::Arc<std::sync::OnceLock<Vec<u32>>>,
    special_token_ids: std::sync::Arc<HashSet<u32>>,
    permits: std::sync::Arc<tokio::sync::Semaphore>,
    root_seed: u64,
}

#[derive(Clone, Copy)]
enum SyntheticTextField {
    InitialPrompt,
    ResetPrompt(usize),
    Environment(usize),
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
    /// When the trajectory was due to start: its scaled `start_after_ms` under open-loop
    /// admission, zero under closed-loop admission.
    pub scheduled_at: Duration,
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
    /// Schema v2 live assistant blocks that had no previous reply to substitute.
    pub live_block_fallbacks: u64,
    /// Open-loop admissions delayed because `max_active_agents` was reached.
    pub late_admissions: u64,
    /// Largest gap between a trajectory's scheduled and actual admission.
    pub max_admission_lag: Duration,
    /// Highest `schema_version` present in the replayed manifest.
    pub trajectory_schema_version: Option<u32>,
}

#[derive(Debug)]
struct AgentWorkerReport {
    agent_id: usize,
    trajectory_id: String,
    routing_slot: usize,
    scheduled_at: Duration,
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
    live_block_fallbacks: u64,
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
            scheduled_at: Duration::ZERO,
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
            live_block_fallbacks: 0,
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
    /// Schema v2 blocks: (block count, live block count).
    blocks: Option<(usize, usize)>,
    stream: Option<bool>,
    max_tokens: Option<usize>,
}

/// Manifest schema versions accepted by `--agent-plans-jsonl`.
pub(crate) const SUPPORTED_TRAJECTORY_PLAN_SCHEMA_VERSIONS: [u32; 2] = [1, 2];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrajectoryPlanSpec {
    schema_version: u32,
    trajectory_id: String,
    requests: Vec<TrajectoryRequestSpec>,
    #[serde(default, rename = "metadata")]
    _metadata: Option<Value>,
    /// Schema v2: admission offset from benchmark start under open-loop admission.
    #[serde(default)]
    start_after_ms: Option<u64>,
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
    /// Schema v2: template/scaffolding tokens the backend adds on top of the content.
    #[serde(default)]
    overhead_tokens: Option<usize>,
    /// Schema v2: request streamed output for this request.
    #[serde(default)]
    stream: Option<bool>,
    /// Schema v2: output cap for this request.
    #[serde(default)]
    max_tokens: Option<usize>,
    /// Schema v2: ordered content blocks defining the prompt.
    #[serde(default)]
    blocks: Option<Vec<BlockSpec>>,
}

impl TrajectoryRequestSpec {
    fn has_schema_v2_fields(&self) -> Option<&'static str> {
        if self.overhead_tokens.is_some() {
            Some("overhead_tokens")
        } else if self.stream.is_some() {
            Some("stream")
        } else if self.max_tokens.is_some() {
            Some("max_tokens")
        } else if self.blocks.is_some() {
            Some("blocks")
        } else {
            None
        }
    }

    fn block_content_tokens(&self) -> Option<Result<usize>> {
        self.blocks.as_ref().map(|blocks| {
            blocks
                .iter()
                .try_fold(0usize, |total, block| total.checked_add(block.tokens))
                .ok_or_else(|| anyhow!("block token sum overflowed usize"))
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BlockRole {
    ToolDefinition,
    System,
    User,
    Assistant,
    Tool,
    ToolCall,
}

impl BlockRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::ToolDefinition => "tool_definition",
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::ToolCall => "tool_call",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockSpec {
    seed: String,
    tokens: usize,
    role: BlockRole,
    #[serde(default)]
    live: bool,
}

impl BlockSpec {
    /// A live assistant block is substituted with the model's previous reply instead of generated.
    fn is_live(&self) -> bool {
        self.live && self.role == BlockRole::Assistant
    }
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
    let (plans, root_seed, trajectory_schema_version) = build_agent_plans(&config)?;
    let text_generator = SyntheticTextGenerator::new(
        std::sync::Arc::new(load_tokenizer(&config.tokenizer_model)?),
        root_seed,
    )?;
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
    let mut accumulator = ReportAccumulator::with_capacity(total_agents);
    let outcome = match config.admission {
        AdmissionMode::ClosedLoop => {
            run_closed_loop(
                plans,
                requested_max_active_agents,
                &client,
                &config,
                &text_generator,
                &mut accumulator,
            )
            .await?
        }
        AdmissionMode::OpenLoop => {
            run_open_loop(
                plans,
                config.max_active_agents,
                &client,
                &config,
                &text_generator,
                &mut accumulator,
            )
            .await?
        }
    };

    if config.dry_run {
        let mut dry_run_records = std::mem::take(&mut accumulator.dry_run_records);
        dry_run_records.sort_by_key(|record| (record.agent_id, record.invocation));
        for record in dry_run_records {
            println!("{}", format_dry_run_record(&record));
        }
    }

    Ok(accumulator.into_report(
        total_agents,
        planned_tool_invocations,
        trajectory_schema_version,
        outcome,
    ))
}

fn format_dry_run_record(record: &DryRunRecord) -> String {
    let mut line = format!(
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
    if let Some((block_count, live_block_count)) = record.blocks {
        line.push_str(&format!(
            " blocks={block_count} live_blocks={live_block_count}"
        ));
    }
    if let Some(stream) = record.stream {
        line.push_str(&format!(" stream={stream}"));
    }
    if let Some(max_tokens) = record.max_tokens {
        line.push_str(&format!(" max_tokens={max_tokens}"));
    }
    line
}

/// Scheduler results that do not come from individual worker reports.
#[derive(Debug)]
struct AdmissionOutcome {
    total_duration: Duration,
    max_active_agents: usize,
    last_agent_admitted_at: Duration,
    late_admissions: u64,
    max_admission_lag: Duration,
}

#[derive(Debug, Default)]
struct ReportAccumulator {
    completed_agents: usize,
    successful_requests: u64,
    failed_requests: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    estimated_cached_input_tokens: u64,
    total_tool_call_latency: Duration,
    latencies: Vec<Duration>,
    agent_end_to_end_latencies: Vec<Duration>,
    failures: Vec<AgentFailureRecord>,
    dry_run_records: Vec<DryRunRecord>,
    agent_lifecycles: Vec<AgentLifecycleRecord>,
    live_block_fallbacks: u64,
}

impl ReportAccumulator {
    fn with_capacity(total_agents: usize) -> Self {
        Self {
            agent_lifecycles: Vec::with_capacity(total_agents),
            ..Self::default()
        }
    }

    fn absorb(&mut self, worker: AgentWorkerReport, finished_at: Duration) {
        self.agent_lifecycles.push(AgentLifecycleRecord {
            agent_id: worker.agent_id,
            trajectory_id: worker.trajectory_id.clone(),
            routing_slot: worker.routing_slot,
            scheduled_at: worker.scheduled_at,
            admitted_at: worker.admitted_at,
            finished_at,
            completed: worker.completed,
        });
        if worker.completed {
            self.completed_agents += 1;
            self.agent_end_to_end_latencies
                .push(worker.end_to_end_latency);
        }
        self.successful_requests = self
            .successful_requests
            .saturating_add(worker.successful_requests);
        self.failed_requests = self.failed_requests.saturating_add(worker.failed_requests);
        self.total_input_tokens = self.total_input_tokens.saturating_add(worker.input_tokens);
        self.total_output_tokens = self
            .total_output_tokens
            .saturating_add(worker.output_tokens);
        self.estimated_cached_input_tokens = self
            .estimated_cached_input_tokens
            .saturating_add(worker.estimated_cached_input_tokens);
        self.total_tool_call_latency = self
            .total_tool_call_latency
            .saturating_add(worker.tool_call_latency);
        self.live_block_fallbacks = self
            .live_block_fallbacks
            .saturating_add(worker.live_block_fallbacks);
        self.latencies.extend(worker.latencies);
        self.failures.extend(worker.failures);
        self.dry_run_records.extend(worker.dry_run_records);
    }

    fn into_report(
        mut self,
        total_agents: usize,
        planned_tool_invocations: u64,
        trajectory_schema_version: Option<u32>,
        outcome: AdmissionOutcome,
    ) -> AgentBenchmarkReport {
        let total_duration = outcome.total_duration;
        let final_drain_duration = total_duration.saturating_sub(outcome.last_agent_admitted_at);
        self.latencies.sort();
        self.agent_end_to_end_latencies.sort();
        self.failures
            .sort_by_key(|failure| (failure.agent_id, failure.invocation));
        self.agent_lifecycles.sort_by_key(|record| record.agent_id);
        let total_requests = self
            .successful_requests
            .saturating_add(self.failed_requests);
        let duration_secs = total_duration.as_secs_f64();

        AgentBenchmarkReport {
            total_agents,
            max_active_agents: outcome.max_active_agents,
            completed_agents: self.completed_agents,
            planned_tool_invocations,
            total_requests,
            successful_requests: self.successful_requests,
            failed_requests: self.failed_requests,
            total_input_tokens: self.total_input_tokens,
            total_output_tokens: self.total_output_tokens,
            estimated_cached_input_tokens: self.estimated_cached_input_tokens,
            total_tool_call_latency: self.total_tool_call_latency,
            total_duration,
            last_agent_admitted_at: outcome.last_agent_admitted_at,
            final_drain_duration,
            input_tokens_per_second: rate(self.total_input_tokens, duration_secs),
            output_tokens_per_second: rate(self.total_output_tokens, duration_secs),
            requests_per_second: rate(total_requests, duration_secs),
            latency_p50: percentile(&self.latencies, 0.50),
            latency_p90: percentile(&self.latencies, 0.90),
            latency_p99: percentile(&self.latencies, 0.99),
            agent_end_to_end_latency_p50: percentile(&self.agent_end_to_end_latencies, 0.50),
            agent_end_to_end_latency_p90: percentile(&self.agent_end_to_end_latencies, 0.90),
            agent_end_to_end_latency_p99: percentile(&self.agent_end_to_end_latencies, 0.99),
            failures: self.failures,
            agent_lifecycles: self.agent_lifecycles,
            live_block_fallbacks: self.live_block_fallbacks,
            late_admissions: outcome.late_admissions,
            max_admission_lag: outcome.max_admission_lag,
            trajectory_schema_version,
        }
    }
}

/// Closed-loop admission: keep up to `max_active` trajectories running and refill a freed slot
/// from the manifest queue whenever a trajectory finishes.
async fn run_closed_loop(
    plans: Vec<AgentPlan>,
    requested_max_active_agents: usize,
    client: &Client,
    config: &std::sync::Arc<AgentLoopConfig>,
    text_generator: &SyntheticTextGenerator,
    accumulator: &mut ReportAccumulator,
) -> Result<AdmissionOutcome> {
    let mut admission = RollingAdmission::new(plans, requested_max_active_agents)?;
    let max_active_agents = admission.max_active;
    // Initial prompt synthesis is deliberately outside the benchmark clock. The live working set
    // still starts together instead of being accidentally ramped by client-side token generation.
    let initial_admissions =
        materialize_initial_prompts(admission.initial_admissions(), text_generator).await?;
    let start = Instant::now();

    let mut join_set = JoinSet::new();
    for (routing_slot, plan) in initial_admissions {
        spawn_agent(
            &mut join_set,
            plan,
            routing_slot,
            Duration::ZERO,
            client,
            config,
            text_generator,
        );
    }
    let mut last_agent_admitted_at = Duration::ZERO;
    // Keep prompt preparation off the sole worker-completion drain path. Sequence numbers retain
    // manifest FIFO admission even when later prompt preparations finish first.
    let mut replacement_preparations = JoinSet::new();
    let mut prepared_replacements = BTreeMap::new();
    let mut next_replacement_sequence = 0usize;
    let mut next_sequence_to_admit = 0usize;

    loop {
        if join_set.is_empty() && replacement_preparations.is_empty() {
            break;
        }

        tokio::select! {
            joined = join_set.join_next(), if !join_set.is_empty() => {
                let joined = joined.expect("guarded non-empty agent worker set returned no task");
                let worker = joined.map_err(|err| anyhow!("agent worker task failed: {}", err))??;
                let routing_slot = worker.routing_slot;
                accumulator.absorb(worker, start.elapsed());

                if let Some((routing_slot, plan)) = admission.replacement(routing_slot) {
                    let text_generator = text_generator.clone();
                    let replacement_sequence = next_replacement_sequence;
                    next_replacement_sequence = next_replacement_sequence
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("replacement admission sequence overflowed"))?;
                    spawn_admission_preparation(
                        &mut replacement_preparations,
                        replacement_sequence,
                        (routing_slot, plan),
                        move |(routing_slot, plan)| async move {
                            materialize_initial_prompt(plan, &text_generator)
                                .await
                                .map(|plan| (routing_slot, plan))
                        },
                    );
                }
            }
            prepared = replacement_preparations.join_next(), if !replacement_preparations.is_empty() => {
                let prepared = prepared
                    .expect("guarded non-empty admission preparation set returned no task")
                    .map_err(|error| anyhow!("replacement prompt worker failed: {error}"))??;
                let (replacement_sequence, replacement) = prepared;
                if prepared_replacements
                    .insert(replacement_sequence, replacement)
                    .is_some()
                {
                    return Err(anyhow!(
                        "duplicate prepared replacement sequence {replacement_sequence}"
                    ));
                }
                for (routing_slot, plan) in take_prepared_in_fifo_order(
                    &mut prepared_replacements,
                    &mut next_sequence_to_admit,
                )? {
                    let admitted_at = start.elapsed();
                    last_agent_admitted_at = admitted_at;
                    spawn_agent(
                        &mut join_set,
                        plan,
                        routing_slot,
                        admitted_at,
                        client,
                        config,
                        text_generator,
                    );
                }
            }
        }
    }

    if !prepared_replacements.is_empty() {
        return Err(anyhow!(
            "prepared replacement admissions contain an unresolved FIFO gap"
        ));
    }

    Ok(AdmissionOutcome {
        total_duration: start.elapsed(),
        max_active_agents,
        last_agent_admitted_at,
        late_admissions: 0,
        max_admission_lag: Duration::ZERO,
    })
}

/// Upper bound on trajectories whose first request is prepared ahead of their open-loop start
/// time. Bounds memory while keeping generation off the admission path.
const OPEN_LOOP_PREPARE_LOOKAHEAD: usize = 256;

#[derive(Debug, Default)]
struct OpenLoopStats {
    late_admissions: u64,
    max_admission_lag: Duration,
    last_agent_admitted_at: Duration,
}

/// Routing slots for open-loop admission: the lowest free slot is reused so that concurrent
/// trajectories occupy consecutive slots (balanced data-parallel routing).
#[derive(Debug, Default)]
struct SlotPool {
    free: BinaryHeap<Reverse<usize>>,
    allocated: usize,
}

impl SlotPool {
    fn acquire(&mut self) -> usize {
        match self.free.pop() {
            Some(Reverse(slot)) => slot,
            None => {
                let slot = self.allocated;
                self.allocated += 1;
                slot
            }
        }
    }

    fn release(&mut self, slot: usize) {
        self.free.push(Reverse(slot));
    }

    /// Highest number of simultaneously held slots so far.
    fn peak(&self) -> usize {
        self.allocated
    }
}

struct OpenLoopShared {
    start: tokio::time::Instant,
    cap: Option<std::sync::Arc<tokio::sync::Semaphore>>,
    slots: std::sync::Mutex<SlotPool>,
    stats: std::sync::Mutex<OpenLoopStats>,
}

/// Open-loop admission: every trajectory starts at its `start_after` offset. The optional cap
/// only delays admission (counted as late) instead of queueing trajectories for free slots.
async fn run_open_loop(
    mut plans: Vec<AgentPlan>,
    max_active_agents: Option<usize>,
    client: &Client,
    config: &std::sync::Arc<AgentLoopConfig>,
    text_generator: &SyntheticTextGenerator,
    accumulator: &mut ReportAccumulator,
) -> Result<AdmissionOutcome> {
    plans.sort_by_key(|plan| plan.start_after);
    let lookahead = std::sync::Arc::new(tokio::sync::Semaphore::new(OPEN_LOOP_PREPARE_LOOKAHEAD));
    let remaining = plans.split_off(plans.len().min(OPEN_LOOP_PREPARE_LOOKAHEAD));
    // The earliest trajectories are prepared before the clock starts, as in closed-loop mode.
    let prepared =
        materialize_initial_prompts(plans.into_iter().enumerate().collect(), text_generator)
            .await?;

    let shared = std::sync::Arc::new(OpenLoopShared {
        start: tokio::time::Instant::now(),
        cap: max_active_agents.map(|cap| std::sync::Arc::new(tokio::sync::Semaphore::new(cap))),
        slots: std::sync::Mutex::new(SlotPool::default()),
        stats: std::sync::Mutex::new(OpenLoopStats::default()),
    });
    let mut join_set = JoinSet::new();
    for (_, plan) in prepared {
        let permit = std::sync::Arc::clone(&lookahead)
            .try_acquire_owned()
            .map_err(|_| anyhow!("open-loop lookahead permits exhausted before admission"))?;
        spawn_open_loop_agent(
            &mut join_set,
            plan,
            permit,
            &shared,
            client,
            config,
            text_generator,
        );
    }

    let mut pending = remaining.into_iter();
    let mut next_plan = pending.next();
    loop {
        if next_plan.is_none() && join_set.is_empty() {
            break;
        }
        tokio::select! {
            permit = std::sync::Arc::clone(&lookahead).acquire_owned(), if next_plan.is_some() => {
                let permit = permit.map_err(|_| anyhow!("open-loop lookahead pool closed unexpectedly"))?;
                let plan = next_plan.take().expect("guarded pending plan is present");
                spawn_open_loop_agent(
                    &mut join_set,
                    plan,
                    permit,
                    &shared,
                    client,
                    config,
                    text_generator,
                );
                next_plan = pending.next();
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                let joined = joined.expect("guarded non-empty agent worker set returned no task");
                let worker = joined.map_err(|err| anyhow!("agent worker task failed: {}", err))??;
                accumulator.absorb(worker, shared.start.elapsed());
            }
        }
    }

    let stats = shared
        .stats
        .lock()
        .map_err(|_| anyhow!("open-loop statistics lock poisoned"))?;
    let peak_active = shared
        .slots
        .lock()
        .map_err(|_| anyhow!("open-loop slot pool lock poisoned"))?
        .peak();
    Ok(AdmissionOutcome {
        total_duration: shared.start.elapsed(),
        max_active_agents: peak_active,
        last_agent_admitted_at: stats.last_agent_admitted_at,
        late_admissions: stats.late_admissions,
        max_admission_lag: stats.max_admission_lag,
    })
}

fn spawn_open_loop_agent(
    join_set: &mut JoinSet<Result<AgentWorkerReport>>,
    plan: AgentPlan,
    lookahead_permit: tokio::sync::OwnedSemaphorePermit,
    shared: &std::sync::Arc<OpenLoopShared>,
    client: &Client,
    config: &std::sync::Arc<AgentLoopConfig>,
    text_generator: &SyntheticTextGenerator,
) {
    let shared = std::sync::Arc::clone(shared);
    let client = client.clone();
    let config = std::sync::Arc::clone(config);
    let text_generator = text_generator.clone();
    join_set.spawn(async move {
        let plan = if plan.initial_content.is_some() {
            plan
        } else {
            materialize_initial_prompt(plan, &text_generator).await?
        };
        let scheduled_at = plan.start_after;
        tokio::time::sleep_until(shared.start + scheduled_at).await;
        let (cap_permit, late) = admit_under_cap(shared.cap.as_ref()).await?;
        let admitted_at = shared.start.elapsed();
        drop(lookahead_permit);
        {
            let mut stats = shared
                .stats
                .lock()
                .map_err(|_| anyhow!("open-loop statistics lock poisoned"))?;
            if late {
                stats.late_admissions += 1;
            }
            stats.max_admission_lag = stats
                .max_admission_lag
                .max(admitted_at.saturating_sub(scheduled_at));
            stats.last_agent_admitted_at = stats.last_agent_admitted_at.max(admitted_at);
        }
        let routing_slot = shared
            .slots
            .lock()
            .map_err(|_| anyhow!("open-loop slot pool lock poisoned"))?
            .acquire();
        let result = run_agent(
            plan,
            routing_slot,
            admitted_at,
            client,
            config,
            text_generator,
        )
        .await;
        if let Ok(mut slots) = shared.slots.lock() {
            slots.release(routing_slot);
        }
        drop(cap_permit);
        let mut report = result?;
        report.scheduled_at = scheduled_at;
        Ok(report)
    });
}

/// Acquire an active-agent permit; `true` when the cap was full and admission had to wait.
async fn admit_under_cap(
    cap: Option<&std::sync::Arc<tokio::sync::Semaphore>>,
) -> Result<(Option<tokio::sync::OwnedSemaphorePermit>, bool)> {
    let Some(cap) = cap else {
        return Ok((None, false));
    };
    match std::sync::Arc::clone(cap).try_acquire_owned() {
        Ok(permit) => Ok((Some(permit), false)),
        Err(tokio::sync::TryAcquireError::NoPermits) => {
            let permit = std::sync::Arc::clone(cap)
                .acquire_owned()
                .await
                .map_err(|_| anyhow!("active-agent cap closed unexpectedly"))?;
            Ok((Some(permit), true))
        }
        Err(tokio::sync::TryAcquireError::Closed) => {
            Err(anyhow!("active-agent cap closed unexpectedly"))
        }
    }
}

async fn materialize_initial_prompts(
    admissions: Vec<(usize, AgentPlan)>,
    text_generator: &SyntheticTextGenerator,
) -> Result<Vec<(usize, AgentPlan)>> {
    let mut join_set = JoinSet::new();
    for (routing_slot, plan) in admissions {
        let text_generator = text_generator.clone();
        spawn_admission_preparation(&mut join_set, routing_slot, plan, move |plan| async move {
            materialize_initial_prompt(plan, &text_generator).await
        });
    }

    let mut prepared = Vec::with_capacity(join_set.len());
    while let Some(joined) = join_set.join_next().await {
        prepared.push(joined.map_err(|error| anyhow!("initial-prompt worker failed: {error}"))??);
    }
    prepared.sort_by_key(|(routing_slot, _)| *routing_slot);
    Ok(prepared)
}

async fn materialize_initial_prompt(
    mut plan: AgentPlan,
    text_generator: &SyntheticTextGenerator,
) -> Result<AgentPlan> {
    let initial_content = match plan.turns.first().and_then(|turn| turn.blocks.as_deref()) {
        Some(blocks) => {
            let mut cache = BlockCache::new();
            let generated = text_generator
                .generate_blocks(missing_generated_blocks(blocks, &cache))
                .await
                .with_context(|| {
                    format!(
                        "failed to generate initial blocks for trajectory {:?}",
                        plan.trajectory_id
                    )
                })?;
            for (seed, text) in generated {
                cache.insert(seed, BlockContent::Generated(text));
            }
            InitialContent::Blocks(cache)
        }
        None => InitialContent::Prompt(
            text_generator
                .generate(
                    plan.agent_id,
                    SyntheticTextField::InitialPrompt,
                    plan.initial_prompt_tokens,
                )
                .await
                .with_context(|| {
                    format!(
                        "failed to generate initial prompt for trajectory {:?}",
                        plan.trajectory_id
                    )
                })?,
        ),
    };
    plan.initial_content = Some(initial_content);
    Ok(plan)
}

/// Blocks that still need generated content: not live, not cached, first occurrence per seed.
fn missing_generated_blocks(blocks: &[BlockSpec], cache: &BlockCache) -> Vec<BlockSpec> {
    let mut seen = HashSet::new();
    blocks
        .iter()
        .filter(|block| {
            !block.is_live() && !cache.contains_key(&block.seed) && seen.insert(block.seed.clone())
        })
        .cloned()
        .collect()
}

fn spawn_admission_preparation<T, Prepare, Prepared>(
    join_set: &mut JoinSet<Result<(usize, T)>>,
    admission_key: usize,
    item: T,
    prepare: Prepare,
) where
    T: Send + 'static,
    Prepare: FnOnce(T) -> Prepared + Send + 'static,
    Prepared: std::future::Future<Output = Result<T>> + Send + 'static,
{
    join_set.spawn(async move {
        prepare(item)
            .await
            .map(|prepared| (admission_key, prepared))
    });
}

fn take_prepared_in_fifo_order<T>(
    prepared: &mut BTreeMap<usize, T>,
    next_sequence: &mut usize,
) -> Result<Vec<T>> {
    let mut ready = Vec::new();
    while let Some(item) = prepared.remove(next_sequence) {
        ready.push(item);
        *next_sequence = next_sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("replacement admission sequence overflowed"))?;
    }
    Ok(ready)
}

fn spawn_agent(
    join_set: &mut JoinSet<Result<AgentWorkerReport>>,
    plan: AgentPlan,
    routing_slot: usize,
    admitted_at: Duration,
    client: &Client,
    config: &std::sync::Arc<AgentLoopConfig>,
    text_generator: &SyntheticTextGenerator,
) {
    let client = client.clone();
    let config = std::sync::Arc::clone(config);
    let text_generator = text_generator.clone();
    join_set.spawn(async move {
        run_agent(
            plan,
            routing_slot,
            admitted_at,
            client,
            config,
            text_generator,
        )
        .await
    });
}

/// Agent plans plus the root seed and, for replay, the highest manifest schema version.
type BuiltAgentPlans = (Vec<AgentPlan>, u64, Option<u32>);

fn build_agent_plans(config: &AgentLoopConfig) -> Result<BuiltAgentPlans> {
    let root_seed = match config.seed {
        Some(seed) => seed,
        None => rand::thread_rng().gen(),
    };

    if let Some(path) = config.agent_plans_jsonl.as_deref() {
        let specs = load_trajectory_plan_specs(path)?;
        let schema_version = specs.iter().map(|spec| spec.schema_version).max();
        return Ok((
            build_replay_agent_plans(config, specs)?,
            root_seed,
            schema_version,
        ));
    }

    let mut plans = Vec::with_capacity(config.agent_count);
    for agent_id in 0..config.agent_count {
        let mut input_rng = stream_rng(root_seed, agent_id, 0);
        let mut invocation_rng = stream_rng(root_seed, agent_id, 1);
        let mut output_rng = stream_rng(root_seed, agent_id, 2);
        let mut environment_rng = stream_rng(root_seed, agent_id, 3);
        let mut tool_call_latency_rng = stream_rng(root_seed, agent_id, 5);

        let initial_prompt_tokens = config.input_tokens.sample(&mut input_rng, "input tokens")?;
        let invocation_count = config
            .tool_invocations
            .sample(&mut invocation_rng, "tool invocations")?;
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
            turns.push(AgentTurnPlan {
                input_content_tokens: target_prompt_tokens,
                target_prompt_tokens,
                output_tokens,
                environment_tokens,
                tool_call_latency: Duration::from_millis(tool_call_latency_ms),
                reset_prompt_tokens: None,
                reset_before: false,
                blocks: None,
                stream: None,
                max_tokens: None,
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
            initial_prompt_tokens,
            initial_content: None,
            turns,
            start_after: Duration::ZERO,
        });
    }
    Ok((plans, root_seed, None))
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
    if !SUPPORTED_TRAJECTORY_PLAN_SCHEMA_VERSIONS.contains(&spec.schema_version) {
        return Err(anyhow!(
            "unsupported schema_version {}; expected one of {:?}",
            spec.schema_version,
            SUPPORTED_TRAJECTORY_PLAN_SCHEMA_VERSIONS
        ));
    }
    let schema_v2 = spec.schema_version >= 2;
    if !schema_v2 && spec.start_after_ms.is_some() {
        return Err(anyhow!("start_after_ms requires schema_version 2"));
    }
    if spec.trajectory_id.trim().is_empty() {
        return Err(anyhow!("trajectory_id must not be empty"));
    }
    if spec.requests.is_empty() {
        return Err(anyhow!("requests must contain at least one request"));
    }

    let uses_blocks = spec.requests[0].blocks.is_some();
    for (request_index, request) in spec.requests.iter().enumerate() {
        let request_number = request_index + 1;
        if !schema_v2 {
            if let Some(field) = request.has_schema_v2_fields() {
                return Err(anyhow!(
                    "request {request_number} field {field} requires schema_version 2"
                ));
            }
        }
        if request.prompt_tokens == 0 {
            return Err(anyhow!(
                "request {request_number} prompt_tokens must be greater than zero"
            ));
        }
        if request.output_tokens == 0 {
            return Err(anyhow!(
                "request {request_number} output_tokens must be greater than zero"
            ));
        }
        if request.max_tokens == Some(0) {
            return Err(anyhow!(
                "request {request_number} max_tokens must be greater than zero"
            ));
        }
        if request_index == 0 && request.reset_before && !schema_v2 {
            return Err(anyhow!("the first request cannot set reset_before"));
        }
        if request.blocks.is_some() != uses_blocks {
            return Err(anyhow!(
                "request {request_number} must {} blocks because the trajectory's first request {}",
                if uses_blocks { "define" } else { "omit" },
                if uses_blocks { "does" } else { "does not" }
            ));
        }
        if let Some(blocks) = request.blocks.as_deref() {
            validate_blocks(blocks)
                .with_context(|| format!("request {request_number} blocks are invalid"))?;
        }
    }

    if !uses_blocks {
        // Without blocks the next prompt is inferred from the current one, so every transition
        // must leave room for the appended output and environment response.
        let overheads = effective_overheads(&spec.requests, 0, 0);
        for request_index in 1..spec.requests.len() {
            let previous = &spec.requests[request_index - 1];
            let request = &spec.requests[request_index];
            if request.reset_before {
                continue;
            }
            let previous_content = previous
                .prompt_tokens
                .saturating_sub(overheads[request_index - 1]);
            let content = request
                .prompt_tokens
                .saturating_sub(overheads[request_index]);
            inferred_environment_tokens_from_content(
                previous_content,
                previous.output_tokens,
                content,
            )
            .with_context(|| {
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

fn validate_blocks(blocks: &[BlockSpec]) -> Result<()> {
    if blocks.is_empty() {
        return Err(anyhow!("blocks must contain at least one block"));
    }
    for (block_index, block) in blocks.iter().enumerate() {
        if block.seed.is_empty() {
            return Err(anyhow!("block {} seed must not be empty", block_index + 1));
        }
        if block.live && block.role != BlockRole::Assistant {
            return Err(anyhow!(
                "block {} is live but has role {}; only assistant blocks can be live",
                block_index + 1,
                block.role.as_str()
            ));
        }
    }
    Ok(())
}

/// Template/scaffolding tokens the backend adds to each request's content.
///
/// A per-request `overhead_tokens` wins. Otherwise a first or reset request carries the
/// initial overhead and every appended turn adds the turn overhead to the previous value.
fn effective_overheads(
    requests: &[TrajectoryRequestSpec],
    initial_overhead_tokens: usize,
    turn_overhead_tokens: usize,
) -> Vec<usize> {
    let mut overheads: Vec<usize> = Vec::with_capacity(requests.len());
    for (request_index, request) in requests.iter().enumerate() {
        let overhead = match request.overhead_tokens {
            Some(overhead) => overhead,
            None if request_index == 0 || request.reset_before => initial_overhead_tokens,
            None => overheads[request_index - 1].saturating_add(turn_overhead_tokens),
        };
        overheads.push(overhead);
    }
    overheads
}

/// Environment tokens appended after the current turn so that the next content target is met.
fn inferred_environment_tokens_from_content(
    current_content_tokens: usize,
    current_output_tokens: usize,
    next_content_tokens: usize,
) -> Result<usize> {
    let current_context = current_content_tokens
        .checked_add(current_output_tokens)
        .ok_or_else(|| anyhow!("trajectory content-token count overflowed usize"))?;
    next_content_tokens
        .checked_sub(current_context)
        .ok_or_else(|| {
            anyhow!(
                "next content ({next_content_tokens}) is smaller than current content plus output ({current_context})"
            )
        })
}

fn scale_millis(millis: u64, time_scale: f64) -> Duration {
    if time_scale == 1.0 {
        Duration::from_millis(millis)
    } else {
        Duration::from_secs_f64(millis as f64 / 1000.0 / time_scale)
    }
}

fn build_replay_agent_plans(
    config: &AgentLoopConfig,
    specs: Vec<TrajectoryPlanSpec>,
) -> Result<Vec<AgentPlan>> {
    let mut plans = Vec::with_capacity(specs.len());
    for (agent_id, spec) in specs.into_iter().enumerate() {
        let overheads = effective_overheads(
            &spec.requests,
            config.replay_initial_overhead_tokens,
            config.replay_turn_overhead_tokens,
        );
        let plan = if spec.requests[0].blocks.is_some() {
            build_block_replay_plan(config, agent_id, &spec, &overheads)?
        } else {
            build_growth_replay_plan(config, agent_id, &spec, &overheads)?
        };
        plans.push(plan);
    }
    Ok(plans)
}

/// Schema v1 semantics (also v2 without blocks): the next prompt is the current prompt plus the
/// model output plus an inferred environment response.
fn build_growth_replay_plan(
    config: &AgentLoopConfig,
    agent_id: usize,
    spec: &TrajectoryPlanSpec,
    overheads: &[usize],
) -> Result<AgentPlan> {
    let content_tokens = |request_index: usize| -> Result<usize> {
        let request = &spec.requests[request_index];
        request
            .prompt_tokens
            .checked_sub(overheads[request_index])
            .filter(|tokens| *tokens > 0)
            .ok_or_else(|| {
                anyhow!(
                    "trajectory {:?} request {} prompt ({}) must exceed its replay overhead ({})",
                    spec.trajectory_id,
                    request_index + 1,
                    request.prompt_tokens,
                    overheads[request_index]
                )
            })
    };

    let initial_prompt_tokens = content_tokens(0)?;
    let mut turns = Vec::with_capacity(spec.requests.len());
    let mut current_input_content_tokens = initial_prompt_tokens;

    for (request_index, request) in spec.requests.iter().enumerate() {
        let reset_prompt = if request.reset_before && request_index > 0 {
            let reset_prompt_tokens = content_tokens(request_index)?;
            current_input_content_tokens = reset_prompt_tokens;
            Some(reset_prompt_tokens)
        } else {
            None
        };
        let environment_tokens = match spec.requests.get(request_index + 1) {
            Some(next) if !next.reset_before => inferred_environment_tokens_from_content(
                current_input_content_tokens,
                request.output_tokens,
                content_tokens(request_index + 1)?,
            )
            .with_context(|| {
                format!(
                    "trajectory {:?} request {} cannot satisfy its replay overhead ({} tokens); lower the overhead or set reset_before=true",
                    spec.trajectory_id,
                    request_index + 2,
                    overheads[request_index + 1]
                )
            })?,
            _ => 0,
        };
        turns.push(AgentTurnPlan {
            input_content_tokens: current_input_content_tokens,
            target_prompt_tokens: request.prompt_tokens,
            output_tokens: request.output_tokens,
            environment_tokens,
            tool_call_latency: scale_millis(request.delay_after_ms, config.time_scale),
            reset_prompt_tokens: reset_prompt,
            reset_before: request.reset_before && request_index > 0,
            blocks: None,
            stream: request.stream,
            max_tokens: request.max_tokens,
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

    Ok(AgentPlan {
        agent_id,
        trajectory_id: spec.trajectory_id.clone(),
        user_tag: generate_user_tag(config.user_tagging, config.user_prefix.as_deref(), agent_id),
        initial_prompt_tokens,
        initial_content: None,
        turns,
        start_after: scale_millis(spec.start_after_ms.unwrap_or(0), config.time_scale),
    })
}

/// Schema v2 with blocks: every request's prompt is fully described by its blocks, so no
/// environment growth is inferred. `prompt_tokens` remains the reporting target.
fn build_block_replay_plan(
    config: &AgentLoopConfig,
    agent_id: usize,
    spec: &TrajectoryPlanSpec,
    overheads: &[usize],
) -> Result<AgentPlan> {
    let mut turns = Vec::with_capacity(spec.requests.len());
    let mut warned_mismatch = false;
    let mut initial_prompt_tokens = 0;

    for (request_index, request) in spec.requests.iter().enumerate() {
        let blocks = request
            .blocks
            .clone()
            .ok_or_else(|| anyhow!("validated block trajectory lost its blocks"))?;
        let content_tokens = request
            .block_content_tokens()
            .transpose()?
            .unwrap_or_default();
        let expected_prompt_tokens = content_tokens.saturating_add(overheads[request_index]);
        if expected_prompt_tokens != request.prompt_tokens && !warned_mismatch {
            warned_mismatch = true;
            eprintln!(
                "warning: trajectory {:?} request {}: prompt_tokens {} != sum(blocks.tokens) {} + overhead_tokens {}; replaying the blocks as written",
                spec.trajectory_id,
                request_index + 1,
                request.prompt_tokens,
                content_tokens,
                overheads[request_index]
            );
        }
        if request_index == 0 {
            initial_prompt_tokens = content_tokens;
        }
        turns.push(AgentTurnPlan {
            input_content_tokens: content_tokens,
            target_prompt_tokens: request.prompt_tokens,
            output_tokens: request.output_tokens,
            environment_tokens: 0,
            tool_call_latency: scale_millis(request.delay_after_ms, config.time_scale),
            reset_prompt_tokens: None,
            reset_before: request.reset_before && request_index > 0,
            blocks: Some(blocks),
            stream: request.stream,
            max_tokens: request.max_tokens,
        });
    }

    Ok(AgentPlan {
        agent_id,
        trajectory_id: spec.trajectory_id.clone(),
        user_tag: generate_user_tag(config.user_tagging, config.user_prefix.as_deref(), agent_id),
        initial_prompt_tokens,
        initial_content: None,
        turns,
        start_after: scale_millis(spec.start_after_ms.unwrap_or(0), config.time_scale),
    })
}

fn stream_rng(root_seed: u64, agent_id: usize, stream_id: u64) -> rand::rngs::StdRng {
    let mixed = root_seed
        .wrapping_add((agent_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(stream_id.wrapping_mul(0xD1B5_4A32_D192_ED03));
    rand::rngs::StdRng::seed_from_u64(mixed)
}

impl SyntheticTextGenerator {
    fn new(tokenizer: std::sync::Arc<Tokenizer>, root_seed: u64) -> Result<Self> {
        let (eligible_token_ids, special_token_ids) = synthetic_token_ids(&tokenizer)?;
        let worker_count = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .min(32);
        Ok(Self {
            tokenizer,
            eligible_token_ids: std::sync::Arc::new(eligible_token_ids),
            json_safe_token_ids: std::sync::Arc::new(std::sync::OnceLock::new()),
            special_token_ids: std::sync::Arc::new(special_token_ids),
            permits: std::sync::Arc::new(tokio::sync::Semaphore::new(worker_count)),
            root_seed,
        })
    }

    /// Generate a schema v2 block's content from its seed alone, so equal seeds produce identical
    /// bytes in every trajectory, request, and run.
    async fn generate_block(&self, block: &BlockSpec) -> Result<std::sync::Arc<str>> {
        let permit = std::sync::Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|_| anyhow!("synthetic-text worker pool closed unexpectedly"))?;
        let tokenizer = std::sync::Arc::clone(&self.tokenizer);
        let eligible_token_ids = std::sync::Arc::clone(&self.eligible_token_ids);
        let json_safe_token_ids = std::sync::Arc::clone(&self.json_safe_token_ids);
        let special_token_ids = std::sync::Arc::clone(&self.special_token_ids);
        let block = block.clone();

        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            generate_block_text(
                &tokenizer,
                &eligible_token_ids,
                &json_safe_token_ids,
                &special_token_ids,
                &block,
            )
            .map(std::sync::Arc::from)
        })
        .await
        .map_err(|error| anyhow!("synthetic-text worker failed: {error}"))?
    }

    /// Generate several blocks concurrently through the bounded worker pool, returning
    /// `(seed, text)` pairs in input order.
    async fn generate_blocks(
        &self,
        blocks: Vec<BlockSpec>,
    ) -> Result<Vec<(String, std::sync::Arc<str>)>> {
        let mut join_set = JoinSet::new();
        for (index, block) in blocks.into_iter().enumerate() {
            let generator = self.clone();
            join_set.spawn(async move {
                generator
                    .generate_block(&block)
                    .await
                    .map(|text| (index, block.seed, text))
            });
        }
        let mut generated = Vec::with_capacity(join_set.len());
        while let Some(joined) = join_set.join_next().await {
            generated.push(joined.map_err(|error| anyhow!("block worker failed: {error}"))??);
        }
        generated.sort_by_key(|(index, _, _)| *index);
        Ok(generated
            .into_iter()
            .map(|(_, seed, text)| (seed, text))
            .collect())
    }

    async fn generate(
        &self,
        agent_id: usize,
        field: SyntheticTextField,
        target_tokens: usize,
    ) -> Result<String> {
        if target_tokens == 0 {
            return Ok(String::new());
        }

        let permit = std::sync::Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|_| anyhow!("synthetic-text worker pool closed unexpectedly"))?;
        let tokenizer = std::sync::Arc::clone(&self.tokenizer);
        let eligible_token_ids = std::sync::Arc::clone(&self.eligible_token_ids);
        let special_token_ids = std::sync::Arc::clone(&self.special_token_ids);
        let mut rng = stream_rng(self.root_seed, agent_id, synthetic_text_stream_id(field));

        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            generate_synthetic_text_with_ids(
                &tokenizer,
                &eligible_token_ids,
                &special_token_ids,
                target_tokens,
                &mut rng,
            )
        })
        .await
        .map_err(|error| anyhow!("synthetic-text worker failed: {error}"))?
    }
}

fn seed_digest_hex(seed: &str) -> String {
    format!("{:x}", Sha256::digest(seed.as_bytes()))
}

/// Deterministic RNG derived from a block seed and nothing else.
fn block_seed_rng(seed: &str) -> rand::rngs::StdRng {
    let digest = Sha256::digest(seed.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    rand::rngs::StdRng::seed_from_u64(u64::from_le_bytes(bytes))
}

fn synthetic_tool_name(seed: &str) -> String {
    format!("fn_{}", &seed_digest_hex(seed)[..12])
}

fn synthetic_block_tool_call_id(seed: &str) -> String {
    format!("call_{}", &seed_digest_hex(seed)[..24])
}

fn synthetic_tool_definition(name: &str, description: &str) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}}
        }
    })
}

fn synthetic_tool_call_arguments(input: &str) -> String {
    serde_json::to_string(&json!({ "input": input }))
        .expect("a JSON object with one string value always serializes")
}

fn generate_block_text(
    tokenizer: &Tokenizer,
    eligible_token_ids: &[u32],
    json_safe_token_ids: &std::sync::OnceLock<Vec<u32>>,
    special_token_ids: &HashSet<u32>,
    block: &BlockSpec,
) -> Result<String> {
    let seed = block.seed.as_str();
    let text = match block.role {
        BlockRole::ToolDefinition => {
            let name = synthetic_tool_name(seed);
            generate_wrapped_text(
                tokenizer,
                json_safe_token_ids
                    .get_or_init(|| json_safe_token_ids_for(tokenizer, eligible_token_ids)),
                special_token_ids,
                block.tokens,
                || block_seed_rng(seed),
                |description| {
                    serde_json::to_string(&synthetic_tool_definition(&name, description))
                        .expect("a synthetic tool definition always serializes")
                },
            )
        }
        BlockRole::ToolCall => generate_wrapped_text(
            tokenizer,
            json_safe_token_ids
                .get_or_init(|| json_safe_token_ids_for(tokenizer, eligible_token_ids)),
            special_token_ids,
            block.tokens,
            || block_seed_rng(seed),
            synthetic_tool_call_arguments,
        ),
        BlockRole::System | BlockRole::User | BlockRole::Assistant | BlockRole::Tool => {
            let mut rng = block_seed_rng(seed);
            generate_synthetic_text_with_ids(
                tokenizer,
                eligible_token_ids,
                special_token_ids,
                block.tokens,
                &mut rng,
            )
        }
    };
    text.with_context(|| {
        format!(
            "failed to generate {} block content for seed {:?} ({} tokens)",
            block.role.as_str(),
            block.seed,
            block.tokens
        )
    })
}

/// Token ids whose text survives JSON string serialization byte-for-byte.
fn json_safe_token_ids_for(tokenizer: &Tokenizer, eligible_token_ids: &[u32]) -> Vec<u32> {
    let safe: Vec<u32> = eligible_token_ids
        .iter()
        .copied()
        .filter(|id| {
            tokenizer
                .decode(&[*id], false)
                .map(|text| {
                    text.chars().all(|character| {
                        character != '"' && character != '\\' && !character.is_control()
                    })
                })
                .unwrap_or(false)
        })
        .collect();
    if safe.is_empty() {
        eligible_token_ids.to_vec()
    } else {
        safe
    }
}

/// Generate text so that `wrap(text)` (a serialized JSON form) re-encodes to `target_tokens`.
///
/// The content length is adjusted by the measured difference until the wrapped form matches;
/// when the tokenizer cannot hit the target exactly the closest constructible form is used.
fn generate_wrapped_text<SeedRng, Wrap>(
    tokenizer: &Tokenizer,
    eligible_token_ids: &[u32],
    special_token_ids: &HashSet<u32>,
    target_tokens: usize,
    seed_rng: SeedRng,
    wrap: Wrap,
) -> Result<String>
where
    SeedRng: Fn() -> rand::rngs::StdRng,
    Wrap: Fn(&str) -> String,
{
    if target_tokens == 0 {
        return Ok(String::new());
    }
    let wrapped_tokens = |text: &str| -> Result<usize> {
        tokenizer
            .encode(wrap(text).as_str(), false)
            .map(|encoding| encoding.len())
            .map_err(|error| anyhow!("failed to encode wrapped synthetic text: {error}"))
    };
    let scaffold_tokens = wrapped_tokens("")?;
    let mut content_target = target_tokens.saturating_sub(scaffold_tokens).max(1);
    let mut tried = HashSet::new();
    let mut best: Option<(usize, String)> = None;

    while tried.len() < 8 && tried.insert(content_target) {
        let mut rng = seed_rng();
        let text = generate_synthetic_text_with_ids(
            tokenizer,
            eligible_token_ids,
            special_token_ids,
            content_target,
            &mut rng,
        )?;
        let actual_tokens = wrapped_tokens(&text)?;
        if actual_tokens == target_tokens {
            return Ok(text);
        }
        let distance = actual_tokens.abs_diff(target_tokens);
        if best
            .as_ref()
            .is_none_or(|(best_distance, _)| distance < *best_distance)
        {
            best = Some((distance, text));
        }
        let adjusted = content_target as i64 + target_tokens as i64 - actual_tokens as i64;
        content_target = adjusted.max(1) as usize;
    }

    best.map(|(_, text)| text)
        .ok_or_else(|| anyhow!("failed to construct wrapped synthetic text"))
}

/// Request content assembled from schema v2 blocks.
#[derive(Debug, Default)]
struct BlockRequest {
    tools: Vec<Value>,
    messages: Vec<Value>,
}

fn build_block_request(blocks: &[BlockSpec], cache: &BlockCache) -> Result<BlockRequest> {
    let mut request = BlockRequest::default();
    let mut pending_tool_call_ids = VecDeque::new();

    for block in blocks {
        let content = cache.get(&block.seed).ok_or_else(|| {
            anyhow!(
                "{} block with seed {:?} has no prepared content",
                block.role.as_str(),
                block.seed
            )
        })?;
        let text = match content {
            BlockContent::Generated(text) => text.as_ref(),
            BlockContent::Live(message) => {
                if block.role != BlockRole::Assistant {
                    return Err(anyhow!(
                        "live content cannot be used for a {} block",
                        block.role.as_str()
                    ));
                }
                request.messages.push(message.clone());
                continue;
            }
        };
        match block.role {
            BlockRole::ToolDefinition => request.tools.push(synthetic_tool_definition(
                &synthetic_tool_name(&block.seed),
                text,
            )),
            BlockRole::System => request
                .messages
                .push(json!({"role": "system", "content": text})),
            BlockRole::User => request
                .messages
                .push(json!({"role": "user", "content": text})),
            BlockRole::Assistant => request
                .messages
                .push(json!({"role": "assistant", "content": text})),
            BlockRole::Tool => {
                let tool_call_id = pending_tool_call_ids
                    .pop_front()
                    .unwrap_or_else(|| synthetic_block_tool_call_id(&block.seed));
                request.messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": text,
                }));
            }
            BlockRole::ToolCall => {
                let tool_call_id = synthetic_block_tool_call_id(&block.seed);
                let tool_call = json!({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": synthetic_tool_name(&block.seed),
                        "arguments": synthetic_tool_call_arguments(text),
                    }
                });
                attach_tool_call(&mut request.messages, tool_call)?;
                pending_tool_call_ids.push_back(tool_call_id);
            }
        }
    }
    Ok(request)
}

/// Attach a synthetic tool call to the preceding assistant message, creating one if needed.
fn attach_tool_call(messages: &mut Vec<Value>, tool_call: Value) -> Result<()> {
    let last_is_assistant = messages
        .last()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        == Some("assistant");
    if !last_is_assistant {
        messages.push(json!({"role": "assistant", "content": null, "tool_calls": []}));
    }
    let assistant = messages
        .last_mut()
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow!("assistant message is not an object"))?;
    match assistant.get_mut("tool_calls") {
        Some(Value::Array(tool_calls)) => tool_calls.push(tool_call),
        _ => {
            assistant.insert("tool_calls".to_string(), json!([tool_call]));
        }
    }
    Ok(())
}

fn synthetic_text_stream_id(field: SyntheticTextField) -> u64 {
    match field {
        SyntheticTextField::InitialPrompt => 0x1000_0000_0000_0000,
        SyntheticTextField::ResetPrompt(turn_index) => {
            0x2000_0000_0000_0000u64.wrapping_add((turn_index as u64).wrapping_mul(2))
        }
        SyntheticTextField::Environment(turn_index) => {
            0x2000_0000_0000_0001u64.wrapping_add((turn_index as u64).wrapping_mul(2))
        }
    }
}

fn synthetic_token_ids(tokenizer: &Tokenizer) -> Result<(Vec<u32>, HashSet<u32>)> {
    let vocab_size = tokenizer.get_vocab_size(false);
    if vocab_size == 0 {
        return Err(anyhow!("tokenizer vocabulary is empty"));
    }

    let special_token_ids: HashSet<u32> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter_map(|(id, token)| token.special.then_some(id))
        .collect();
    let eligible_token_ids: Vec<u32> = (0..vocab_size as u32)
        .filter(|id| !special_token_ids.contains(id))
        .collect();
    if eligible_token_ids.is_empty() {
        return Err(anyhow!(
            "tokenizer vocabulary contains no non-special tokens"
        ));
    }
    Ok((eligible_token_ids, special_token_ids))
}

#[cfg(test)]
fn generate_synthetic_text<R: Rng + ?Sized>(
    tokenizer: &Tokenizer,
    target_tokens: usize,
    rng: &mut R,
) -> Result<String> {
    if target_tokens == 0 {
        return Ok(String::new());
    }
    let (eligible_token_ids, special_token_ids) = synthetic_token_ids(tokenizer)?;
    generate_synthetic_text_with_ids(
        tokenizer,
        &eligible_token_ids,
        &special_token_ids,
        target_tokens,
        rng,
    )
}

fn generate_synthetic_text_with_ids<R: Rng + ?Sized>(
    tokenizer: &Tokenizer,
    eligible_token_ids: &[u32],
    special_token_ids: &HashSet<u32>,
    target_tokens: usize,
    rng: &mut R,
) -> Result<String> {
    if target_tokens == 0 {
        return Ok(String::new());
    }

    let mut text = decode_random_tokens(tokenizer, eligible_token_ids, target_tokens, rng)
        .context("failed to decode synthetic tokens")?;
    let mut last_actual_tokens = None;

    for _ in 0..256 {
        let encoding = tokenizer
            .encode(text.as_str(), false)
            .map_err(|err| anyhow!("failed to re-encode synthetic text: {}", err))?;
        let actual_tokens = encoding.len();
        last_actual_tokens = Some(actual_tokens);
        let encoded_special = encoding
            .get_ids()
            .iter()
            .any(|id| special_token_ids.contains(id));
        if encoded_special || text.is_empty() {
            text =
                resample_different_text(tokenizer, eligible_token_ids, target_tokens, &text, rng)
                    .context("failed to replace invalid synthetic text")?;
            continue;
        }
        if actual_tokens == target_tokens && !text.is_empty() {
            return Ok(text);
        }

        if actual_tokens > target_tokens {
            text = match repair_oversized_text(tokenizer, &text, &encoding, target_tokens)? {
                Some(candidate) => candidate,
                None => resample_different_text(
                    tokenizer,
                    eligible_token_ids,
                    target_tokens,
                    &text,
                    rng,
                )
                .context("failed to resample oversized synthetic text")?,
            };
        } else {
            let missing = target_tokens - actual_tokens;
            let extra = decode_random_tokens(tokenizer, eligible_token_ids, missing.max(1), rng)
                .context("failed to extend synthetic text")?;
            let original_length = text.len();
            if !text.ends_with(char::is_whitespace) {
                text.push(' ');
            }
            text.push_str(&extra);
            if text.len() == original_length {
                text = resample_different_text(
                    tokenizer,
                    eligible_token_ids,
                    target_tokens,
                    &text,
                    rng,
                )
                .context("failed to replace a non-progressing synthetic extension")?;
            }
        }
    }

    Err(anyhow!(
        "failed to generate synthetic text with exactly {target_tokens} tokens after 256 attempts (last count: {})",
        last_actual_tokens
            .map(|count| count.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    ))
}

fn decode_random_tokens<R: Rng + ?Sized>(
    tokenizer: &Tokenizer,
    eligible_token_ids: &[u32],
    count: usize,
    rng: &mut R,
) -> Result<String> {
    let ids: Vec<u32> = (0..count)
        .map(|_| eligible_token_ids[rng.gen_range(0..eligible_token_ids.len())])
        .collect();
    tokenizer
        .decode(&ids, false)
        .map_err(|error| anyhow!("tokenizer decode failed: {error}"))
}

fn resample_different_text<R: Rng + ?Sized>(
    tokenizer: &Tokenizer,
    eligible_token_ids: &[u32],
    target_tokens: usize,
    previous: &str,
    rng: &mut R,
) -> Result<String> {
    for _ in 0..8 {
        let candidate = decode_random_tokens(tokenizer, eligible_token_ids, target_tokens, rng)?;
        if candidate != previous {
            return Ok(candidate);
        }
    }
    Err(anyhow!(
        "tokenizer produced the same candidate in 8 consecutive deterministic resamples"
    ))
}

fn repair_oversized_text(
    tokenizer: &Tokenizer,
    text: &str,
    encoding: &tokenizers::Encoding,
    target_tokens: usize,
) -> Result<Option<String>> {
    let mut boundaries_examined = 0usize;
    let mut previous_boundary = None;
    let mut best_under_target: Option<(usize, String)> = None;

    for &(_, end_offset) in encoding.get_offsets()[..target_tokens].iter().rev() {
        if previous_boundary == Some(end_offset) {
            continue;
        }
        previous_boundary = Some(end_offset);
        if end_offset == 0 || end_offset >= text.len() || !text.is_char_boundary(end_offset) {
            continue;
        }

        let candidate = &text[..end_offset];
        let candidate_tokens = tokenizer
            .encode(candidate, false)
            .map_err(|err| anyhow!("failed to verify trimmed synthetic text: {err}"))?
            .len();
        if candidate_tokens == target_tokens {
            return Ok(Some(candidate.to_string()));
        }
        if candidate_tokens < target_tokens
            && best_under_target
                .as_ref()
                .is_none_or(|(best_count, _)| candidate_tokens > *best_count)
        {
            best_under_target = Some((candidate_tokens, candidate.to_string()));
        }

        boundaries_examined += 1;
        if boundaries_examined >= 16 {
            break;
        }
    }

    Ok(best_under_target.map(|(_, candidate)| candidate))
}

async fn generate_next_reset_prompt(
    text_generator: &SyntheticTextGenerator,
    plan: &AgentPlan,
    current_turn_index: usize,
) -> Result<Option<String>> {
    let next_turn_index = current_turn_index + 1;
    let Some(reset_prompt_tokens) = plan
        .turns
        .get(next_turn_index)
        .and_then(|turn| turn.reset_prompt_tokens)
    else {
        return Ok(None);
    };

    text_generator
        .generate(
            plan.agent_id,
            SyntheticTextField::ResetPrompt(next_turn_index),
            reset_prompt_tokens,
        )
        .await
        .map(Some)
        .with_context(|| {
            format!(
                "failed to prefetch reset prompt for trajectory {:?} invocation {}",
                plan.trajectory_id,
                next_turn_index + 1
            )
        })
}

async fn run_agent(
    mut plan: AgentPlan,
    routing_slot: usize,
    admitted_at: Duration,
    client: Client,
    config: std::sync::Arc<AgentLoopConfig>,
    text_generator: SyntheticTextGenerator,
) -> Result<AgentWorkerReport> {
    let initial_content = plan.initial_content.take().ok_or_else(|| {
        anyhow!(
            "trajectory {:?} was admitted without materialized initial content",
            plan.trajectory_id
        )
    })?;
    match (initial_content, plan.uses_blocks()) {
        (InitialContent::Prompt(initial_prompt), false) => {
            run_growth_agent(
                plan,
                initial_prompt,
                routing_slot,
                admitted_at,
                client,
                config,
                text_generator,
            )
            .await
        }
        (InitialContent::Blocks(cache), true) => {
            run_block_agent(
                plan,
                cache,
                routing_slot,
                admitted_at,
                client,
                config,
                text_generator,
            )
            .await
        }
        _ => Err(anyhow!(
            "trajectory {:?} initial content does not match its plan shape",
            plan.trajectory_id
        )),
    }
}

/// Schema v1 semantics: a growing conversation where each turn appends the model reply and a
/// synthetic environment response sized from the manifest.
async fn run_growth_agent(
    plan: AgentPlan,
    initial_prompt: String,
    routing_slot: usize,
    admitted_at: Duration,
    client: Client,
    config: std::sync::Arc<AgentLoopConfig>,
    text_generator: SyntheticTextGenerator,
) -> Result<AgentWorkerReport> {
    let agent_start = Instant::now();
    let mut report = AgentWorkerReport::new(
        plan.agent_id,
        plan.trajectory_id.clone(),
        routing_slot,
        admitted_at,
    );
    let mut messages = if config.dry_run {
        Vec::new()
    } else {
        vec![json!({
            "role": "user",
            "content": initial_prompt,
        })]
    };
    let mut previous_prompt_tokens = 0u64;
    let mut prefetched_reset_prompt = None;

    for (turn_index, turn) in plan.turns.iter().enumerate() {
        let invocation = turn_index + 1;
        if turn.reset_prompt_tokens.is_some() {
            let reset_prompt = prefetched_reset_prompt.take().ok_or_else(|| {
                anyhow!(
                    "trajectory {:?} invocation {} is missing its prefetched reset prompt",
                    plan.trajectory_id,
                    invocation
                )
            })?;
            if !config.dry_run {
                messages = vec![json!({
                    "role": "user",
                    "content": reset_prompt,
                })];
            }
            previous_prompt_tokens = 0;
        }
        if config.dry_run {
            let environment = text_generator.generate(
                plan.agent_id,
                SyntheticTextField::Environment(turn_index),
                turn.environment_tokens,
            );
            let next_reset = generate_next_reset_prompt(&text_generator, &plan, turn_index);
            let (environment_content, next_reset_prompt) = tokio::join!(environment, next_reset);
            environment_content.with_context(|| {
                format!(
                    "failed to generate environment content for trajectory {:?} invocation {}",
                    plan.trajectory_id, invocation
                )
            })?;
            prefetched_reset_prompt = next_reset_prompt?;
            report.dry_run_records.push(dry_run_record(
                &plan,
                turn,
                routing_slot,
                admitted_at,
                invocation,
            ));
            continue;
        }

        let body = serialize_request_body_with(
            &config,
            &messages,
            RequestOptions {
                output_tokens: turn.output_tokens,
                max_tokens: turn.max_tokens,
                stream: turn.stream.unwrap_or(false),
                tools: None,
            },
            plan.user_tag.as_deref(),
        )?;
        let request = request_with_retries(
            &client,
            &config,
            &body,
            plan.user_tag.as_deref(),
            routing_slot,
            turn.stream.unwrap_or(false),
        );
        let environment = text_generator.generate(
            plan.agent_id,
            SyntheticTextField::Environment(turn_index),
            turn.environment_tokens,
        );
        let next_reset = generate_next_reset_prompt(&text_generator, &plan, turn_index);
        let (request_result, environment_content, next_reset_prompt) =
            tokio::join!(request, environment, next_reset);
        prefetched_reset_prompt = next_reset_prompt?;
        let environment_content = environment_content.with_context(|| {
            format!(
                "failed to generate environment content for trajectory {:?} invocation {}",
                plan.trajectory_id, invocation
            )
        })?;

        match request_result {
            Ok(result) => {
                record_success(&mut report, &result, &mut previous_prompt_tokens);

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
                    &environment_content,
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

/// Schema v2 semantics: every request's prompt is assembled from its blocks. Generated block
/// text is cached per trajectory, and live assistant blocks carry the model's previous reply.
async fn run_block_agent(
    plan: AgentPlan,
    mut cache: BlockCache,
    routing_slot: usize,
    admitted_at: Duration,
    client: Client,
    config: std::sync::Arc<AgentLoopConfig>,
    text_generator: SyntheticTextGenerator,
) -> Result<AgentWorkerReport> {
    let agent_start = Instant::now();
    let mut report = AgentWorkerReport::new(
        plan.agent_id,
        plan.trajectory_id.clone(),
        routing_slot,
        admitted_at,
    );
    let mut previous_prompt_tokens = 0u64;
    let mut last_reply: Option<Value> = None;
    let mut prefetched: Vec<(String, std::sync::Arc<str>)> = Vec::new();

    for (turn_index, turn) in plan.turns.iter().enumerate() {
        let invocation = turn_index + 1;
        let blocks = turn.blocks.as_deref().ok_or_else(|| {
            anyhow!(
                "trajectory {:?} invocation {} has no blocks",
                plan.trajectory_id,
                invocation
            )
        })?;
        if turn.reset_before {
            cache.clear();
            last_reply = None;
            previous_prompt_tokens = 0;
        }
        for (seed, text) in prefetched.drain(..) {
            cache.entry(seed).or_insert(BlockContent::Generated(text));
        }
        let missing = missing_generated_blocks(blocks, &cache);
        if !missing.is_empty() {
            for (seed, text) in
                text_generator
                    .generate_blocks(missing)
                    .await
                    .with_context(|| {
                        format!(
                            "failed to generate blocks for trajectory {:?} invocation {}",
                            plan.trajectory_id, invocation
                        )
                    })?
            {
                cache.insert(seed, BlockContent::Generated(text));
            }
        }

        if config.dry_run {
            let next_blocks = plan
                .turns
                .get(turn_index + 1)
                .and_then(|next| next.blocks.as_deref())
                .map(|next| missing_generated_blocks(next, &cache))
                .unwrap_or_default();
            prefetched = text_generator.generate_blocks(next_blocks).await?;
            report.dry_run_records.push(dry_run_record(
                &plan,
                turn,
                routing_slot,
                admitted_at,
                invocation,
            ));
            continue;
        }

        report.live_block_fallbacks +=
            resolve_live_blocks(blocks, &mut cache, &mut last_reply, &text_generator)
                .await
                .with_context(|| {
                    format!(
                        "failed to resolve live blocks for trajectory {:?} invocation {}",
                        plan.trajectory_id, invocation
                    )
                })?;

        let block_request = build_block_request(blocks, &cache).with_context(|| {
            format!(
                "failed to assemble request for trajectory {:?} invocation {}",
                plan.trajectory_id, invocation
            )
        })?;
        let stream = turn.stream.unwrap_or(false);
        let body = serialize_request_body_with(
            &config,
            &block_request.messages,
            RequestOptions {
                output_tokens: turn.output_tokens,
                max_tokens: turn.max_tokens,
                stream,
                tools: (!block_request.tools.is_empty()).then_some(block_request.tools.as_slice()),
            },
            plan.user_tag.as_deref(),
        )?;
        let request = request_with_retries(
            &client,
            &config,
            &body,
            plan.user_tag.as_deref(),
            routing_slot,
            stream,
        );
        let next_blocks = plan
            .turns
            .get(turn_index + 1)
            .and_then(|next| next.blocks.as_deref())
            .map(|next| missing_generated_blocks(next, &cache))
            .unwrap_or_default();
        let prefetch = text_generator.generate_blocks(next_blocks);
        let (request_result, prefetched_now) = tokio::join!(request, prefetch);
        prefetched = prefetched_now.with_context(|| {
            format!(
                "failed to prefetch blocks for trajectory {:?} invocation {}",
                plan.trajectory_id,
                invocation + 1
            )
        })?;

        match request_result {
            Ok(result) => {
                record_success(&mut report, &result, &mut previous_prompt_tokens);
                last_reply = Some(normalize_assistant_message(result.assistant_message)?);

                if !turn.tool_call_latency.is_zero() {
                    tokio::time::sleep(turn.tool_call_latency).await;
                    report.tool_call_latency = report
                        .tool_call_latency
                        .saturating_add(turn.tool_call_latency);
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

/// Substitute the previous reply into the first unseen live block; any further unseen live
/// block (or a live block with no previous reply) is generated instead. Returns the fallback count.
async fn resolve_live_blocks(
    blocks: &[BlockSpec],
    cache: &mut BlockCache,
    last_reply: &mut Option<Value>,
    text_generator: &SyntheticTextGenerator,
) -> Result<u64> {
    let mut fallbacks = 0;
    for block in blocks.iter().filter(|block| block.is_live()) {
        if cache.contains_key(&block.seed) {
            continue;
        }
        match last_reply.take() {
            Some(reply) => {
                cache.insert(block.seed.clone(), BlockContent::Live(reply));
            }
            None => {
                let text = text_generator.generate_block(block).await?;
                cache.insert(block.seed.clone(), BlockContent::Generated(text));
                fallbacks += 1;
            }
        }
    }
    Ok(fallbacks)
}

fn record_success(
    report: &mut AgentWorkerReport,
    result: &RequestResult,
    previous_prompt_tokens: &mut u64,
) {
    report.successful_requests += 1;
    report.input_tokens = report.input_tokens.saturating_add(result.prompt_tokens);
    report.output_tokens = report
        .output_tokens
        .saturating_add(result.completion_tokens);
    report.estimated_cached_input_tokens =
        report
            .estimated_cached_input_tokens
            .saturating_add(estimated_cache_hit(
                *previous_prompt_tokens,
                result.prompt_tokens,
            ));
    *previous_prompt_tokens = result.prompt_tokens;
    report.latencies.push(result.latency);
}

fn dry_run_record(
    plan: &AgentPlan,
    turn: &AgentTurnPlan,
    routing_slot: usize,
    admitted_at: Duration,
    invocation: usize,
) -> DryRunRecord {
    DryRunRecord {
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
        reset_before: turn.reset_prompt_tokens.is_some() || turn.reset_before,
        blocks: turn.blocks.as_deref().map(|blocks| {
            (
                blocks.len(),
                blocks.iter().filter(|block| block.is_live()).count(),
            )
        }),
        stream: turn.stream,
        max_tokens: turn.max_tokens,
    }
}

fn generate_user_tag(enabled: bool, prefix: Option<&str>, agent_id: usize) -> Option<String> {
    enabled.then(|| match prefix {
        Some(prefix) => format!("{prefix}-{agent_id}"),
        None => Uuid::new_v4().to_string(),
    })
}

#[derive(Serialize)]
struct AgentRequestPayload<'a> {
    model: &'a str,
    messages: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_new_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ignore_eos: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

/// Per-request shape: the planned output length plus schema v2 overrides.
#[derive(Clone, Copy, Debug, Default)]
struct RequestOptions<'a> {
    output_tokens: usize,
    /// Output cap override; the planned output length remains the floor (clamped to the cap).
    max_tokens: Option<usize>,
    stream: bool,
    tools: Option<&'a [Value]>,
}

fn serialize_request_body_with(
    config: &AgentLoopConfig,
    messages: &[Value],
    options: RequestOptions<'_>,
    user_tag: Option<&str>,
) -> Result<Bytes> {
    let output_cap = options.max_tokens.unwrap_or(options.output_tokens);
    let output_floor = options.output_tokens.min(output_cap);
    let payload = AgentRequestPayload {
        model: &config.model,
        messages,
        tools: options.tools,
        max_tokens: (!config.sglang).then_some(output_cap),
        min_tokens: (!config.sglang).then_some(output_floor),
        max_new_tokens: config.sglang.then_some(output_cap),
        min_new_tokens: config.sglang.then_some(output_floor),
        ignore_eos: config.ignore_eos.then_some(true),
        stream: options.stream.then_some(true),
        stream_options: options.stream.then(|| json!({"include_usage": true})),
        user: user_tag,
    };
    serde_json::to_vec(&payload)
        .map(Bytes::from)
        .context("failed to serialize agent request")
}

#[cfg(test)]
fn build_request_body(
    config: &AgentLoopConfig,
    messages: &[Value],
    output_tokens: usize,
    user_tag: Option<&str>,
) -> Value {
    serde_json::from_slice(
        &serialize_request_body_with(
            config,
            messages,
            RequestOptions {
                output_tokens,
                ..RequestOptions::default()
            },
            user_tag,
        )
        .unwrap(),
    )
    .unwrap()
}

async fn request_with_retries(
    client: &Client,
    config: &AgentLoopConfig,
    body: &Bytes,
    user_tag: Option<&str>,
    routing_slot: usize,
    stream: bool,
) -> Result<RequestResult> {
    let start = Instant::now();
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        match single_attempt(client, config, body, user_tag, routing_slot, stream).await {
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
    body: &Bytes,
    user_tag: Option<&str>,
    routing_slot: usize,
    stream: bool,
) -> Result<RequestResult> {
    if config.verbose {
        let parsed: Value = serde_json::from_slice(body)
            .context("failed to parse serialized request for verbose logging")?;
        println!("[AGENT REQUEST] {}", sanitize_request(&parsed));
    }

    let request = client
        .post(config.endpoint.clone())
        .headers(build_request_headers(config, user_tag, routing_slot)?);
    let mut response = request.body(body.clone()).send().await?;
    let status = response.status();
    if !status.is_success() {
        let bytes = response.bytes().await?;
        return Err(anyhow!(
            "request failed ({}) {}",
            status,
            truncate_text(&String::from_utf8_lossy(&bytes), 500)
        ));
    }

    let (usage, assistant_message) = if stream {
        let mut lines = SseLineBuffer::default();
        let mut completion = StreamedCompletion::default();
        while let Some(chunk) = response.chunk().await? {
            for payload in lines.push(&chunk) {
                completion.absorb(&payload)?;
            }
        }
        if let Some(payload) = lines.finish() {
            completion.absorb(&payload)?;
        }
        let (assistant_message, usage) = completion.into_parts()?;
        if config.verbose {
            println!(
                "[AGENT RESPONSE] {}",
                sanitize_response(&json!({
                    "choices": [{"message": assistant_message}],
                    "usage": usage,
                }))
            );
        }
        (usage, assistant_message)
    } else {
        let bytes = response.bytes().await?;
        let payload: Value = serde_json::from_slice(&bytes)?;
        if config.verbose {
            println!("[AGENT RESPONSE] {}", sanitize_response(&payload));
        }
        let usage = payload
            .get("usage")
            .cloned()
            .ok_or_else(|| anyhow!("response missing usage field"))?;
        let assistant_message = payload
            .pointer("/choices/0/message")
            .cloned()
            .ok_or_else(|| anyhow!("response missing choices[0].message"))?;
        (usage, assistant_message)
    };
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("usage.prompt_tokens missing or not an integer"))?;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    Ok(RequestResult {
        prompt_tokens,
        completion_tokens,
        assistant_message,
        latency: Duration::ZERO,
    })
}

/// Splits a server-sent-events byte stream into complete `data:` payloads.
#[derive(Debug, Default)]
struct SseLineBuffer {
    pending: Vec<u8>,
}

impl SseLineBuffer {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(chunk);
        let mut payloads = Vec::new();
        while let Some(newline) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=newline).collect();
            if let Some(payload) = sse_data_payload(&line[..line.len() - 1]) {
                payloads.push(payload);
            }
        }
        payloads
    }

    fn finish(&mut self) -> Option<String> {
        let line = std::mem::take(&mut self.pending);
        sse_data_payload(&line)
    }
}

fn sse_data_payload(line: &[u8]) -> Option<String> {
    let line = String::from_utf8_lossy(line);
    let line = line.trim_end_matches('\r');
    let payload = line.strip_prefix("data:")?.trim();
    (!payload.is_empty()).then(|| payload.to_string())
}

/// Assistant message and usage accumulated from streamed chat-completion chunks.
#[derive(Debug, Default)]
struct StreamedCompletion {
    content: String,
    reasoning_content: Option<String>,
    usage: Option<Value>,
    chunks: usize,
}

impl StreamedCompletion {
    fn absorb(&mut self, payload: &str) -> Result<()> {
        if payload == "[DONE]" {
            return Ok(());
        }
        let event: Value =
            serde_json::from_str(payload).context("invalid JSON in streamed response chunk")?;
        self.chunks += 1;
        if let Some(usage) = event.get("usage").filter(|usage| !usage.is_null()) {
            self.usage = Some(usage.clone());
        }
        if let Some(delta) = event.pointer("/choices/0/delta") {
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                self.content.push_str(text);
            }
            if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) {
                self.reasoning_content
                    .get_or_insert_with(String::new)
                    .push_str(text);
            }
        }
        Ok(())
    }

    fn into_parts(self) -> Result<(Value, Value)> {
        if self.chunks == 0 {
            return Err(anyhow!("streamed response contained no chunks"));
        }
        let usage = self
            .usage
            .ok_or_else(|| anyhow!("streamed response missing usage; the backend must honor stream_options.include_usage"))?;
        let mut message = Map::new();
        message.insert("role".to_string(), Value::String("assistant".to_string()));
        message.insert("content".to_string(), Value::String(self.content));
        if let Some(reasoning_content) = self.reasoning_content {
            message.insert(
                "reasoning_content".to_string(),
                Value::String(reasoning_content),
            );
        }
        Ok((Value::Object(message), usage))
    }
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
            truncate_string_field(message, "content");
            if let Some(tool_calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for tool_call in tool_calls {
                    if let Some(function) = tool_call.get_mut("function") {
                        truncate_string_field(function, "arguments");
                    }
                }
            }
        }
    }
    if let Some(tools) = sanitized.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools {
            if let Some(function) = tool.get_mut("function") {
                truncate_string_field(function, "description");
            }
        }
    }
    sanitized
}

fn truncate_string_field(object: &mut Value, key: &str) {
    if let Some(field) = object.get_mut(key) {
        if let Some(text) = field.as_str() {
            *field = Value::String(truncate_text(text, 50));
        }
    }
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

    fn byte_level_tokenizer() -> Tokenizer {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;

        let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let vocab: Vocab = alphabet
            .into_iter()
            .enumerate()
            .map(|(id, character)| (character.to_string(), id as u32))
            .collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer
    }

    fn word_level_tokenizer() -> Tokenizer {
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
        tokenizer
    }

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
    fn synthetic_text_resamples_when_byte_offsets_cannot_be_trimmed() {
        let tokenizer = byte_level_tokenizer();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let text = generate_synthetic_text(&tokenizer, 1, &mut rng).unwrap();
        assert_eq!(tokenizer.encode(text, false).unwrap().len(), 1);
    }

    #[test]
    fn synthetic_text_handles_the_glm_regression_target_exactly() {
        let tokenizer = byte_level_tokenizer();
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);

        let text = generate_synthetic_text(&tokenizer, 2_857, &mut rng).unwrap();
        assert_eq!(tokenizer.encode(text, false).unwrap().len(), 2_857);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lazy_text_is_deterministic_across_scheduling_and_unique_per_agent() {
        let tokenizer = std::sync::Arc::new(word_level_tokenizer());
        let generator = SyntheticTextGenerator::new(tokenizer.clone(), 42).unwrap();

        let initial = generator
            .generate(0, SyntheticTextField::InitialPrompt, 64)
            .await
            .unwrap();
        let environment = generator
            .generate(0, SyntheticTextField::Environment(7), 64)
            .await
            .unwrap();
        let (environment_reordered, initial_reordered) = tokio::join!(
            generator.generate(0, SyntheticTextField::Environment(7), 64),
            generator.generate(0, SyntheticTextField::InitialPrompt, 64),
        );
        assert_eq!(environment_reordered.unwrap(), environment);
        assert_eq!(initial_reordered.unwrap(), initial);

        let other_agent = generator
            .generate(1, SyntheticTextField::InitialPrompt, 64)
            .await
            .unwrap();
        assert_ne!(other_agent, initial);
        for text in [&initial, &environment, &other_agent] {
            assert_eq!(tokenizer.encode(text.as_str(), false).unwrap().len(), 64);
        }
        assert!(generator.permits.available_permits() <= 32);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn initial_prompts_are_materialized_before_admission() {
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            2,
            SampleSpec::fixed(32).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(2).unwrap(),
        )
        .unwrap()
        .with_seed(7);
        let (plans, root_seed, schema_version) = build_agent_plans(&config).unwrap();
        assert_eq!(schema_version, None);
        assert!(plans.iter().all(|plan| plan.initial_content.is_none()));
        let tokenizer = std::sync::Arc::new(word_level_tokenizer());
        let generator = SyntheticTextGenerator::new(tokenizer.clone(), root_seed).unwrap();
        let admissions = plans.into_iter().enumerate().collect();

        let prepared = materialize_initial_prompts(admissions, &generator)
            .await
            .unwrap();

        assert_eq!(prepared.len(), 2);
        let prompts: Vec<_> = prepared
            .iter()
            .map(|(_, plan)| match plan.initial_content.as_ref().unwrap() {
                InitialContent::Prompt(prompt) => prompt,
                InitialContent::Blocks(_) => panic!("synthetic plans use a single prompt"),
            })
            .collect();
        assert_ne!(prompts[0], prompts[1]);
        assert!(prompts.iter().all(|prompt| tokenizer
            .encode(prompt.as_str(), false)
            .unwrap()
            .len()
            == 32));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reset_prefetch_matches_direct_deterministic_generation() {
        let input = r#"{"schema_version":1,"trajectory_id":"reset","requests":[{"prompt_tokens":32,"output_tokens":4},{"prompt_tokens":24,"output_tokens":3,"reset_before":true}]}"#;
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
        .unwrap();
        let plans = build_replay_agent_plans(&config, specs).unwrap();
        let tokenizer = std::sync::Arc::new(word_level_tokenizer());
        let generator = SyntheticTextGenerator::new(tokenizer.clone(), 99).unwrap();

        let prefetched = generate_next_reset_prompt(&generator, &plans[0], 0)
            .await
            .unwrap()
            .unwrap();
        let direct = generator
            .generate(0, SyntheticTextField::ResetPrompt(1), 24)
            .await
            .unwrap();

        assert_eq!(prefetched, direct);
        assert_eq!(tokenizer.encode(prefetched, false).unwrap().len(), 24);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn initial_agents_are_admitted_at_benchmark_time_zero() {
        let tokenizer_path = std::env::temp_dir().join(format!(
            "batchbench-agent-tokenizer-{}.json",
            Uuid::new_v4()
        ));
        word_level_tokenizer().save(&tokenizer_path, false).unwrap();
        let config = AgentLoopConfig::try_new(
            "http://localhost:8000/v1/chat/completions",
            None,
            "test-model",
            8,
            SampleSpec::fixed(32).unwrap(),
            SampleSpec::fixed(4).unwrap(),
            SampleSpec::fixed(6).unwrap(),
            SampleSpec::fixed(1).unwrap(),
        )
        .unwrap()
        .with_tokenizer_model(tokenizer_path.display().to_string())
        .with_seed(7)
        .with_dry_run(true)
        .with_max_active_agents(4)
        .unwrap();

        let report = run_agent_benchmark(config).await.unwrap();
        std::fs::remove_file(&tokenizer_path).unwrap();

        assert_eq!(report.completed_agents, 8);
        assert_eq!(report.max_active_agents, 4);
        assert!(report
            .agent_lifecycles
            .iter()
            .filter(|record| record.agent_id < 4)
            .all(|record| record.admitted_at == Duration::ZERO));
        assert!(report
            .agent_lifecycles
            .iter()
            .filter(|record| record.agent_id >= 4)
            .all(|record| record.admitted_at > Duration::ZERO));
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
        // prompt 100 / output 20 followed by prompt 150: 30 environment tokens without overhead,
        // 20 once the appended turn carries 10 tokens of template overhead.
        assert_eq!(
            inferred_environment_tokens_from_content(100, 20, 150).unwrap(),
            30
        );
        assert_eq!(
            inferred_environment_tokens_from_content(100, 20, 140).unwrap(),
            20
        );
        assert!(inferred_environment_tokens_from_content(100, 20, 110).is_err());
    }

    #[test]
    fn effective_overheads_follow_globals_unless_overridden() {
        let input = r#"{"schema_version":2,"trajectory_id":"o","requests":[{"prompt_tokens":100,"output_tokens":20},{"prompt_tokens":150,"output_tokens":10},{"prompt_tokens":200,"output_tokens":10,"overhead_tokens":7},{"prompt_tokens":250,"output_tokens":10},{"prompt_tokens":60,"output_tokens":5,"reset_before":true}]}"#;
        let specs = parse_trajectory_plan_specs(std::io::Cursor::new(input), "fixture").unwrap();
        assert_eq!(
            effective_overheads(&specs[0].requests, 5, 10),
            vec![5, 15, 7, 17, 5]
        );
        assert_eq!(
            effective_overheads(&specs[0].requests, 0, 0),
            vec![0, 0, 7, 7, 0]
        );
    }

    #[test]
    fn replay_compiler_applies_overhead_and_reset_boundaries() {
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

        let plans = build_replay_agent_plans(&config, specs).unwrap();
        let plan = &plans[0];
        assert_eq!(plan.trajectory_id, "shape");
        assert_eq!(plan.initial_prompt_tokens, 95);
        assert!(plan.initial_content.is_none());
        assert_eq!(plan.turns.len(), 3);
        assert_eq!(plan.turns[0].input_content_tokens, 95);
        assert_eq!(plan.turns[0].target_prompt_tokens, 100);
        assert_eq!(plan.turns[0].environment_tokens, 20);
        assert_eq!(plan.turns[1].input_content_tokens, 135);
        assert_eq!(plan.turns[1].target_prompt_tokens, 150);
        assert_eq!(plan.turns[1].environment_tokens, 0);
        assert_eq!(plan.turns[1].tool_call_latency, Duration::from_millis(25));
        assert!(plan.turns[1].reset_prompt_tokens.is_none());
        assert_eq!(plan.turns[2].input_content_tokens, 55);
        assert_eq!(plan.turns[2].target_prompt_tokens, 60);
        assert_eq!(plan.turns[2].reset_prompt_tokens, Some(55));
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
        let wrong_version = r#"{"schema_version":3,"trajectory_id":"v3","requests":[{"prompt_tokens":10,"output_tokens":1}]}"#;
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn near_simultaneous_replacements_prepare_concurrently() {
        let mut preparations: JoinSet<Result<(usize, &'static str)>> = JoinSet::new();
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(3));
        let started = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        for (routing_slot, item) in [(0, "first"), (1, "second")] {
            let barrier = barrier.clone();
            let started = started.clone();
            spawn_admission_preparation(
                &mut preparations,
                routing_slot,
                item,
                move |item| async move {
                    started.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    barrier.wait().await;
                    Ok(item)
                },
            );
        }

        tokio::time::timeout(Duration::from_secs(1), barrier.wait())
            .await
            .expect("both replacement preparations should start before either is admitted");
        assert_eq!(started.load(std::sync::atomic::Ordering::SeqCst), 2);

        let mut prepared = Vec::new();
        while let Some(joined) = preparations.join_next().await {
            prepared.push(joined.unwrap().unwrap());
        }
        prepared.sort_by_key(|(routing_slot, _)| *routing_slot);
        assert_eq!(prepared, vec![(0, "first"), (1, "second")]);
    }

    #[test]
    fn out_of_order_preparations_are_admitted_in_fifo_order() {
        let mut prepared = BTreeMap::from([(1, "second")]);
        let mut next_sequence = 0;
        assert!(
            take_prepared_in_fifo_order(&mut prepared, &mut next_sequence)
                .unwrap()
                .is_empty()
        );

        prepared.insert(0, "first");
        assert_eq!(
            take_prepared_in_fifo_order(&mut prepared, &mut next_sequence).unwrap(),
            vec!["first", "second"]
        );
        assert_eq!(next_sequence, 2);
        assert!(prepared.is_empty());
    }

    #[test]
    fn rolling_admission_validates_its_limit() {
        assert!(RollingAdmission::new(vec![1], 0).is_err());
        let mut admission = RollingAdmission::new(vec![1], 2).unwrap();
        assert_eq!(admission.initial_admissions(), vec![(0, 1)]);
    }

    fn replay_config() -> AgentLoopConfig {
        AgentLoopConfig::try_new(
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
    }

    const V2_BLOCK_MANIFEST: &str = concat!(
        r#"{"schema_version":2,"trajectory_id":"first","start_after_ms":0,"requests":["#,
        r#"{"prompt_tokens":30,"output_tokens":4,"overhead_tokens":6,"delay_after_ms":200,"stream":true,"max_tokens":16,"blocks":[{"seed":"tools","tokens":8,"role":"tool_definition"},{"seed":"sys","tokens":10,"role":"system"},{"seed":"u1","tokens":6,"role":"user"}]},"#,
        r#"{"prompt_tokens":46,"output_tokens":3,"overhead_tokens":8,"blocks":[{"seed":"tools","tokens":8,"role":"tool_definition"},{"seed":"sys","tokens":10,"role":"system"},{"seed":"u1","tokens":6,"role":"user"},{"seed":"a1","tokens":4,"role":"assistant","live":true},{"seed":"u2","tokens":10,"role":"user"}]}]}"#,
        "\n",
        r#"{"schema_version":2,"trajectory_id":"second","start_after_ms":400,"requests":["#,
        r#"{"prompt_tokens":24,"output_tokens":2,"overhead_tokens":6,"blocks":[{"seed":"tools","tokens":8,"role":"tool_definition"},{"seed":"sys","tokens":10,"role":"system"},{"seed":"u9","tokens":0,"role":"user"}]}]}"#,
        "\n"
    );

    #[test]
    fn v1_manifests_reject_schema_v2_fields() {
        for (field, request_extra, trajectory_extra) in [
            ("overhead_tokens", r#","overhead_tokens":3"#, ""),
            ("stream", r#","stream":true"#, ""),
            ("max_tokens", r#","max_tokens":8"#, ""),
            (
                "blocks",
                r#","blocks":[{"seed":"s","tokens":10,"role":"user"}]"#,
                "",
            ),
            ("start_after_ms", "", r#","start_after_ms":10"#),
        ] {
            let input = format!(
                r#"{{"schema_version":1,"trajectory_id":"v1"{trajectory_extra},"requests":[{{"prompt_tokens":10,"output_tokens":1{request_extra}}}]}}"#
            );
            let error =
                parse_trajectory_plan_specs(std::io::Cursor::new(input), "fixture").unwrap_err();
            let message = format!("{error:#}");
            assert!(
                message.contains(field) && message.contains("requires schema_version 2"),
                "{field}: {message}"
            );
        }
    }

    #[test]
    fn v2_manifest_parses_blocks_and_open_loop_fields() {
        let specs = parse_trajectory_plan_specs(std::io::Cursor::new(V2_BLOCK_MANIFEST), "fixture")
            .unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[1].start_after_ms, Some(400));
        let first = &specs[0].requests[0];
        assert_eq!(first.overhead_tokens, Some(6));
        assert_eq!(first.stream, Some(true));
        assert_eq!(first.max_tokens, Some(16));
        let blocks = first.blocks.as_ref().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].role, BlockRole::ToolDefinition);
        assert!(!blocks[0].live);
        assert!(specs[0].requests[1].blocks.as_ref().unwrap()[3].is_live());
        assert_eq!(first.block_content_tokens().unwrap().unwrap(), 24);
    }

    #[test]
    fn v2_manifest_rejects_invalid_blocks_and_mixed_shapes() {
        let unknown_role = r#"{"schema_version":2,"trajectory_id":"r","requests":[{"prompt_tokens":10,"output_tokens":1,"blocks":[{"seed":"s","tokens":10,"role":"narrator"}]}]}"#;
        let error =
            parse_trajectory_plan_specs(std::io::Cursor::new(unknown_role), "fixture").unwrap_err();
        assert!(format!("{error:#}").contains("unknown variant"));

        let live_user = r#"{"schema_version":2,"trajectory_id":"l","requests":[{"prompt_tokens":10,"output_tokens":1,"blocks":[{"seed":"s","tokens":10,"role":"user","live":true}]}]}"#;
        let error =
            parse_trajectory_plan_specs(std::io::Cursor::new(live_user), "fixture").unwrap_err();
        assert!(format!("{error:#}").contains("only assistant blocks can be live"));

        let mixed = r#"{"schema_version":2,"trajectory_id":"m","requests":[{"prompt_tokens":10,"output_tokens":1,"blocks":[{"seed":"s","tokens":10,"role":"user"}]},{"prompt_tokens":20,"output_tokens":1}]}"#;
        let error =
            parse_trajectory_plan_specs(std::io::Cursor::new(mixed), "fixture").unwrap_err();
        assert!(format!("{error:#}").contains("must define blocks"));

        let unknown_block_field = r#"{"schema_version":2,"trajectory_id":"u","requests":[{"prompt_tokens":10,"output_tokens":1,"blocks":[{"seed":"s","tokens":10,"role":"user","hash":"x"}]}]}"#;
        let error =
            parse_trajectory_plan_specs(std::io::Cursor::new(unknown_block_field), "fixture")
                .unwrap_err();
        assert!(format!("{error:#}").contains("unknown field"));

        // Schema v2 allows reset_before on the first request: the session predates the window.
        let first_reset = r#"{"schema_version":2,"trajectory_id":"f","requests":[{"prompt_tokens":10,"output_tokens":1,"reset_before":true}]}"#;
        assert!(parse_trajectory_plan_specs(std::io::Cursor::new(first_reset), "fixture").is_ok());
        let v1_first_reset = first_reset.replace("\"schema_version\":2", "\"schema_version\":1");
        assert!(
            parse_trajectory_plan_specs(std::io::Cursor::new(v1_first_reset), "fixture").is_err()
        );
    }

    #[test]
    fn v2_per_request_overhead_overrides_the_global_flags() {
        let input = r#"{"schema_version":2,"trajectory_id":"o","requests":[{"prompt_tokens":100,"output_tokens":20},{"prompt_tokens":150,"output_tokens":10,"overhead_tokens":30},{"prompt_tokens":200,"output_tokens":5}]}"#;
        let specs = parse_trajectory_plan_specs(std::io::Cursor::new(input), "fixture").unwrap();
        let config = replay_config().with_replay_prompt_overhead(5, 10);
        let plans = build_replay_agent_plans(&config, specs).unwrap();
        let turns = &plans[0].turns;
        // content: 95, 120 (150 - 30), 160 (200 - (30 + 10)); environment = next - current - output
        assert_eq!(turns[0].input_content_tokens, 95);
        assert_eq!(turns[0].environment_tokens, 5);
        assert_eq!(turns[1].input_content_tokens, 120);
        assert_eq!(turns[1].environment_tokens, 30);
        assert_eq!(turns[2].input_content_tokens, 160);
        assert!(plans[0].turns.iter().all(|turn| turn.blocks.is_none()));
    }

    #[test]
    fn v2_block_plans_use_block_sums_and_scaled_timings() {
        let specs = parse_trajectory_plan_specs(std::io::Cursor::new(V2_BLOCK_MANIFEST), "fixture")
            .unwrap();
        let config = replay_config().with_time_scale(2.0).unwrap();
        let plans = build_replay_agent_plans(&config, specs).unwrap();

        let first = &plans[0];
        assert!(first.uses_blocks());
        assert_eq!(first.initial_prompt_tokens, 24);
        assert_eq!(first.turns[0].input_content_tokens, 24);
        assert_eq!(first.turns[0].target_prompt_tokens, 30);
        assert_eq!(first.turns[0].environment_tokens, 0);
        assert_eq!(first.turns[0].tool_call_latency, Duration::from_millis(100));
        assert_eq!(first.turns[0].stream, Some(true));
        assert_eq!(first.turns[0].max_tokens, Some(16));
        assert_eq!(first.turns[1].input_content_tokens, 38);
        assert_eq!(first.start_after, Duration::ZERO);
        assert_eq!(plans[1].start_after, Duration::from_millis(200));
        assert_eq!(scale_millis(250, 1.0), Duration::from_millis(250));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn block_text_depends_on_the_seed_alone() {
        let tokenizer = std::sync::Arc::new(byte_level_tokenizer());
        let first = SyntheticTextGenerator::new(tokenizer.clone(), 1).unwrap();
        let second = SyntheticTextGenerator::new(tokenizer.clone(), 2).unwrap();
        let block = BlockSpec {
            seed: "shared-system-prompt".to_string(),
            tokens: 40,
            role: BlockRole::System,
            live: false,
        };

        let from_first = first.generate_block(&block).await.unwrap();
        let from_second = second.generate_block(&block).await.unwrap();
        assert_eq!(from_first, from_second);
        assert_eq!(
            tokenizer.encode(from_first.as_ref(), false).unwrap().len(),
            40
        );

        let other = second
            .generate_block(&BlockSpec {
                seed: "other".to_string(),
                ..block.clone()
            })
            .await
            .unwrap();
        assert_ne!(other, from_first);
        assert!(first
            .generate_block(&BlockSpec {
                tokens: 0,
                ..block.clone()
            })
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tool_blocks_serialize_to_the_requested_token_count() {
        let tokenizer = std::sync::Arc::new(byte_level_tokenizer());
        let generator = SyntheticTextGenerator::new(tokenizer.clone(), 7).unwrap();
        let definition = BlockSpec {
            seed: "tool-a".to_string(),
            tokens: 200,
            role: BlockRole::ToolDefinition,
            live: false,
        };
        let description = generator.generate_block(&definition).await.unwrap();
        assert!(description
            .chars()
            .all(|c| c != '"' && c != '\\' && !c.is_control()));
        let serialized = serde_json::to_string(&synthetic_tool_definition(
            &synthetic_tool_name(&definition.seed),
            &description,
        ))
        .unwrap();
        assert_eq!(tokenizer.encode(serialized, false).unwrap().len(), 200);

        // Targets below the JSON scaffolding cannot be met; the closest form is used instead.
        let tiny = generator
            .generate_block(&BlockSpec {
                tokens: 3,
                ..definition.clone()
            })
            .await
            .unwrap();
        assert!(!tiny.is_empty());

        let call = BlockSpec {
            seed: "call-a".to_string(),
            tokens: 32,
            role: BlockRole::ToolCall,
            live: false,
        };
        let input = generator.generate_block(&call).await.unwrap();
        let arguments = synthetic_tool_call_arguments(&input);
        assert_eq!(
            tokenizer.encode(arguments.as_str(), false).unwrap().len(),
            32
        );
        assert_eq!(
            serde_json::from_str::<Value>(&arguments).unwrap()["input"],
            input.as_ref()
        );
    }

    #[test]
    fn block_request_assembles_tools_messages_and_tool_call_ids() {
        let blocks: Vec<BlockSpec> = serde_json::from_str(
            r#"[
                {"seed":"t","tokens":1,"role":"tool_definition"},
                {"seed":"s","tokens":1,"role":"system"},
                {"seed":"u","tokens":1,"role":"user"},
                {"seed":"a","tokens":1,"role":"assistant","live":true},
                {"seed":"c","tokens":1,"role":"tool_call"},
                {"seed":"r","tokens":1,"role":"tool"},
                {"seed":"c2","tokens":1,"role":"tool_call"},
                {"seed":"r2","tokens":1,"role":"tool"},
                {"seed":"a2","tokens":1,"role":"assistant"}
            ]"#,
        )
        .unwrap();
        let mut cache = BlockCache::new();
        for seed in ["t", "s", "u", "c", "r", "c2", "r2", "a2"] {
            cache.insert(
                seed.to_string(),
                BlockContent::Generated(std::sync::Arc::from(format!("text-{seed}"))),
            );
        }
        cache.insert(
            "a".to_string(),
            BlockContent::Live(json!({"role": "assistant", "content": "live reply"})),
        );

        let request = build_block_request(&blocks, &cache).unwrap();
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.tools[0]["function"]["description"], "text-t");
        assert_eq!(
            request.tools[0]["function"]["name"],
            synthetic_tool_name("t")
        );
        let roles: Vec<&str> = request
            .messages
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect();
        assert_eq!(
            roles,
            vec![
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant"
            ]
        );
        // The live reply carries the first synthetic tool call; its result references that id.
        assert_eq!(request.messages[2]["content"], "live reply");
        let call_id = request.messages[2]["tool_calls"][0]["id"].as_str().unwrap();
        assert_eq!(call_id, synthetic_block_tool_call_id("c"));
        assert_eq!(request.messages[3]["tool_call_id"], call_id);
        assert_eq!(request.messages[3]["content"], "text-r");
        // A tool call without a preceding assistant message gets its own envelope.
        assert!(request.messages[4]["content"].is_null());
        assert_eq!(
            request.messages[5]["tool_call_id"],
            synthetic_block_tool_call_id("c2")
        );
        assert_eq!(
            serde_json::from_str::<Value>(
                request.messages[4]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap()
            )
            .unwrap()["input"],
            "text-c2"
        );
        assert_eq!(request.messages[6]["content"], "text-a2");

        let missing = build_block_request(&blocks[..1], &BlockCache::new()).unwrap_err();
        assert!(missing.to_string().contains("no prepared content"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn live_blocks_substitute_the_previous_reply_or_fall_back() {
        let tokenizer = std::sync::Arc::new(word_level_tokenizer());
        let generator = SyntheticTextGenerator::new(tokenizer.clone(), 3).unwrap();
        let blocks: Vec<BlockSpec> = serde_json::from_str(
            r#"[
                {"seed":"u","tokens":4,"role":"user"},
                {"seed":"a1","tokens":4,"role":"assistant","live":true},
                {"seed":"a2","tokens":4,"role":"assistant","live":true}
            ]"#,
        )
        .unwrap();
        let mut cache = BlockCache::new();
        let mut last_reply = Some(json!({"role": "assistant", "content": "reply"}));

        let fallbacks = resolve_live_blocks(&blocks, &mut cache, &mut last_reply, &generator)
            .await
            .unwrap();
        assert_eq!(fallbacks, 1);
        assert!(last_reply.is_none());
        assert!(matches!(cache.get("a1"), Some(BlockContent::Live(_))));
        match cache.get("a2") {
            Some(BlockContent::Generated(text)) => {
                assert_eq!(tokenizer.encode(text.as_ref(), false).unwrap().len(), 4)
            }
            other => panic!("expected generated fallback, got {other:?}"),
        }

        // Seeds already in the cache are reused without consuming a new reply.
        last_reply = Some(json!({"role": "assistant", "content": "second reply"}));
        let fallbacks = resolve_live_blocks(&blocks, &mut cache, &mut last_reply, &generator)
            .await
            .unwrap();
        assert_eq!(fallbacks, 0);
        assert!(last_reply.is_some());
        // Live blocks never need generation; only the uncached user block is outstanding.
        let outstanding = |cache: &BlockCache| {
            missing_generated_blocks(&blocks, cache)
                .iter()
                .map(|block| block.seed.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(outstanding(&cache), vec!["u"]);
        assert_eq!(outstanding(&BlockCache::new()), vec!["u"]);
    }

    #[test]
    fn request_body_applies_stream_max_tokens_and_tools() {
        let config = replay_config();
        let messages = vec![json!({"role": "user", "content": "start"})];
        let tools = vec![synthetic_tool_definition("fn_a", "desc")];
        let body: Value = serde_json::from_slice(
            &serialize_request_body_with(
                &config,
                &messages,
                RequestOptions {
                    output_tokens: 8,
                    max_tokens: Some(32),
                    stream: true,
                    tools: Some(&tools),
                },
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(body["max_tokens"], 32);
        assert_eq!(body["min_tokens"], 8);
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["tools"][0]["function"]["name"], "fn_a");

        let sglang: Value = serde_json::from_slice(
            &serialize_request_body_with(
                &replay_config().with_sglang(true),
                &messages,
                RequestOptions {
                    output_tokens: 8,
                    max_tokens: Some(4),
                    stream: false,
                    tools: None,
                },
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(sglang["max_new_tokens"], 4);
        assert_eq!(sglang["min_new_tokens"], 4);
        assert!(sglang.get("stream").is_none());
        assert!(sglang.get("tools").is_none());
        assert!(sglang.get("max_tokens").is_none());
    }

    #[test]
    fn streamed_chunks_assemble_an_assistant_message_with_usage() {
        let mut lines = SseLineBuffer::default();
        let mut completion = StreamedCompletion::default();
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"think\"}}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n",
            ": keep-alive\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\ndata: {\"choices\":[],",
        );
        for payload in lines.push(stream.as_bytes()) {
            completion.absorb(&payload).unwrap();
        }
        for payload in
            lines.push(b"\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":2}}\n\ndata: [DONE]")
        {
            completion.absorb(&payload).unwrap();
        }
        if let Some(payload) = lines.finish() {
            completion.absorb(&payload).unwrap();
        }
        let (message, usage) = completion.into_parts().unwrap();
        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "Hello");
        assert_eq!(message["reasoning_content"], "think");
        assert_eq!(usage["prompt_tokens"], 12);

        let missing_usage = StreamedCompletion {
            chunks: 1,
            ..StreamedCompletion::default()
        };
        assert!(missing_usage.into_parts().is_err());
        assert!(StreamedCompletion::default().into_parts().is_err());
    }

    #[test]
    fn slot_pool_reuses_the_lowest_free_slot() {
        let mut pool = SlotPool::default();
        assert_eq!((pool.acquire(), pool.acquire(), pool.acquire()), (0, 1, 2));
        pool.release(1);
        pool.release(0);
        assert_eq!(pool.acquire(), 0);
        assert_eq!(pool.acquire(), 1);
        assert_eq!(pool.acquire(), 3);
        assert_eq!(pool.peak(), 4);
    }

    #[tokio::test]
    async fn admission_cap_delays_and_counts_late_admissions() {
        let cap = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let (_permit, late) = admit_under_cap(Some(&cap)).await.unwrap();
        assert!(!late);
        assert!(admit_under_cap(None).await.unwrap().0.is_none());

        let waiting = tokio::spawn({
            let cap = cap.clone();
            async move { admit_under_cap(Some(&cap)).await.unwrap().1 }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiting.is_finished());
        drop(_permit);
        assert!(waiting.await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn open_loop_replay_admits_trajectories_at_their_scaled_offsets() {
        let directory =
            std::env::temp_dir().join(format!("batchbench-open-loop-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let tokenizer_path = directory.join("tokenizer.json");
        word_level_tokenizer().save(&tokenizer_path, false).unwrap();
        let manifest_path = directory.join("plans.jsonl");
        std::fs::write(&manifest_path, V2_BLOCK_MANIFEST).unwrap();

        let config = replay_config()
            .with_tokenizer_model(tokenizer_path.display().to_string())
            .with_agent_plans_jsonl(&manifest_path)
            .with_admission(AdmissionMode::OpenLoop)
            .with_time_scale(2.0)
            .unwrap()
            .with_dry_run(true);
        let report = run_agent_benchmark(config).await.unwrap();
        std::fs::remove_dir_all(&directory).unwrap();

        assert_eq!(report.trajectory_schema_version, Some(2));
        assert_eq!(report.completed_agents, 2);
        assert_eq!(report.planned_tool_invocations, 3);
        assert_eq!(report.late_admissions, 0);
        assert_eq!(report.live_block_fallbacks, 0);
        let second = report
            .agent_lifecycles
            .iter()
            .find(|record| record.trajectory_id == "second")
            .unwrap();
        assert_eq!(second.scheduled_at, Duration::from_millis(200));
        assert!(second.admitted_at >= Duration::from_millis(200));
        assert!(report.max_admission_lag >= second.admitted_at - second.scheduled_at);
        assert!(report.total_duration >= Duration::from_millis(200));
        assert!(report.max_active_agents >= 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closed_loop_dry_run_accepts_v2_block_manifests() {
        let directory =
            std::env::temp_dir().join(format!("batchbench-closed-loop-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let tokenizer_path = directory.join("tokenizer.json");
        word_level_tokenizer().save(&tokenizer_path, false).unwrap();
        let manifest_path = directory.join("plans.jsonl");
        std::fs::write(&manifest_path, V2_BLOCK_MANIFEST).unwrap();

        let config = replay_config()
            .with_tokenizer_model(tokenizer_path.display().to_string())
            .with_agent_plans_jsonl(&manifest_path)
            .with_max_active_agents(1)
            .unwrap()
            .with_dry_run(true);
        let report = run_agent_benchmark(config).await.unwrap();
        std::fs::remove_dir_all(&directory).unwrap();

        assert_eq!(report.completed_agents, 2);
        assert_eq!(report.max_active_agents, 1);
        assert_eq!(report.late_admissions, 0);
        assert!(report
            .agent_lifecycles
            .iter()
            .all(|record| record.scheduled_at == Duration::ZERO));
    }
}
