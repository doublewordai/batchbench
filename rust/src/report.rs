use std::time::Duration;

#[derive(Debug, Clone)]
pub struct FailureRecord {
    pub user_id: usize,
    pub error: String,
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_duration: Duration,
    pub prompt_tokens_per_second: f64,
    pub completion_tokens_per_second: f64,
    pub requests_per_second: f64,
    pub latency_p50: Option<Duration>,
    pub latency_p90: Option<Duration>,
    pub latency_p99: Option<Duration>,
    pub failures: Vec<FailureRecord>,
}

impl BenchmarkReport {
    pub fn total_token_throughput(&self) -> f64 {
        self.prompt_tokens_per_second + self.completion_tokens_per_second
    }
}
