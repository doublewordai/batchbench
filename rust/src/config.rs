use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Url;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct RequestEntry {
    pub body: Value,
    pub line_idx: usize,
    pub input_tokens: usize,
}

#[derive(Clone, Debug)]
pub enum RunMode {
    /// Execute a fixed number of requests per user and then stop.
    Finite { requests_per_user: usize },
    /// Continuously execute requests for the provided wall-clock duration.
    LongRunning { duration: Duration },
}

#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub endpoint: Url,
    pub user_count: usize,
    pub mode: RunMode,
    pub request_body: RequestEntry,
    pub requests: Vec<RequestEntry>,
    pub request_timeout: Duration,
    pub max_retries: usize,
    pub retry_delay: Duration,
    pub headers: HeaderMap,
    pub verbose: bool,
    /// Optional log-normal distribution parameters (mu, sigma, max) for sampling output token counts
    pub output_lognorm: Option<(f64, f64, Option<usize>)>,
    /// Optional random seed for reproducible benchmarking
    pub seed: Option<u64>,
    /// Dry-run mode: skip HTTP and log selected request + token info
    pub dry_run: bool,
    /// Use SGLang sampling parameter names for output-token constraints
    pub sglang: bool,
}

impl BenchmarkConfig {
    pub fn try_new(
        endpoint: impl AsRef<str>,
        api_key: Option<String>,
        user_count: usize,
        mode: RunMode,
        request_body: RequestEntry,
    ) -> Result<Self> {
        if user_count == 0 {
            return Err(anyhow!("user_count must be greater than zero"));
        }

        match &mode {
            RunMode::Finite { requests_per_user } => {
                if *requests_per_user == 0 {
                    return Err(anyhow!(
                        "requests_per_user must be greater than zero for finite mode"
                    ));
                }
            }
            RunMode::LongRunning { duration } => {
                if duration.is_zero() {
                    return Err(anyhow!(
                        "duration must be greater than zero for long running mode"
                    ));
                }
            }
        }

        let endpoint = Url::parse(endpoint.as_ref())
            .with_context(|| format!("invalid endpoint URL: {}", endpoint.as_ref()))?;

        let mut headers = HeaderMap::new();
        if let Some(api_key) = api_key {
            if !api_key.is_empty() {
                let auth_value = format!("Bearer {}", api_key);
                let header_value = HeaderValue::from_str(&auth_value)
                    .context("failed to build Authorization header from api_key")?;
                headers.insert(AUTHORIZATION, header_value);
            }
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(Self {
            endpoint,
            user_count,
            mode,
            request_body,
            requests: Vec::new(),
            request_timeout: Duration::from_secs(6000),
            max_retries: 2,
            retry_delay: Duration::from_millis(250),
            headers,
            verbose: false,
            output_lognorm: None,
            seed: None,
            dry_run: false,
            sglang: false,
        })
    }

    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        if !request_timeout.is_zero() {
            self.request_timeout = request_timeout;
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

    pub fn with_output_lognorm(mut self, mu: f64, sigma: f64, max: Option<usize>) -> Self {
        self.output_lognorm = Some((mu, sigma, max));
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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

    pub fn add_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.insert(name, value);
        self
    }

    pub fn headers_mut(&mut self) -> &mut HeaderMap {
        &mut self.headers
    }

    pub fn with_request_list(mut self, requests: Vec<RequestEntry>) -> Result<Self> {
        if requests.is_empty() {
            return Err(anyhow!("request list cannot be empty"));
        }
        self.request_body = requests
            .first()
            .cloned()
            .unwrap_or_else(|| self.request_body.clone());
        self.requests = requests;
        Ok(self)
    }
}
