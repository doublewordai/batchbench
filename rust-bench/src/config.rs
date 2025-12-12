use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Url;
use serde_json::Value;

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
    pub request_body: Value,
    pub per_user_bodies: Option<Vec<Value>>,
    pub random_request_pool: Option<Vec<Value>>,
    pub request_timeout: Duration,
    pub max_retries: usize,
    pub retry_delay: Duration,
    pub headers: HeaderMap,
    pub verbose: bool,
    /// Optional log-normal distribution parameters (mu, sigma, max) for sampling output token counts
    pub output_lognorm: Option<(f64, f64, Option<usize>)>,
    /// Optional random seed for reproducible benchmarking
    pub seed: Option<u64>,
}

impl BenchmarkConfig {
    pub fn try_new(
        endpoint: impl AsRef<str>,
        api_key: Option<String>,
        user_count: usize,
        mode: RunMode,
        request_body: Value,
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
            per_user_bodies: None,
            random_request_pool: None,
            request_timeout: Duration::from_secs(6000),
            max_retries: 2,
            retry_delay: Duration::from_millis(250),
            headers,
            verbose: false,
            output_lognorm: None,
            seed: None,
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

    pub fn add_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.insert(name, value);
        self
    }

    pub fn headers_mut(&mut self) -> &mut HeaderMap {
        &mut self.headers
    }

    pub fn with_per_user_bodies(mut self, bodies: Vec<Value>) -> Result<Self> {
        if bodies.len() < self.user_count {
            return Err(anyhow!(
                "per-user request bodies length ({}) is less than user_count ({})",
                bodies.len(),
                self.user_count
            ));
        }

        self.request_body = bodies
            .first()
            .cloned()
            .unwrap_or_else(|| self.request_body.clone());
        self.per_user_bodies = Some(bodies);
        Ok(self)
    }

    pub fn with_random_request_pool(mut self, bodies: Vec<Value>) -> Result<Self> {
        if bodies.is_empty() {
            return Err(anyhow!("random request pool cannot be empty"));
        }

        self.request_body = bodies
            .first()
            .cloned()
            .unwrap_or_else(|| self.request_body.clone());
        self.random_request_pool = Some(bodies);
        Ok(self)
    }

    pub fn request_body_for(&self, user_id: usize) -> Result<&Value> {
        if let Some(bodies) = &self.per_user_bodies {
            bodies
                .get(user_id)
                .ok_or_else(|| anyhow!("no request body configured for user {}", user_id))
        } else {
            Ok(&self.request_body)
        }
    }

    pub fn random_request_body(&self, user_id: usize, request_id: usize) -> Result<&Value> {
        if let Some(pool) = &self.random_request_pool {
            use rand::{Rng, SeedableRng};
            let idx = if let Some(seed) = self.seed {
                // Use seeded RNG for reproducibility
                // Mix in user_id and request_id to ensure different requests get different indices
                let mixed_seed = seed
                    .wrapping_add(user_id as u64)
                    .wrapping_add((request_id as u64).wrapping_mul(65537)); // Use prime multiplier for better distribution
                let mut rng = rand::rngs::StdRng::seed_from_u64(mixed_seed);
                rng.gen_range(0..pool.len())
            } else {
                let mut rng = rand::thread_rng();
                rng.gen_range(0..pool.len())
            };
            Ok(&pool[idx])
        } else {
            Err(anyhow!("random request pool is not configured"))
        }
    }
}
