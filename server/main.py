"""BatchBench Results Server - receives and stores profiling results."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

app = FastAPI(title="BatchBench Results Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = Path(os.environ.get("BATCHBENCH_DATA_DIR", "./batchbench_data"))
RESULTS_CSV = DATA_DIR / "results.csv"
CONFIGS_DIR = DATA_DIR / "configs"


class ProfilingResult(BaseModel):
    # Identification
    project_name: str
    experiment_name: str
    
    # Model and dataset
    model: str
    dataset_path: str
    dataset_size: int
    
    # Benchmark configuration
    users: int
    requests_per_user: int
    output_tokens: Optional[int] = None
    output_vary: Optional[int] = None
    request_timeout_secs: int
    max_retries: int
    retry_delay_ms: int
    random_requests: bool
    
    # Results
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_duration_seconds: float
    requests_per_second: float
    prompt_tokens_per_second: float
    completion_tokens_per_second: float
    latency_p50_ms: Optional[float] = None
    latency_p90_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    
    # Timestamp
    timestamp: str
    
    # Full config for storage
    full_config: Dict[str, Any]

    class Config:
        extra = "allow"  # Ignore legacy/extra fields like top-level hardware


def ensure_dirs():
    """Ensure data directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic hash of the configuration."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def save_config(config: Dict[str, Any], config_hash: str) -> Path:
    """Save configuration to a JSON file."""
    config_path = CONFIGS_DIR / f"{config_hash}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def append_to_csv(result: ProfilingResult, config_hash: str):
    """Append a result row to the CSV file."""
    file_exists = RESULTS_CSV.exists()
    
    fieldnames = [
        "timestamp",
        "project_name",
        "experiment_name",
        "config_hash",
        "model",
        "dataset_path",
        "dataset_size",
        "users",
        "requests_per_user",
        "output_tokens",
        "output_vary",
        "request_timeout_secs",
        "max_retries",
        "retry_delay_ms",
        "random_requests",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_duration_seconds",
        "requests_per_second",
        "prompt_tokens_per_second",
        "completion_tokens_per_second",
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p99_ms",
    ]
    
    row = {
        "timestamp": result.timestamp,
        "project_name": result.project_name,
        "experiment_name": result.experiment_name,
        "config_hash": config_hash,
        "model": result.model,
        "dataset_path": result.dataset_path,
        "dataset_size": result.dataset_size,
        "users": result.users,
        "requests_per_user": result.requests_per_user,
        "output_tokens": result.output_tokens,
        "output_vary": result.output_vary,
        "request_timeout_secs": result.request_timeout_secs,
        "max_retries": result.max_retries,
        "retry_delay_ms": result.retry_delay_ms,
        "random_requests": result.random_requests,
        "total_requests": result.total_requests,
        "successful_requests": result.successful_requests,
        "failed_requests": result.failed_requests,
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
        "total_duration_seconds": result.total_duration_seconds,
        "requests_per_second": result.requests_per_second,
        "prompt_tokens_per_second": result.prompt_tokens_per_second,
        "completion_tokens_per_second": result.completion_tokens_per_second,
        "latency_p50_ms": result.latency_p50_ms,
        "latency_p90_ms": result.latency_p90_ms,
        "latency_p99_ms": result.latency_p99_ms,
    }
    
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@app.on_event("startup")
async def startup():
    ensure_dirs()


@app.post("/api/results")
async def submit_results(result: ProfilingResult):
    """Receive profiling results, save config and append to CSV."""
    try:
        # Compute hash of full config
        config_hash = compute_config_hash(result.full_config)
        
        # Save config file (idempotent - same config = same hash = same file)
        config_path = save_config(result.full_config, config_hash)
        
        # Append to CSV
        append_to_csv(result, config_hash)
        
        return {
            "status": "success",
            "config_hash": config_hash,
            "config_path": str(config_path),
            "csv_path": str(RESULTS_CSV),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/results")
async def get_results(
    project_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    limit: int = 100,
):
    """Retrieve results from CSV with optional filtering."""
    if not RESULTS_CSV.exists():
        return {"results": [], "total": 0}
    
    results = []
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if project_name and row.get("project_name") != project_name:
                continue
            if experiment_name and row.get("experiment_name") != experiment_name:
                continue
            results.append(row)
    
    total = len(results)
    results = results[-limit:]  # Return most recent
    
    return {"results": results, "total": total}


@app.get("/api/configs/{config_hash}")
async def get_config(config_hash: str):
    """Retrieve a configuration by its hash."""
    config_path = CONFIGS_DIR / f"{config_hash}.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    
    with open(config_path, "r") as f:
        return json.load(f)
