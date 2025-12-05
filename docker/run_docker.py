#!/usr/bin/env python3
"""
Docker entrypoint script for batchbench - runs offline or online benchmarks.
Configured via YAML file instead of environment variables.
"""
import csv
import json
import os
import sys
import subprocess
import time
import signal
import shlex
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import urllib.request
import urllib.error
import yaml


def log(message: str) -> None:
    """Log a message with timestamp to stderr."""
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    print(f'[{timestamp}] {message}', file=sys.stderr)


def die(message: str) -> None:
    """Log an error message and exit."""
    log(f'ERROR: {message}')
    sys.exit(1)


def print_command(args: List[str]) -> None:
    """Print a command with proper shell quoting."""
    quoted = ' '.join(shlex.quote(arg) for arg in args)
    print(f'    {quoted}', file=sys.stderr)


def load_config(config_path: str) -> Tuple[Dict[str, Any], str]:
    """Load configuration from YAML file and return parsed data plus raw text."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        config = yaml.safe_load(raw_text) or {}
        return config, raw_text
    except FileNotFoundError:
        die(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        die(f"Error parsing config file: {e}")


def add_args_from_dict(args: List[str], config: Dict[str, Any], mappings: List[tuple]) -> None:
    """Add command-line arguments from config dictionary based on mappings.
    
    Args:
        args: List to append arguments to
        config: Configuration dictionary
        mappings: List of (config_key, cli_flag) tuples
    """
    for config_key, cli_flag in mappings:
        value = config.get(config_key)
        if value is not None:
            args.extend([cli_flag, str(value)])


def add_extra_args(args: List[str], extra_args: Any) -> None:
    """Add extra arguments from config.
    
    Args:
        args: List to append arguments to
        extra_args: Either a list of args or a string to split
    """
    if not extra_args:
        return
    
    if isinstance(extra_args, list):
        args.extend(str(arg) for arg in extra_args)
    elif isinstance(extra_args, str):
        args.extend(shlex.split(extra_args))


def read_results_csv(results_path: str) -> Dict[str, Any]:
    """Load the single-row CSV summary emitted by batchbench.online."""
    if not os.path.isfile(results_path):
        die(f"Results CSV not found at {results_path}")

    with open(results_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        try:
            row = next(reader)
        except StopIteration:
            die(f"Results CSV {results_path} is empty")

    def require(field: str) -> str:
        value = row.get(field)
        if value is None or value == "":
            die(f"Results CSV missing required field '{field}'")
        return value

    def parse_int(field: str, optional: bool = False) -> Optional[int]:
        value = row.get(field)
        if value is None or value == "":
            if optional:
                return None
            die(f"Results CSV missing required integer field '{field}'")
        try:
            return int(value)
        except ValueError:
            die(f"Results CSV field '{field}' must be an integer (got '{value}')")

    def parse_float(field: str, optional: bool = False) -> Optional[float]:
        value = row.get(field)
        if value is None or value == "":
            if optional:
                return None
            die(f"Results CSV missing required float field '{field}'")
        try:
            return float(value)
        except ValueError:
            die(f"Results CSV field '{field}' must be a float (got '{value}')")

    def parse_bool(field: str) -> bool:
        value = row.get(field)
        if value is None or value == "":
            return False
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y"):
            return True
        if lowered in ("false", "0", "no", "n"):
            return False
        die(f"Results CSV field '{field}' must be boolean (got '{value}')")

    return {
        'timestamp': require('timestamp'),
        'model': require('model'),
        'dataset_path': require('dataset_path'),
        'dataset_size': parse_int('dataset_size'),
        'users': parse_int('users'),
        'requests_per_user': parse_int('requests_per_user'),
        'total_requests': parse_int('total_requests'),
        'successful_requests': parse_int('successful_requests'),
        'failed_requests': parse_int('failed_requests'),
        'total_prompt_tokens': parse_int('total_prompt_tokens'),
        'total_completion_tokens': parse_int('total_completion_tokens'),
        'total_duration_seconds': parse_float('total_duration_seconds'),
        'requests_per_second': parse_float('requests_per_second'),
        'prompt_tokens_per_second': parse_float('prompt_tokens_per_second'),
        'completion_tokens_per_second': parse_float('completion_tokens_per_second'),
        'latency_p50_ms': parse_float('latency_p50_ms', optional=True),
        'latency_p90_ms': parse_float('latency_p90_ms', optional=True),
        'latency_p99_ms': parse_float('latency_p99_ms', optional=True),
        'random_requests': parse_bool('random_requests'),
        'output_tokens': parse_int('output_tokens', optional=True),
        'output_vary': parse_int('output_vary', optional=True),
        'output_lognorm_mu': parse_float('output_lognorm_mu', optional=True),
        'output_lognorm_sigma': parse_float('output_lognorm_sigma', optional=True),
        'request_timeout_secs': parse_int('request_timeout_secs'),
        'max_retries': parse_int('max_retries'),
        'retry_delay_ms': parse_int('retry_delay_ms'),
        'host': require('host'),
        'endpoint': require('endpoint'),
    }


def post_results_to_server(server_url: str, payload: Dict[str, Any]) -> None:
    """Send benchmark results to the configured BatchBench server."""
    target = f"{server_url.rstrip('/')}/api/results"
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        target,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, 'status', response.getcode())
            body = response.read().decode('utf-8').strip()
            log(f"Posted results to {target} (status {status})")
            if body:
                try:
                    parsed = json.loads(body)
                    if isinstance(parsed, dict) and 'config_hash' in parsed:
                        log(f"Server returned config hash: {parsed['config_hash']}")
                except json.JSONDecodeError:
                    log("Server response was not valid JSON; skipping parse")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode('utf-8', errors='replace') if hasattr(exc, 'read') else ''
        log(f"Failed to post results (HTTP {exc.code}): {error_body}")
    except urllib.error.URLError as exc:
        log(f"Failed to reach results server: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        log(f"Unexpected error posting results: {exc}")


def run_offline(config: Dict[str, Any]) -> None:
    """Run offline benchmark.
    
    Args:
        config: Full configuration dictionary
    """
    log("Starting offline benchmark")
    offline_config = config.get('offline', {})
    
    args = ['python', '-m', 'batchbench.offline']

    # Add optional arguments based on configuration
    mappings = [
        ('model', '--model'),
        ('num_reqs', '--num_reqs'),
        ('icl', '--icl'),
        ('ocl', '--ocl'),
        ('throughput_dir', '--throughput_dir'),
        ('tensor_parallel_size', '--tensor_parallel_size'),
        ('pipeline_parallel_size', '--pipeline_parallel_size'),
        ('gpu_memory_utilization', '--gpu_memory_utilization'),
        ('max_num_batched_tokens', '--max_num_batched_tokens'),
    ]

    add_args_from_dict(args, offline_config, mappings)
    add_extra_args(args, offline_config.get('extra_args'))

    log("Invoking offline runner with:")
    print_command(args)
    subprocess.run(args, check=True)


def wait_for_local_http(url: str, pid: int, retries: int = 60, delay: int = 1) -> None:
    """Wait for a local HTTP server to become reachable.
    
    Args:
        url: URL to check
        pid: Process ID         retries: Number of retry attempts
to monitor
        delay: Delay in seconds between attempts
    """
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlopen(url, timeout=delay)
            log(f"Server became reachable after {attempt} attempt(s)")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

        # Check if process is still alive
        try:
            os.kill(pid, 0)
        except OSError:
            die("vLLM server process exited before readiness probe succeeded")

        time.sleep(delay)

    die(f"Timed out after {retries} attempts waiting for {url}")


class ServerManager:
    """Context manager for vLLM server lifecycle."""
    
    def __init__(self):
        self.server_pid: Optional[int] = None
        self.server_process: Optional[subprocess.Popen] = None

    def start_server(self, args: List[str]) -> int:
        """Start the vLLM server process."""
        log("Launching vLLM server:")
        print_command(args)
        
        self.server_process = subprocess.Popen(args)
        self.server_pid = self.server_process.pid
        return self.server_pid

    def cleanup(self) -> None:
        """Stop the vLLM server."""
        if self.server_pid is not None:
            log(f"Stopping vLLM server (PID {self.server_pid})")
            try:
                os.kill(self.server_pid, signal.SIGTERM)
            except OSError:
                pass
            
            if self.server_process:
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.kill(self.server_pid, signal.SIGKILL)
                    except OSError:
                        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def run_online(config: Dict[str, Any], raw_config_yaml: str) -> None:
    """Run online benchmark.
    
    Args:
        config: Full configuration dictionary
        raw_config_yaml: Original YAML text for downstream reporting
    """
    log("Starting online benchmark")
    
    online_config = config.get('online', {})
    generate_config = config.get('generate', {})

    # Prepare request dataset
    dataset_base = generate_config.get('dataset_path', '/tmp/batchbench_requests.jsonl')
    log(f"Preparing request dataset at: {dataset_base}")

    if os.path.isfile(dataset_base):
        log(f"Using existing dataset: {dataset_base}")
        dataset_path = dataset_base
    else:
        # Generate dataset
        generate_args = ['python', '-m', 'batchbench.generate', '--output', dataset_base]
        
        mappings = [
            ('count', '--count'),
            ('prefix_overlap', '--prefix-overlap'),
            ('approx_input_tokens', '--approx-input-tokens'),
            ('tokenizer_model', '--tokenizer-model'),
            ('token_tolerance', '--token-tolerance'),
            ('huggingface_token', '--huggingface-token'),
            ('model', '--model'),
            ('dist_mode', '--dist-mode'),
            ('dist_median', '--dist-median'),
            ('dist_sigma', '--dist-sigma'),
            ('dist_max', '--dist-max'),
        ]

        add_args_from_dict(generate_args, generate_config, mappings)
        add_extra_args(generate_args, generate_config.get('extra_args'))

        log("Generating request dataset:")
        print_command(generate_args)
        
        try:
            subprocess.run(generate_args, check=True)
            dataset_path = dataset_base
            log(f"Generated dataset: {dataset_path}")
        except subprocess.CalledProcessError:
            die("Request dataset generation failed")

    # Get online model
    online_model = online_config.get('model')
    if not online_model:
        die("online.model must be set for online mode")

    # Server configuration
    server_config = online_config.get('server', {})
    gpu_name = server_config.get('gpu_name')
    if gpu_name is not None:
        gpu_name = str(gpu_name)
    gpu_count = server_config.get('gpu_count')
    if gpu_count is not None:
        try:
            gpu_count = int(gpu_count)
        except (TypeError, ValueError):
            die("online.server.gpu_count must be an integer when provided")
    server_host = server_config.get('host', '0.0.0.0')
    server_port = server_config.get('port', 8000)
    
    client_config = online_config.get('client', {})
    base_url = client_config.get('host') or f'http://127.0.0.1:{server_port}'
    readiness_url = server_config.get('health_url') or f'{base_url}/v1/models'
    results_csv_path = str(client_config.get('results_csv') or '/tmp/batchbench_results.csv')
    log(f"Benchmark CSV output will be written to: {results_csv_path}")

    # Build server arguments
    server_args = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        '--host', server_host,
        '--port', str(server_port),
        '--model', online_model,
    ]

    server_mappings = [
        ('tensor_parallel_size', '--tensor-parallel-size'),
        ('pipeline_parallel_size', '--pipeline-parallel-size'),
        ('max_num_batched_tokens', '--max-num-batched-tokens'),
        ('gpu_memory_utilization', '--gpu-memory-utilization'),
    ]

    add_args_from_dict(server_args, server_config, server_mappings)
    add_extra_args(server_args, server_config.get('extra_args'))

    # Start server and run benchmark
    with ServerManager() as manager:
        # Start server
        server_pid = manager.start_server(server_args)
        
        # Wait for server to be ready
        wait_retries = server_config.get('wait_retries', 60)
        wait_delay = server_config.get('wait_delay_secs', 1)
        wait_for_local_http(readiness_url, server_pid, wait_retries, wait_delay)

        # Build online benchmark arguments
        online_args = [
            'python', '-m', 'batchbench.online',
            '--jsonl', dataset_path,
            '--model', online_model,
            '--host', base_url,
        ]

        client_mappings = [
            ('endpoint', '--endpoint'),
            ('users', '--users'),
            ('requests_per_user', '--requests-per-user'),
            ('api_key', '--api-key'),
            ('api_key_env', '--api-key-env'),
            ('request_timeout_secs', '--request-timeout-secs'),
            ('max_retries', '--max-retries'),
            ('retry_delay_ms', '--retry-delay-ms'),
            ('output_tokens', '--output-tokens'),
            ('output_vary', '--output-vary'),
            ('output_lognorm_mu', '--output-lognorm-mu'),
            ('output_lognorm_sigma', '--output-lognorm-sigma'),
        ]

        add_args_from_dict(online_args, client_config, client_mappings)
        online_args.extend(['--results-csv', results_csv_path])

        # Boolean flags
        if client_config.get('random_requests', False):
            online_args.append('--random-requests')
        if client_config.get('verbose', False):
            online_args.append('--verbose')

        add_extra_args(online_args, client_config.get('extra_args'))

        log("Running online benchmark:")
        print_command(online_args)
        subprocess.run(online_args, check=True)

    if os.path.isfile(results_csv_path):
        log(f"Benchmark summary saved to {results_csv_path}")
    else:
        log(f"WARNING: expected results CSV at {results_csv_path} but it was not created")

    server_url = client_config.get('server_url')
    if server_url:
        server_url = str(server_url)
    if not server_url:
        return

    project_name = client_config.get('project_name')
    experiment_name = client_config.get('experiment_name')
    if project_name:
        project_name = str(project_name)
    if experiment_name:
        experiment_name = str(experiment_name)
    if not project_name or not experiment_name:
        die('client.server_url requires both project_name and experiment_name')

    if not os.path.isfile(results_csv_path):
        die(f"Cannot post results because {results_csv_path} was not created")

    results = read_results_csv(results_csv_path)
    payload = {
        'project_name': project_name,
        'experiment_name': experiment_name,
        'model': results['model'],
        'dataset_path': results['dataset_path'],
        'dataset_size': results['dataset_size'],
        'users': results['users'],
        'requests_per_user': results['requests_per_user'],
        'output_tokens': results['output_tokens'],
        'output_vary': results['output_vary'],
        'output_lognorm_mu': results['output_lognorm_mu'],
        'output_lognorm_sigma': results['output_lognorm_sigma'],
        'request_timeout_secs': results['request_timeout_secs'],
        'max_retries': results['max_retries'],
        'retry_delay_ms': results['retry_delay_ms'],
        'random_requests': results['random_requests'],
        'total_requests': results['total_requests'],
        'successful_requests': results['successful_requests'],
        'failed_requests': results['failed_requests'],
        'total_prompt_tokens': results['total_prompt_tokens'],
        'total_completion_tokens': results['total_completion_tokens'],
        'total_duration_seconds': results['total_duration_seconds'],
        'requests_per_second': results['requests_per_second'],
        'prompt_tokens_per_second': results['prompt_tokens_per_second'],
        'completion_tokens_per_second': results['completion_tokens_per_second'],
        'latency_p50_ms': results['latency_p50_ms'],
        'latency_p90_ms': results['latency_p90_ms'],
        'latency_p99_ms': results['latency_p99_ms'],
        'timestamp': results['timestamp'],
        'host': results['host'],
        'endpoint': results['endpoint'],
        'hardware': {
            'gpu_name': gpu_name,
            'gpu_count': gpu_count,
        },
        'full_config': config,
        'config_yaml': raw_config_yaml,
    }

    post_results_to_server(server_url, payload)

def main() -> None:
    """Main entrypoint."""
    # Get config file path from environment variable or use default
    config_path = os.environ.get('CONFIG_FILE', '/etc/batchbench/config.yaml')
    
    if not os.path.isfile(config_path):
        die(f"Config file not found: {config_path}. Set CONFIG_FILE environment variable or mount config to default path.")
    
    log(f"Loading configuration from: {config_path}")
    config, raw_config_yaml = load_config(config_path)
    
    mode = config.get('mode', 'offline').lower()

    if mode == 'offline':
        run_offline(config)
    elif mode == 'online':
        run_online(config, raw_config_yaml)
    else:
        die(f"Unknown mode '{mode}'. Expected 'online' or 'offline'.")


if __name__ == '__main__':
    main()
