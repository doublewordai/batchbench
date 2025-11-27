#!/usr/bin/env python3
"""
Docker entrypoint script for batchbench - runs offline or online benchmarks.
Configured via YAML file instead of environment variables.
"""
import os
import sys
import subprocess
import time
import signal
import shlex
from datetime import datetime
from typing import List, Optional, Dict, Any
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
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
        pid: Process ID to monitor
        retries: Number of retry attempts
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


def run_online(config: Dict[str, Any]) -> None:
    """Run online benchmark.
    
    Args:
        config: Full configuration dictionary
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
        ]

        add_args_from_dict(generate_args, generate_config, mappings)
        add_extra_args(generate_args, generate_config.get('extra_args'))

        log("Generating request dataset:")
        print_command(generate_args)
        
        try:
            result = subprocess.run(
                generate_args,
                check=True,
                capture_output=True,
                text=True
            )
            output_lines = result.stdout.strip().split('\n')
            if not output_lines or not output_lines[-1]:
                die("Dataset generator did not return a path")
            dataset_path = output_lines[-1].strip()
            if not dataset_path:
                die("Dataset generator returned an empty path")
            log(f"Using dataset {dataset_path}")
        except subprocess.CalledProcessError:
            die("Request dataset generation failed")

    # Get online model
    online_model = online_config.get('model')
    if not online_model:
        die("online.model must be set for online mode")

    # Server configuration
    server_config = online_config.get('server', {})
    server_host = server_config.get('host', '0.0.0.0')
    server_port = server_config.get('port', 8000)
    
    client_config = online_config.get('client', {})
    base_url = client_config.get('host', f'http://127.0.0.1:{server_port}')
    readiness_url = server_config.get('health_url', f'{base_url}/v1/models')

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
        client_model = online_config.get('client_model', online_model)
        online_args = [
            'python', '-m', 'batchbench.online',
            '--jsonl', dataset_path,
            '--model', client_model,
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
        ]

        add_args_from_dict(online_args, client_config, client_mappings)

        # Boolean flags
        if client_config.get('random_requests', False):
            online_args.append('--random-requests')
        if client_config.get('verbose', False):
            online_args.append('--verbose')

        add_extra_args(online_args, client_config.get('extra_args'))

        log("Running online benchmark:")
        print_command(online_args)
        subprocess.run(online_args, check=True)


def main() -> None:
    """Main entrypoint."""
    # Get config file path from environment variable or use default
    config_path = os.environ.get('CONFIG_FILE', '/etc/batchbench/config.yaml')
    
    if not os.path.isfile(config_path):
        die(f"Config file not found: {config_path}. Set CONFIG_FILE environment variable or mount config to default path.")
    
    log(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    mode = config.get('mode', 'offline').lower()

    if mode == 'offline':
        run_offline(config)
    elif mode == 'online':
        run_online(config)
    else:
        die(f"Unknown mode '{mode}'. Expected 'online' or 'offline'.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        die(f"Command failed with exit code {e.returncode}")
    except Exception as e:
        die(f"Unexpected error: {e}")
