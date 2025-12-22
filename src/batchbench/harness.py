#!/usr/bin/env python3
"""
BatchBench Harness - Automated benchmarking orchestration via Prime Intellect.

This script:
1. Provisions a GPU instance via Prime Intellect API
2. Connects via SSH
3. Pulls the batchbench Docker image and starts container
4. Sets up environment (venv, dependencies)
5. Starts vLLM server and waits for readiness
6. Generates synthetic data
7. Runs online benchmark
"""

import argparse
import os
import sys
import time
from pathlib import Path

import paramiko
import requests
import yaml


# Container name used for docker exec commands
CONTAINER_NAME = "batchbench-run"


# Prime Intellect API base URL
PI_API_BASE = "https://api.primeintellect.ai/api/v1"


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PrimeIntellectClient:
    """Client for Prime Intellect API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an API request."""
        url = f"{PI_API_BASE}{endpoint}"
        resp = requests.request(method, url, headers=self.headers, **kwargs)
        if not resp.ok:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
        return resp.json() if resp.text else {}

    def check_availability(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        region: str | None = None,
        security: str | None = None,
    ) -> list[dict]:
        """Check GPU availability."""
        params = {
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
        }
        if region:
            params["regions"] = region
        if security:
            params["security"] = security

        print(f"  API params: {params}")
        result = self._request("GET", "/availability/gpus", params=params)
        print(f"  API response: {result.get('totalCount', 0)} items found")
        return result.get("items", [])

    def create_pod(
        self,
        gpu_config: dict,
        name: str,
        image: str,
        team_id: str | None = None,
        disk_size_gb: int = 100,
    ) -> dict:
        """Create a new pod instance.

        gpu_config should be an item from check_availability() response.
        """
        payload = {
            "pod": {
                "cloudId": gpu_config.get("cloudId"),
                "gpuType": gpu_config.get("gpuType"),
                "socket": gpu_config.get("socket"),
                "gpuCount": gpu_config.get("gpuCount", 1),
                "name": name,
                "diskSize": disk_size_gb,
                "image": image,
                "dataCenterId": gpu_config.get("dataCenter"),
            },
            "provider": {
                "type": gpu_config.get("provider"),
            },
        }
        if team_id:
            payload["team"] = {"teamId": team_id}
        return self._request("POST", "/pods/", json=payload)

    def get_pod(self, pod_id: str) -> dict:
        """Get pod details."""
        return self._request("GET", f"/pods/{pod_id}")

    def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod."""
        self._request("DELETE", f"/pods/{pod_id}")

    def wait_for_pod_ready(
        self, pod_id: str, timeout: int = 600, poll_interval: int = 10
    ) -> dict:
        """Wait for pod to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            pod = self.get_pod(pod_id)
            status = pod.get("status", "").lower()
            print(f"  Pod status: {status}")

            if status == "active":
                return pod
            elif status in ("stopped", "error", "terminated"):
                raise RuntimeError(f"Pod entered {status} state")

            time.sleep(poll_interval)

        raise TimeoutError(f"Pod not ready after {timeout}s")


class SSHSession:
    """Persistent SSH session for running multiple commands."""

    def __init__(self, host: str, port: int, username: str, key_path: Path):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Try different key formats
        try:
            key = paramiko.Ed25519Key.from_private_key_file(str(key_path))
        except paramiko.ssh_exception.SSHException:
            try:
                key = paramiko.RSAKey.from_private_key_file(str(key_path))
            except paramiko.ssh_exception.SSHException:
                raise RuntimeError(f"Could not load SSH key from {key_path}")

        self.client.connect(hostname=host, port=port, username=username, pkey=key)

    def run(self, command: str, timeout: int = 60, stream: bool = False) -> tuple[str, str, int]:
        """Run a command and return (stdout, stderr, exit_code).

        If stream=True, prints output in real-time as it arrives.
        """
        _, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        channel = stdout.channel

        if stream:
            # Stream both stdout and stderr in real-time
            stdout_lines = []
            stderr_lines = []
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    chunk = channel.recv(1024).decode()
                    print(chunk, end="", flush=True)
                    stdout_lines.append(chunk)
                if channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(1024).decode()
                    print(chunk, end="", flush=True)
                    stderr_lines.append(chunk)
                time.sleep(0.1)
            # Get any remaining output
            while channel.recv_ready():
                chunk = channel.recv(1024).decode()
                print(chunk, end="", flush=True)
                stdout_lines.append(chunk)
            while channel.recv_stderr_ready():
                chunk = channel.recv_stderr(1024).decode()
                print(chunk, end="", flush=True)
                stderr_lines.append(chunk)
            exit_code = channel.recv_exit_status()
            return "".join(stdout_lines), "".join(stderr_lines), exit_code
        else:
            exit_code = channel.recv_exit_status()
            return stdout.read().decode(), stderr.read().decode(), exit_code

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def run_synthetic_generation(
    docker_exec: callable,
    config: dict,
    verbose: bool = False,
) -> str:
    """Generate synthetic data inside the container. Returns the output file path."""
    gen_cfg = config.get("generate", {})

    output = gen_cfg.get("output", "data/synthetic.jsonl")
    count = gen_cfg.get("count", 1000)
    seed = gen_cfg.get("seed", 42)
    prefix_overlap = gen_cfg.get("prefix_overlap", 0.0)
    dist_mode = gen_cfg.get("dist_mode", "lognormal")

    tokenizer = gen_cfg.get("tokenizer")
    if not tokenizer:
        print("ERROR: No tokenizer specified in generate config")
        sys.exit(1)

    # Build command arguments
    cmd_args = [
        f"--count {count}",
        f"--output {output}",
        f"--seed {seed}",
        f"--prefix-overlap {prefix_overlap}",
        f"--tokenizer {tokenizer}",
        f"--dist-mode {dist_mode}",
    ]

    if dist_mode == "lognormal":
        cmd_args.append(f"--dist-median {gen_cfg.get('dist_median', 1000)}")
        cmd_args.append(f"--dist-sigma {gen_cfg.get('dist_sigma', 0.5)}")
        cmd_args.append(f"--dist-max {gen_cfg.get('dist_max', 128000)}")
    else:
        cmd_args.append(f"--approx-input-tokens {gen_cfg.get('approx_input_tokens', 512)}")
        if gen_cfg.get("token_tolerance"):
            cmd_args.append(f"--token-tolerance {gen_cfg['token_tolerance']}")

    generate_cmd = "python -m batchbench.generate " + " ".join(cmd_args)

    full_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"cd batchbench && {generate_cmd}"
    )

    print(f"\nGenerating synthetic data ({count} samples)...")
    if verbose:
        print(f"Command: {generate_cmd}")

    stdout, stderr, exit_code = docker_exec(full_cmd, timeout=600, stream=True)
    if exit_code != 0:
        print(f"Synthetic generation failed: {stderr}")
        sys.exit(1)

    print("Synthetic data generation complete")
    return output


def run_benchmark(
    docker_exec: callable,
    config: dict,
    synthetic_data_path: str,
    verbose: bool = False,
) -> None:
    """Run the online benchmark inside the container."""
    bench_cfg = config.get("benchmark", {})
    vllm_cfg = config.get("vllm", {})
    vllm_args = vllm_cfg.get("args") or {}

    model = vllm_cfg.get("model")
    if not model:
        print("ERROR: No model specified in vllm config")
        sys.exit(1)

    port = vllm_args.get("port", 8000)
    host = f"http://localhost:{port}"

    # Build command arguments
    cmd_args = [
        f"--jsonl {synthetic_data_path}",
        f"--model {model}",
        f"--host {host}",
        f"--users {bench_cfg.get('users', 32)}",
        f"--requests-per-user {bench_cfg.get('requests_per_user', 10)}",
    ]

    if bench_cfg.get("request_timeout_secs"):
        cmd_args.append(f"--request-timeout-secs {bench_cfg['request_timeout_secs']}")

    if bench_cfg.get("random_requests", True):
        cmd_args.append("--random-requests")

    if bench_cfg.get("seed"):
        cmd_args.append(f"--seed {bench_cfg['seed']}")

    if bench_cfg.get("max_retries"):
        cmd_args.append(f"--max-retries {bench_cfg['max_retries']}")

    if bench_cfg.get("retry_delay_ms"):
        cmd_args.append(f"--retry-delay-ms {bench_cfg['retry_delay_ms']}")

    # Output token distribution - lognormal or fixed
    if bench_cfg.get("output_lognorm_mu"):
        cmd_args.append(f"--output-lognorm-mu {bench_cfg['output_lognorm_mu']}")
        cmd_args.append(f"--output-lognorm-sigma {bench_cfg.get('output_lognorm_sigma', 0.5)}")
        cmd_args.append(f"--output-lognorm-max {bench_cfg.get('output_lognorm_max', 2000)}")
    elif bench_cfg.get("output_tokens"):
        cmd_args.append(f"--output-tokens {bench_cfg['output_tokens']}")
        if bench_cfg.get("output_vary"):
            cmd_args.append(f"--output-vary {bench_cfg['output_vary']}")

    if bench_cfg.get("results_csv"):
        cmd_args.append(f"--results-csv {bench_cfg['results_csv']}")

    benchmark_cmd = "python -m batchbench.online " + " ".join(cmd_args)

    full_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"cd batchbench && {benchmark_cmd}"
    )

    print(f"\nRunning online benchmark...")
    print(f"  Users: {bench_cfg.get('users', 32)}")
    print(f"  Requests per user: {bench_cfg.get('requests_per_user', 10)}")
    if verbose:
        print(f"Command: {benchmark_cmd}")

    stdout, stderr, exit_code = docker_exec(full_cmd, timeout=3600, stream=True)
    if exit_code != 0:
        print(f"Benchmark failed: {stderr}")
        sys.exit(1)

    print("\nBenchmark complete!")


def start_vllm_server(
    docker_exec: callable,
    config: dict,
    verbose: bool = False,
) -> None:
    """Start vLLM server inside the container and wait for it to be ready."""
    vllm_cfg = config.get("vllm", {})
    model = vllm_cfg.get("model")
    if not model:
        print("ERROR: No model specified in vllm config")
        sys.exit(1)

    vllm_args = vllm_cfg.get("args") or {}
    vllm_env = vllm_cfg.get("env") or {}
    startup_timeout = vllm_cfg.get("startup_timeout", 600)
    port = vllm_args.get("port", 8000)

    # Build environment variable exports
    env_exports = " ".join(f"{k}={v}" for k, v in vllm_env.items())

    # Build vllm serve command arguments
    vllm_cmd_args = [f"--model {model}"]
    for arg, value in vllm_args.items():
        if isinstance(value, bool):
            if value:
                vllm_cmd_args.append(f"--{arg}")
        else:
            vllm_cmd_args.append(f"--{arg} {value}")

    vllm_serve_cmd = "vllm serve " + " ".join(vllm_cmd_args)

    # Full command: activate venv, set env vars, run vllm in background
    start_server_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"{env_exports} nohup {vllm_serve_cmd} > /tmp/vllm.log 2>&1 &"
    )

    print(f"\nStarting vLLM server with model: {model}")
    if verbose:
        print(f"Command: {vllm_serve_cmd}")
    stdout, stderr, exit_code = docker_exec(start_server_cmd)
    if exit_code != 0:
        print(f"Failed to start vLLM server: {stderr}")
        sys.exit(1)
    print("vLLM server process started")

    # Poll for server readiness while streaming logs
    print(f"\nWaiting for vLLM server to be ready (timeout: {startup_timeout}s)...")
    print("-" * 60)
    health_url = f"http://localhost:{port}/health"
    poll_interval = 5
    start_time = time.time()
    last_log_line = 0

    while time.time() - start_time < startup_timeout:
        # Fetch and print new log lines
        stdout, _, _ = docker_exec(
            f"tail -n +{last_log_line + 1} /tmp/vllm.log 2>/dev/null",
            timeout=10,
        )
        if stdout.strip():
            print(stdout, end="", flush=True)
            last_log_line += stdout.count("\n")

        # Check health endpoint
        stdout, stderr, exit_code = docker_exec(
            f"curl -s -o /dev/null -w '%{{http_code}}' {health_url}",
            timeout=10,
        )
        if exit_code == 0 and stdout.strip() == "200":
            print("-" * 60)
            print("vLLM server is ready!")
            return

        time.sleep(poll_interval)

    print("-" * 60)
    print(f"ERROR: vLLM server not ready after {startup_timeout}s")
    sys.exit(1)


def run_harness(config_path: str, verbose: bool = False) -> None:
    """Main harness execution."""
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    instance_cfg = config.get("instance", {})
    team_id = os.environ.get("PRIME_TEAM_ID")
    gpu_type = instance_cfg.get("gpu_type", "H100_80GB")
    gpu_count = instance_cfg.get("gpu_count", 1)
    region = instance_cfg.get("region")
    security = instance_cfg.get("security")
    image = instance_cfg.get("image", "ubuntu_22_cuda_12")
    disk_size = instance_cfg.get("disk_size_gb", 100)
    provision_timeout = instance_cfg.get("provision_timeout", 600)
    name_prefix = instance_cfg.get("name_prefix", "batchbench")
    docker_image = instance_cfg.get("docker_image", "tytn/batchbench:cu126")

    # Initialize client
    print("Authenticating with Prime Intellect...")
    client = PrimeIntellectClient(os.environ.get("PRIME_API_KEY"))

    # Check availability
    print(f"Checking availability for {gpu_count}x {gpu_type}...")
    available = client.check_availability(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        region=region,
        security=security,
    )
    if not available:
        print(f"ERROR: No {gpu_type} GPUs available")
        print("Check availability at: https://app.primeintellect.ai/dashboard/create-cluster")
        sys.exit(1)


    # Select cheapest non-runpod GPU
    non_runpod = [opt for opt in available if opt.get("provider") != "runpod"]
    if not non_runpod:
        print("ERROR: No non-runpod GPUs available")
        sys.exit(1)
    selected = min(non_runpod, key=lambda x: float(x.get("prices", {}).get("onDemand", 999)))
    provider = selected.get("provider", "unknown")
    price = selected.get("prices", {}).get("onDemand", "N/A")
    print(f"Found available GPU:")
    print(f"  Provider: {provider}")
    print(f"  Cloud ID: {selected.get('cloudId')}")
    print(f"  GPU Type: {selected.get('gpuType')}")
    print(f"  Socket: {selected.get('socket')}")
    print(f"  Price: ${price}/hr")

    # Create pod
    instance_name = f"{name_prefix}-{int(time.time())}"
    print(f"\nCreating pod '{instance_name}'...")
    pod = client.create_pod(
        gpu_config=selected,
        name=instance_name,
        image=image,
        team_id=team_id,
        disk_size_gb=disk_size,
    )

    pod_id = pod.get("id")
    print(f"Pod created with ID: {pod_id}")

    # Wait for pod to be ready
    print(f"\nWaiting for pod to be ready (timeout: {provision_timeout}s)...")
    try:
        pod = client.wait_for_pod_ready(pod_id, timeout=provision_timeout)
    except (TimeoutError, RuntimeError) as e:
        print(f"ERROR: {e}")
        print("Cleaning up...")
        try:
            client.terminate_pod(pod_id)
        except Exception:
            pass
        sys.exit(1)

    print(f"\nPod is ready!")

    # Extract SSH connection info from sshConnection (format: "user@host -p port")
    ssh_conn = pod.get("sshConnection", "")
    user_host, _, port_str = ssh_conn.partition(" -p ")
    ssh_user, ssh_host = user_host.split("@", 1)
    ssh_port = int(port_str) if port_str else 22
    ssh_key_path = Path(os.environ.get("PRIME_SSH_KEY_PATH"))

    # Wait a moment for SSH to be fully ready
    print("\nWaiting for SSH to be ready...")
    time.sleep(10)

    # Connect and run commands
    print("\nConnecting via SSH...")
    try:
        with SSHSession(ssh_host, ssh_port, ssh_user, ssh_key_path) as ssh:
            # Determine docker command (with or without sudo)
            _, _, rc = ssh.run("docker info > /dev/null 2>&1")
            docker_cmd = "docker" if rc == 0 else "sudo docker"

            # Pull Docker image
            print(f"\nPulling Docker image {docker_image}...")
            stdout, stderr, exit_code = ssh.run(
                f"{docker_cmd} pull {docker_image}",
                timeout=600,
                stream=verbose,
            )
            if exit_code != 0:
                print(f"Docker pull failed: {stderr}")
                sys.exit(1)
            print("Docker image pulled successfully")

            # Start container in detached mode
            print(f"\nStarting container '{CONTAINER_NAME}'...")
            docker_run_cmd = (
                f"{docker_cmd} run -d "
                f"--name {CONTAINER_NAME} "
                f"--gpus all "
                f"{docker_image} "
                f"sleep infinity"
            )
            stdout, stderr, exit_code = ssh.run(docker_run_cmd, stream=verbose)
            if exit_code != 0:
                print(f"Failed to start container: {stderr}")
                sys.exit(1)
            print("Container started successfully")

            # Helper to run commands inside the container
            def docker_exec(cmd: str, timeout: int = 300, stream: bool = False) -> tuple[str, str, int]:
                """Execute a command inside the running container."""
                return ssh.run(
                    f'{docker_cmd} exec {CONTAINER_NAME} bash -c "{cmd}"',
                    timeout=timeout,
                    stream=stream,
                )

            # Clone batchbench repo
            print("\nCloning batchbench repository...")
            repo_url = "https://github.com/doublewordai/batchbench.git"
            stdout, stderr, exit_code = docker_exec(f"git clone {repo_url}", stream=verbose)
            if exit_code != 0:
                print(f"Git clone failed: {stderr}")
                sys.exit(1)
            print("Repository cloned successfully")

            # Install dependencies with uv
            print("\nInstalling dependencies...")
            install_cmd = (
                "source /opt/batchbench/.venv/bin/activate && "
                "cd batchbench && "
                "uv pip install -e ."
            )
            stdout, stderr, exit_code = docker_exec(install_cmd, timeout=300, stream=verbose)
            if exit_code != 0:
                print(f"Dependency installation failed: {stderr}")
                print(f"stdout: {stdout}")
                sys.exit(1)
            print("Dependencies installed successfully")

            # Start vLLM server
            start_vllm_server(docker_exec, config, verbose)

            # Generate synthetic data
            synthetic_data_path = run_synthetic_generation(docker_exec, config, verbose)

            # Run benchmark
            run_benchmark(docker_exec, config, synthetic_data_path, verbose)

    except Exception as e:
        print(f"SSH failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("Instance Summary")
    print("=" * 60)
    print(f"Pod ID: {pod_id}")
    print(f"Name: {instance_name}")
    print(f"SSH: ssh {ssh_user}@{ssh_host} -p {ssh_port}")
    print(f"\nTo terminate: prime pods terminate {pod_id}")
    print("Or via API: DELETE /api/v1/pods/{pod_id}")


def main():
    parser = argparse.ArgumentParser(description="BatchBench Harness - Automated GPU instance provisioning and benchmarking")
    parser.add_argument("config", nargs="?", default="configs/harness.yaml", help="Path to config file (default: configs/harness.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output (e.g., Docker build logs)")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    run_harness(args.config, verbose=args.verbose)


if __name__ == "__main__":
    main()
