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

    def check_availability(self, **params) -> list[dict]:
        print(f"  API params: {params}")
        result = self._request("GET", "/availability/gpus", params=params)
        print(f"  API response: {result.get('totalCount', 0)} items found")
        return result.get("items", [])

    def create_pod(self, gpu_config: dict,name: str, team_id: str, **pod_params) -> dict:
        payload = {
            "pod": {
                "cloudId": gpu_config["cloudId"],
                "gpuType": gpu_config["gpuType"],
                "socket": gpu_config["socket"],
                "gpuCount": gpu_config["gpuCount"],
                "dataCenterId": gpu_config["dataCenter"],
                "name": name,
                **pod_params,
            },
            "provider": {
                "type": gpu_config["provider"],
            },
            "team": {"teamId": team_id},
        }
        return self._request("POST", "/pods/", json=payload)

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}")

    def terminate_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")

    def wait_for_pod_ready(self, pod_id: str, timeout: int = 600, poll_interval: int = 10) -> dict:
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


class ContainerSession:
    """Represents a running Docker container that we can exec commands into."""

    def __init__(self, ssh: SSHSession, docker_cmd: str, container_name: str):
        self.ssh = ssh
        self.docker_cmd = docker_cmd
        self.container_name = container_name

    def exec(self, cmd: str, timeout: int = 300, stream: bool = False) -> tuple[str, str, int]:
        """Execute a command inside the container."""
        return self.ssh.run(
            f'{self.docker_cmd} exec {self.container_name} bash -c "{cmd}"',
            timeout=timeout,
            stream=stream,
        )


def build_cli_command(base: str, args: dict) -> str:
    cmd_args = []
    for arg, value in args.items():
        if isinstance(value, bool) and value:
            cmd_args.append(f"--{arg}")
        elif not isinstance(value, bool):
            cmd_args.append(f"--{arg} {value}")
    return base + " " + " ".join(cmd_args)


def run_synthetic_generation(container: ContainerSession, config: dict, verbose: bool = False) -> str:
    """Generate synthetic data inside the container. Returns the output file path."""
    gen_cfg = config["generate"]
    output = gen_cfg.get("output", "data/output.jsonl")

    generate_cmd = build_cli_command("python -m batchbench.generate", gen_cfg)
    full_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"cd batchbench && {generate_cmd}"
    )

    print(f"\nGenerating synthetic data...")
    if verbose:
        print(f"Command: {generate_cmd}")

    stdout, stderr, exit_code = container.exec(full_cmd, timeout=600, stream=True)
    if exit_code != 0:
        print(f"Synthetic generation failed: {stderr}")
        sys.exit(1)

    print("Synthetic data generation complete")
    return output


def run_benchmark(container: ContainerSession, config: dict, synthetic_data_path: str, verbose: bool = False) -> None:
    """Run the online benchmark inside the container."""
    bench_cfg = config["benchmark"]
    vllm_cfg = config["vllm"]
    vllm_args = vllm_cfg.get("args") or {}

    model = vllm_cfg["model"]
    port = vllm_args.get("port", 8000)
    host = f"http://localhost:{port}"

    # Merge benchmark config with derived values
    args = {"jsonl": synthetic_data_path, "model": model, "host": host, **bench_cfg}
    benchmark_cmd = build_cli_command("python -m batchbench.online", args)

    full_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"cd batchbench && {benchmark_cmd}"
    )

    print(f"\nRunning online benchmark...")
    print(f"  Users: {bench_cfg.get('users', 32)}")
    print(f"  Requests-per-user: {bench_cfg.get('requests-per-user', 10)}")
    if verbose:
        print(f"Command: {benchmark_cmd}")

    stdout, stderr, exit_code = container.exec(full_cmd, timeout=3600, stream=True)
    if exit_code != 0:
        print(f"Benchmark failed: {stderr}")
        sys.exit(1)

    print("\nBenchmark complete!")


def wait_for_vllm_ready(container: ContainerSession, port: int, timeout: int) -> None:
    print(f"\nWaiting for vLLM server to be ready (timeout: {timeout}s)...")
    print("-" * 60)

    health_url = f"http://localhost:{port}/health"
    poll_interval = 5
    start_time = time.time()
    last_log_line = 0

    while time.time() - start_time < timeout:
        # Fetch and print new log lines
        stdout, _, _ = container.exec(
            f"tail -n +{last_log_line + 1} /tmp/vllm.log 2>/dev/null",
            timeout=10,
        )
        if stdout.strip():
            print(stdout, end="", flush=True)
            last_log_line += stdout.count("\n")

        # Check health endpoint
        stdout, _, exit_code = container.exec(
            f"curl -s -o /dev/null -w '%{{http_code}}' {health_url}",
            timeout=10,
        )
        if exit_code == 0 and stdout.strip() == "200":
            print("-" * 60)
            print("vLLM server is ready!")
            return

        time.sleep(poll_interval)

    print("-" * 60)
    print(f"ERROR: vLLM server not ready after {timeout}s")
    sys.exit(1)


def start_vllm_server(container: ContainerSession, config: dict, verbose: bool = False) -> None:
    """Start vLLM server inside the container and wait for it to be ready."""
    vllm_cfg = config["vllm"]
    model = vllm_cfg["model"]
    args = vllm_cfg.get("args") or {}
    env = vllm_cfg.get("env") or {}
    startup_timeout = vllm_cfg.get("startup_timeout", 600)
    port = args.get("port", 8000)

    vllm_cmd = build_cli_command(f"vllm serve {model}", args)
    env_exports = " ".join(f"{k}={v}" for k, v in env.items())
    full_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"{env_exports} nohup {vllm_cmd} > /tmp/vllm.log 2>&1 &"
    )

    print(f"\nStarting vLLM server with model: {model}")
    if verbose:
        print(f"Command: {vllm_cmd}")

    stdout, stderr, exit_code = container.exec(full_cmd)
    if exit_code != 0:
        print(f"Failed to start vLLM server: {stderr}")
        sys.exit(1)
    print("vLLM server process started")

    wait_for_vllm_ready(container, port, startup_timeout)


def provision_instance(config: dict) -> dict:
    instance_cfg = config["instance"]
    availability_params = instance_cfg["availability"]
    create_params = instance_cfg["create"]
    name_prefix = instance_cfg.get("name_prefix", "batchbench")
    provision_timeout = instance_cfg.get("provision_timeout", 600)
    team_id = os.environ["PRIME_TEAM_ID"]

    client = PrimeIntellectClient(os.environ["PRIME_API_KEY"])

    gpu_type = availability_params["gpu_type"]
    gpu_count = availability_params["gpu_count"]
    print(f"Checking availability for {gpu_count}x {gpu_type}...")
    available = client.check_availability(**availability_params)
    if not available:
        print(f"ERROR: No {gpu_type} GPUs available")
        print("Check availability at: https://app.primeintellect.ai/dashboard/create-cluster")
        sys.exit(1)

    # Select cheapest non-runpod GPU
    non_runpod = [opt for opt in available if opt["provider"] != "runpod"]
    if not non_runpod:
        print("ERROR: No non-runpod GPUs available")
        sys.exit(1)
    selected = min(non_runpod, key=lambda x: float(x["prices"]["onDemand"]))
    provider = selected["provider"]
    price = selected["prices"]["onDemand"]
    print(f"Found available GPU:")
    print(f"  Provider: {provider}")
    print(f"  Cloud ID: {selected.get('cloudId')}")
    print(f"  GPU Type: {selected.get('gpuType')}")
    print(f"  Socket: {selected.get('socket')}")
    print(f"  Price: ${price}/hr")

    # Create pod
    instance_name = f"{name_prefix}-{int(time.time())}"
    print(f"\nCreating pod '{instance_name}'...")
    pod = client.create_pod(gpu_config=selected, name=instance_name, team_id=team_id, **create_params)

    pod_id = pod.get("id")
    print(f"Pod created with ID: {pod_id}")
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

    print("\nPod is ready!")

    # Extract SSH connection info from sshConnection (format: "user@host -p port")
    ssh_conn = pod.get("sshConnection", "")
    user_host, _, port_str = ssh_conn.partition(" -p ")
    ssh_user, ssh_host = user_host.split("@", 1)
    ssh_port = int(port_str) if port_str else 22

    return {
        "pod_id": pod_id,
        "instance_name": instance_name,
        "ssh_user": ssh_user,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
    }


def setup_environment(ssh: SSHSession, config: dict, verbose: bool = False) -> ContainerSession:
    """Set up the container environment."""
    instance_cfg = config["instance"]
    docker_image = instance_cfg["docker_image"]

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

    container = ContainerSession(ssh, docker_cmd, CONTAINER_NAME)

    # Clone batchbench repo
    print("\nCloning batchbench repository...")
    repo_url = "https://github.com/doublewordai/batchbench.git"
    stdout, stderr, exit_code = container.exec(f"git clone {repo_url}", stream=verbose)
    if exit_code != 0:
        print(f"Git clone failed: {stderr}")
        sys.exit(1)
    print("Repository cloned successfully")

    # Install dependencies
    print("\nInstalling dependencies...")
    install_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        "cd batchbench && "
        "uv pip install -e ."
    )
    stdout, stderr, exit_code = container.exec(install_cmd, timeout=300, stream=verbose)
    if exit_code != 0:
        print(f"Dependency installation failed: {stderr}")
        print(f"stdout: {stdout}")
        sys.exit(1)
    print("Dependencies installed successfully")

    return container


def run_harness(config_path: str, verbose: bool = False) -> None:
    """Main harness execution."""
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    instance_info = provision_instance(config)
    pod_id = instance_info["pod_id"]
    instance_name = instance_info["instance_name"]
    ssh_user = instance_info["ssh_user"]
    ssh_host = instance_info["ssh_host"]
    ssh_port = instance_info["ssh_port"]
    ssh_key_path = Path(os.environ.get("PRIME_SSH_KEY_PATH"))

    print("\nWaiting for SSH to be ready...")
    time.sleep(10)

    print("\nConnecting via SSH...")
    try:
        with SSHSession(ssh_host, ssh_port, ssh_user, ssh_key_path) as ssh:
            container = setup_environment(ssh, config, verbose)
            start_vllm_server(container, config, verbose)
            synthetic_data_path = run_synthetic_generation(container, config, verbose)
            run_benchmark(container, config, synthetic_data_path, verbose)

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
