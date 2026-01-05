"""
Automated benchmarking orchestration via Prime Intellect.

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
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import paramiko
import requests
import yaml


# Container name used for docker exec commands
CONTAINER_NAME = "batchbench-run"

PI_API_BASE = "https://api.primeintellect.ai/api/v1"

RESULTS_DIR = Path("results")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_run_config(config: dict) -> Path:
    """Save config to results directory, return the run directory path.

    Structure: results/{model_name}/{config_hash}/config.yaml
    """
    # Build run directory path
    model = config["vllm"]["model"]
    model_dir = RESULTS_DIR / re.sub(r'[/\\]', '--', model)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()[:8]
    run_dir = model_dir / config_hash

    # Save config
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return run_dir


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

    def create_pod(self, gpu_config: dict, team_id: str, **pod_params) -> dict:
        payload = {
            "pod": {
                "cloudId": gpu_config["cloudId"],
                "gpuType": gpu_config["gpuType"],
                "socket": gpu_config["socket"],
                "gpuCount": gpu_config["gpuCount"],
                "dataCenterId": gpu_config["dataCenter"],
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
            install_status = (pod.get("installationStatus") or "").lower()
            ssh_conn = pod.get("sshConnection")

            print(f"  Pod status: {status}, installation: {install_status}")

            if status in ("stopped", "error", "terminated"):
                raise RuntimeError(f"Pod entered {status} state")
            if status == "active" and ssh_conn:
                return pod
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

        if not stream:
            exit_code = channel.recv_exit_status()
            return stdout.read().decode(), stderr.read().decode(), exit_code

        stdout_chunks, stderr_chunks = [], []

        def drain(is_ready, recv, chunks):
            while is_ready():
                chunk = recv(1024).decode()
                print(chunk, end="", flush=True)
                chunks.append(chunk)

        while not channel.exit_status_ready():
            drain(channel.recv_ready, channel.recv, stdout_chunks)
            drain(channel.recv_stderr_ready, channel.recv_stderr, stderr_chunks)
            time.sleep(0.1)

        drain(channel.recv_ready, channel.recv, stdout_chunks)
        drain(channel.recv_stderr_ready, channel.recv_stderr, stderr_chunks)

        return "".join(stdout_chunks), "".join(stderr_chunks), channel.recv_exit_status()

    def close(self):
        self.client.close()


class Instance:
    def __init__(self, pod_id: str, client: PrimeIntellectClient, timeout: int = 600):
        """Wait for pod to be ready and extract SSH connection info."""
        self._client = client
        self.pod_id = pod_id

        print(f"Waiting for pod to be ready (timeout: {timeout}s)...")
        try:
            pod = client.wait_for_pod_ready(pod_id, timeout=timeout)
        except (TimeoutError, RuntimeError) as e:
            print(f"ERROR: {e}")
            raise

        print("Pod is ready!")
        ssh_conn = pod.get("sshConnection", "")
        user_host, _, port_str = ssh_conn.partition(" -p ")
        self.ssh_user, self.ssh_host = user_host.split("@", 1)
        self.ssh_port = int(port_str) if port_str else 22

    @classmethod
    def from_pod_id(cls, pod_id: str) -> "Instance":
        """Resume with an existing pod by ID."""
        client = PrimeIntellectClient(os.environ["PRIME_API_KEY"])
        print(f"Resuming pod: {pod_id}")
        return cls(pod_id, client, timeout=60)  # Short timeout - should already be active

    @classmethod
    def provision(cls, config: dict) -> "Instance":
        """Provision a new GPU instance via Prime Intellect API."""
        instance_cfg = config["instance"]
        availability_params = instance_cfg["availability"]
        create_params = instance_cfg["create"]
        provision_timeout = instance_cfg.get("provision-timeout", 600)
        team_id = os.environ["PRIME_TEAM_ID"]

        client = PrimeIntellectClient(os.environ["PRIME_API_KEY"])

        # Check availability
        gpu_type = availability_params["gpu_type"]
        gpu_count = availability_params["gpu_count"]
        print(f"Checking availability for {gpu_count}x {gpu_type}...")
        available = client.check_availability(**availability_params)
        if not available:
            print(f"ERROR: No {gpu_type} GPUs available")
            print("Check availability at: https://app.primeintellect.ai/dashboard/create-cluster")
            sys.exit(1)

        # Select cheapest GPU, excluding problematic providers
        excluded_providers = {"runpod", "hyperstack"}
        valid_gpus = [opt for opt in available if opt["provider"] not in excluded_providers]
        if not valid_gpus:
            print(f"ERROR: No GPUs available (excluded: {', '.join(excluded_providers)})")
            sys.exit(1)
        selected = min(valid_gpus, key=lambda x: float(x["prices"]["onDemand"]))
        provider = selected["provider"]
        price = selected["prices"]["onDemand"]
        print(f"Found available GPU:")
        print(f"  Provider: {provider}")
        print(f"  Cloud ID: {selected.get('cloudId')}")
        print(f"  GPU Type: {selected.get('gpuType')}")
        print(f"  Socket: {selected.get('socket')}")
        print(f"  Price: ${price}/hr")

        # Create pod
        print("\nCreating pod...")
        pod = client.create_pod(gpu_config=selected, team_id=team_id, **create_params)
        pod_id = pod.get("id")
        print(f"Pod created with ID: {pod_id}")

        try:
            return cls(pod_id, client, timeout=provision_timeout)
        except (TimeoutError, RuntimeError):
            print("Cleaning up...")
            try:
                client.terminate_pod(pod_id)
            except Exception:
                pass
            sys.exit(1)

    def terminate(self):
        """Terminate this instance."""
        self._client.terminate_pod(self.pod_id)


class RemoteEnvironment:
    """Manages the remote execution environment (SSH + container)."""

    def __init__(self, instance: Instance, ssh_key_path: Path):
        self.ssh = SSHSession(
            instance.ssh_host,
            instance.ssh_port,
            instance.ssh_user,
            ssh_key_path,
        )
        # Detect docker command (with or without sudo)
        _, _, rc = self.ssh.run("docker info > /dev/null 2>&1")
        self._docker_cmd = "docker" if rc == 0 else "sudo docker"

    def _get_container_info(self) -> tuple[bool, str | None]:
        """Get container state: (is_running, image) or (False, None) if no container."""
        stdout, _, exit_code = self.ssh.run(
            f"{self._docker_cmd} inspect -f '{{{{.State.Running}}}} {{{{.Config.Image}}}}' {CONTAINER_NAME}"
        )
        if exit_code != 0:
            return (False, None)  # Container doesn't exist

        is_running, image = stdout.strip().split(" ", 1)
        return (is_running == "true", image)

    def _stop_and_remove_container(self) -> None:
        self.ssh.run(f"{self._docker_cmd} stop {CONTAINER_NAME} 2>/dev/null")
        self.ssh.run(f"{self._docker_cmd} rm {CONTAINER_NAME} 2>/dev/null")

    def _is_repo_cloned(self) -> bool:
        """Check if batchbench repo was cloned."""
        _, _, exit_code = self.exec("test -d batchbench")
        return exit_code == 0

    def _is_package_installed(self) -> bool:
        """Check if batchbench package was installed."""
        _, _, exit_code = self.exec("/opt/batchbench/.venv/bin/pip show batchbench")
        return exit_code == 0

    def setup(self, config: dict) -> None:
        """Ensure container is running with correct image, repo cloned, deps installed."""
        docker_image = config["instance"]["docker-image"]
        is_running, current_image = self._get_container_info()

        # Ensure correct container is running
        if not is_running or current_image != docker_image:
            if current_image is not None:
                print(f"\nRemoving container with image: {current_image}")
                self._stop_and_remove_container()

            print(f"\nPulling Docker image {docker_image}...")
            stdout, stderr, exit_code = self.ssh.run(
                f"{self._docker_cmd} pull {docker_image}",
                timeout=600,
                stream=True,
            )
            if exit_code != 0:
                print(f"Docker pull failed: {stderr}")
                sys.exit(1)
            print("Docker image pulled successfully")

            print(f"\nStarting container '{CONTAINER_NAME}'...")
            docker_run_cmd = (
                f"{self._docker_cmd} run -d "
                f"--name {CONTAINER_NAME} "
                f"--gpus all "
                f"{docker_image} "
                f"sleep infinity"
            )
            stdout, stderr, exit_code = self.ssh.run(docker_run_cmd, stream=True)
            if exit_code != 0:
                print(f"Failed to start container: {stderr}")
                sys.exit(1)
            print("Container started successfully")

        # Clone batchbench repo (skip if already cloned)
        if not self._is_repo_cloned():
            print("\nCloning batchbench repository...")
            repo_url = "https://github.com/doublewordai/batchbench.git"
            stdout, stderr, exit_code = self.exec(f"git clone {repo_url}", stream=True)
            if exit_code != 0:
                print(f"Git clone failed: {stderr}")
                sys.exit(1)
            print("Repository cloned successfully")

        # Install dependencies (skip if already installed)
        if not self._is_package_installed():
            print("\nInstalling dependencies...")
            install_cmd = (
                "source /opt/batchbench/.venv/bin/activate && "
                "cd batchbench && "
                "uv pip install -e '.[generate]'"
            )
            stdout, stderr, exit_code = self.exec(install_cmd, timeout=300, stream=True)
            if exit_code != 0:
                print(f"Dependency installation failed: {stderr}")
                print(f"stdout: {stdout}")
                sys.exit(1)
            print("Dependencies installed successfully")

    def exec(self, cmd: str, timeout: int = 300, stream: bool = False) -> tuple[str, str, int]:
        """Execute a command inside the container."""
        return self.ssh.run(
            f'{self._docker_cmd} exec {CONTAINER_NAME} bash -c "{cmd}"',
            timeout=timeout,
            stream=stream,
        )

    def close(self) -> None:
        """Close SSH connection."""
        if self.ssh:
            self.ssh.close()

    def __enter__(self) -> "RemoteEnvironment":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def build_cli_command(base: str, args: dict) -> str:
    cmd_args = []
    for arg, value in args.items():
        if isinstance(value, bool) and value:
            cmd_args.append(f"--{arg}")
        elif not isinstance(value, bool):
            cmd_args.append(f"--{arg} {value}")
    return base + " " + " ".join(cmd_args)


def run_synthetic_generation(env: RemoteEnvironment, config: dict) -> str:
    """Generate synthetic data inside the container. Returns the output file path."""
    gen_cfg = config["generate"]
    output = gen_cfg.get("output", "data/output.jsonl")

    generate_cmd = build_cli_command("python -m batchbench.generate", gen_cfg)
    full_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"cd batchbench && {generate_cmd}"
    )

    print(f"\nGenerating synthetic data...")
    stdout, stderr, exit_code = env.exec(full_cmd, timeout=600, stream=True)
    if exit_code != 0:
        print(f"Synthetic generation failed: {stderr}")
        sys.exit(1)

    print("Synthetic data generation complete")
    return output


def run_benchmark(env: RemoteEnvironment, config: dict, synthetic_data_path: str, run_dir: Path) -> None:
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
    stdout, stderr, exit_code = env.exec(full_cmd, timeout=3600, stream=True)
    if exit_code != 0:
        print(f"Benchmark failed: {stderr}")
        sys.exit(1)

    remote_results = bench_cfg.get("results-csv")
    if remote_results:
        stdout, _, rc = env.exec(f"cat {remote_results}")
        if rc == 0:
            local_results = run_dir / "results.csv"
            with open(local_results, "w") as f:
                f.write(stdout)
            print(f"Results saved to: {local_results}")

    print("\nBenchmark complete!")


def wait_for_vllm_ready(env: RemoteEnvironment, port: int, timeout: int) -> None:
    print(f"\nWaiting for vLLM server to be ready (timeout: {timeout}s)...")
    print("-" * 60)

    health_url = f"http://localhost:{port}/health"
    poll_interval = 5
    start_time = time.time()
    last_log_line = 0

    while time.time() - start_time < timeout:
        # Fetch and print new log lines
        stdout, _, _ = env.exec(
            f"tail -n +{last_log_line + 1} /tmp/vllm.log 2>/dev/null",
            timeout=10,
        )
        if stdout.strip():
            print(stdout, end="", flush=True)
            last_log_line += stdout.count("\n")

        # Check health endpoint
        stdout, _, exit_code = env.exec(
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


def stop_vllm_server(env: RemoteEnvironment) -> None:
    env.exec("pkill -f 'vllm serve'; sleep 2; pkill -9 -f 'vllm serve' 2>/dev/null")


def start_vllm_server(env: RemoteEnvironment, config: dict) -> None:
    """Start vLLM server inside the container and wait for it to be ready."""
    # Stop any existing vLLM server first
    stop_vllm_server(env)

    vllm_cfg = config["vllm"]
    model = vllm_cfg["model"]
    args = vllm_cfg.get("args") or {}
    vllm_env = vllm_cfg.get("env") or {}
    startup_timeout = vllm_cfg.get("startup-timeout", 600)
    port = args.get("port", 8000)

    vllm_cmd = build_cli_command(f"vllm serve {model}", args)
    env_exports = " ".join(f"{k}={v}" for k, v in vllm_env.items())
    full_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"{env_exports} nohup {vllm_cmd} > /tmp/vllm.log 2>&1 &"
    )

    print(f"\nStarting vLLM server with model: {model}")
    stdout, stderr, exit_code = env.exec(full_cmd)
    if exit_code != 0:
        print(f"Failed to start vLLM server: {stderr}")
        sys.exit(1)
    print("vLLM server process started")

    wait_for_vllm_ready(env, port, startup_timeout)


def run_harness(config_path: str, resume_pod_id: str = None) -> None:
    """Main harness execution."""
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    ssh_key_path = Path(os.environ["PRIME_SSH_KEY_PATH"])

    run_dir = save_run_config(config)
    print(f"Run directory: {run_dir}")

    # Step 1: Get instance (provision new or reuse existing)
    if resume_pod_id:
        instance = Instance.from_pod_id(resume_pod_id)
    else:
        instance = Instance.provision(config)
        print("Waiting for SSH to be ready...")
        time.sleep(10)

    # Step 2: Connect and run pipeline
    print("\nConnecting via SSH...")
    try:
        with RemoteEnvironment(instance, ssh_key_path) as env:
            env.setup(config)
            start_vllm_server(env, config)
            synthetic_data_path = run_synthetic_generation(env, config)
            run_benchmark(env, config, synthetic_data_path, run_dir)

    except Exception as e:
        print(f"Error: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("Instance Summary")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Pod ID: {instance.pod_id}")
    print(f"SSH: ssh {instance.ssh_user}@{instance.ssh_host} -p {instance.ssh_port}")
    print(f"\nTo terminate: prime pods terminate {instance.pod_id}")


def main():
    parser = argparse.ArgumentParser(description="BatchBench Harness - Automated GPU instance provisioning and benchmarking")
    parser.add_argument("config", nargs="?", default="configs/harness.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, metavar="POD_ID", help="Resume with existing pod")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    run_harness(args.config, resume_pod_id=args.resume)


if __name__ == "__main__":
    main()
