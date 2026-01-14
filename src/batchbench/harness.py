"""
Automated benchmarking orchestration via Prime Intellect.

This script:
1. Provisions a GPU instance via Prime Intellect API
2. Connects via SSH
3. Pulls the batchbench Docker image and starts container
4. Clones the batchbench repository (with pre-built Rust binary)
5. Starts vLLM server and waits for readiness
6. Runs benchmark (generation + load test via Rust binary)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import paramiko
import requests
import yaml


# Container name used for docker exec commands
CONTAINER_NAME = "batchbench-run"

PI_API_BASE = "https://api.primeintellect.ai/api/v1"

RESULTS_DIR = Path("results")


class ProvisioningError(Exception):
    """Raised when instance provisioning fails."""
    pass


class PipelineError(Exception):
    """Raised when pipeline execution fails."""
    pass


@dataclass
class ConfigResult:
    """Result of running a single config."""
    success: bool
    error: Optional[str]
    run_dir: Optional[str]


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_queues(path: Path) -> dict[tuple[str, int], list[Path]]:
    """Build GPU-type queues from config path(s).

    Args:
        path: Single config file or directory of configs

    Returns:
        Dict mapping (gpu_type, gpu_count) to list of config paths
    """
    if path.is_file():
        config_paths = [path]
    elif path.is_dir():
        config_paths = sorted(path.glob("*.yaml"))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    queues: dict[tuple[str, int], list[Path]] = {}
    for config_path in config_paths:
        config = load_config(str(config_path))
        avail = config["instance"]["availability"]
        gpu_key = (avail["gpu_type"], avail["gpu_count"])
        queues.setdefault(gpu_key, []).append(config_path)

    return queues


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

    def list_pods(self) -> list[dict]:
        """List all pods for this account."""
        result = self._request("GET", "/pods")
        return result.get("data", [])

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
                chunk = recv(1024).decode(errors='replace')
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
    def from_pod_id(cls, pod_id: str, client: PrimeIntellectClient) -> "Instance":
        """Resume with an existing pod by ID."""
        print(f"Resuming pod: {pod_id}")
        return cls(pod_id, client, timeout=60)  # Short timeout - should already be active

    @classmethod
    def provision(cls, instance_cfg: dict, client: PrimeIntellectClient, include_spot: bool = False) -> "Instance":
        """Provision a new GPU instance via Prime Intellect API."""
        availability_params = instance_cfg["availability"]
        create_params = instance_cfg["create"]
        provision_timeout = instance_cfg.get("provision-timeout", 600)
        team_id = os.environ["PRIME_TEAM_ID"]

        # Check availability
        gpu_type = availability_params["gpu_type"]
        gpu_count = availability_params["gpu_count"]
        print(f"Checking availability for {gpu_count}x {gpu_type}...")
        available = client.check_availability(**availability_params)
        if not available:
            raise ProvisioningError(
                f"No {gpu_type} GPUs available. "
                "Check availability at: https://app.primeintellect.ai/dashboard/create-cluster"
            )

        # Select cheapest GPU, excluding problematic providers and optionally spot instances
        excluded_providers = {"runpod", "hyperstack"}
        valid_gpus = [
            opt for opt in available
            if opt["provider"] not in excluded_providers
            and (include_spot or opt.get("isSpot") is not True)
        ]
        if not valid_gpus:
            excluded = [*excluded_providers] + ([] if include_spot else ["spot"])
            raise ProvisioningError(f"No GPUs available (excluded: {', '.join(excluded)})")
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
        except (TimeoutError, RuntimeError) as e:
            print("Cleaning up...")
            try:
                client.terminate_pod(pod_id)
            except Exception:
                pass
            raise ProvisioningError(f"Pod failed to become ready: {e}") from e

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

    def setup(self, config: dict) -> None:
        """Ensure container is running with correct image and repo cloned."""
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
                raise PipelineError(f"Docker pull failed: {stderr}")
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
                raise PipelineError(f"Failed to start container: {stderr}")
            print("Container started successfully")

        # Clone batchbench repo (skip if already cloned)
        if not self._is_repo_cloned():
            print("\nCloning batchbench repository...")
            # TODO: Change back to main after merge is complete
            repo_url = "https://github.com/doublewordai/batchbench.git -b harness"
            stdout, stderr, exit_code = self.exec(f"git clone {repo_url}", stream=True)
            if exit_code != 0:
                raise PipelineError(f"Git clone failed: {stderr}")
            print("Repository cloned successfully")

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

    def __exit__(self, *_args) -> None:
        self.close()


def build_cli_command(base: str, args: dict) -> str:
    cmd_args = []
    for arg, value in args.items():
        if isinstance(value, bool) and value:
            cmd_args.append(f"--{arg}")
        elif not isinstance(value, bool):
            cmd_args.append(f"--{arg} {value}")
    return base + " " + " ".join(cmd_args)


def run_benchmark(env: RemoteEnvironment, config: dict, run_dir: Path) -> None:
    """Run the benchmark using the Rust binary inside the container."""
    bench_cfg = config["benchmark"]
    vllm_cfg = config["vllm"]
    vllm_args = vllm_cfg.get("args") or {}

    model = vllm_cfg["model"]
    port = vllm_args.get("port", 8000)
    host = f"http://localhost:{port}"

    # Build args for Rust binary
    bench_args = {"model": model, "host": host, **bench_cfg}
    benchmark_cmd = build_cli_command("batchbench/bin/batchbench", bench_args)

    print("\nRunning benchmark...")
    stdout, stderr, exit_code = env.exec(benchmark_cmd, timeout=3600, stream=True)
    if exit_code != 0:
        raise PipelineError(f"Benchmark failed: {stderr}")

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
    raise PipelineError(f"vLLM server not ready after {timeout}s")


def stop_vllm_server(env: RemoteEnvironment) -> None:
    env.exec("pkill -f 'vllm serve'; sleep 2; pkill -9 -f 'vllm serve' 2>/dev/null")


def start_vllm_server(env: RemoteEnvironment, config: dict) -> None:
    """Start vLLM server inside the container and wait for it to be ready."""
    # Stop any existing vLLM server first
    stop_vllm_server(env)

    vllm_cfg = config["vllm"]
    model = vllm_cfg["model"]
    vllm_args = vllm_cfg.get("args") or {}
    vllm_env = vllm_cfg.get("env") or {}
    startup_timeout = vllm_cfg.get("startup-timeout", 600)
    port = vllm_args.get("port", 8000)

    vllm_cmd = build_cli_command(f"vllm serve {model}", vllm_args)
    env_exports = " ".join(f"{k}={v}" for k, v in vllm_env.items())
    full_cmd = (
        "source /opt/batchbench/.venv/bin/activate && "
        f"{env_exports} nohup {vllm_cmd} > /tmp/vllm.log 2>&1 &"
    )

    print(f"\nStarting vLLM server with model: {model}")
    _, stderr, exit_code = env.exec(full_cmd)
    if exit_code != 0:
        raise PipelineError(f"Failed to start vLLM server: {stderr}")
    print("vLLM server process started")

    wait_for_vllm_ready(env, port, startup_timeout)


class QueueWorker:
    """Worker that processes a queue of configs for a specific GPU type."""

    def __init__(
        self,
        gpu_key: tuple[str, int],
        config_paths: list[Path],
        include_spot: bool,
        save_server_logs: bool,
        keep_alive: bool = False,
        resume_pod_id: Optional[str] = None,
    ):
        self.gpu_key = gpu_key
        self.config_paths = config_paths
        self.include_spot = include_spot
        self.save_server_logs = save_server_logs
        self.keep_alive = keep_alive
        self.resume_pod_id = resume_pod_id
        self.results: list[ConfigResult] = []
        self.instance: Optional[Instance] = None

    def run(self) -> list[ConfigResult]:
        """Process all configs in this queue. Returns results for each config."""
        try:
            instance_cfg = load_config(str(self.config_paths[0]))["instance"]
            self.instance = self.get_instance(instance_cfg)
        except Exception as e:
            for path in self.config_paths:
                self.results.append(ConfigResult(
                    success=False,
                    error=f"Provisioning failed: {e}",
                    run_dir=None,
                ))
            return self.results

        try:
            if not self.resume_pod_id:
                time.sleep(10)  # Wait for SSH to be ready

            for config_path in self.config_paths:
                result = self.run_pipeline(config_path)
                self.results.append(result)
        finally:
            if self.instance:
                if self.keep_alive:
                    print(f"\nInstance kept alive: {self.instance.pod_id}")
                    print(f"SSH: ssh {self.instance.ssh_user}@{self.instance.ssh_host} -p {self.instance.ssh_port}")
                    print(f"To reuse: python -m batchbench.harness <config> --resume {self.instance.pod_id}")
                    print(f"To terminate: prime pods terminate {self.instance.pod_id}")
                else:
                    try:
                        self.instance.terminate()
                    except Exception as e:
                        print(f"WARNING: Failed to terminate instance {self.instance.pod_id}: {e}")
                        print(f"Manually terminate at: https://app.primeintellect.ai/dashboard")

        return self.results

    def get_instance(self, instance_cfg: dict, max_retries: int = 10, retry_delay: int = 60) -> Instance:
        """Get instance - resume existing or provision new with retries."""
        client = PrimeIntellectClient(os.environ["PRIME_API_KEY"])

        # Resume existing pod if specified
        if self.resume_pod_id:
            return Instance.from_pod_id(self.resume_pod_id, client)

        # Provision with retries
        errors = set()
        for attempt in range(max_retries):
            try:
                return Instance.provision(instance_cfg, client, include_spot=self.include_spot)
            except ProvisioningError as e:
                errors.add(str(e))
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        raise ProvisioningError(f"Failed after {max_retries} attempts: {'; '.join(errors)}")

    def run_pipeline(self, config_path: Path) -> ConfigResult:
        """Run single config on the instance."""
        config = load_config(str(config_path))
        ssh_key_path = Path(os.environ["PRIME_SSH_KEY_PATH"])
        run_dir = save_run_config(config)

        try:
            with RemoteEnvironment(self.instance, ssh_key_path) as env:
                env.setup(config)
                start_vllm_server(env, config)
                run_benchmark(env, config, run_dir)
                if self.save_server_logs:
                    fetch_server_logs(env, run_dir)
            return ConfigResult(
                success=True,
                error=None,
                run_dir=str(run_dir),
            )
        except (PipelineError, Exception) as e:
            return ConfigResult(
                success=False,
                error=str(e),
                run_dir=str(run_dir),
            )


def print_summary(results: list[ConfigResult]) -> None:
    """Print summary of all results."""
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total configs: {len(results)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  - {r.run_dir or 'N/A'}: {r.error}")

    if succeeded:
        print("\nSucceeded:")
        for r in succeeded:
            print(f"  - {r.run_dir}")


def run_queues(
    config_path: Path,
    resume_pod_id: Optional[str] = None,
    include_spot: bool = False,
    save_server_logs: bool = False,
    keep_alive: bool = False,
) -> None:
    """Main entry point - process configs with parallel queue execution."""
    queues = build_queues(config_path)
    if not queues:
        print(f"No configs found at {config_path}")
        sys.exit(1)

    workers = [
        QueueWorker(
            gpu_key=gpu_key,
            config_paths=paths,
            include_spot=include_spot,
            save_server_logs=save_server_logs,
            keep_alive=keep_alive,
            resume_pod_id=resume_pod_id,
        )
        for gpu_key, paths in queues.items()
    ]

    all_results: list[ConfigResult] = []
    print(f"\nStarting {len(workers)} parallel queue(s)...\n")
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(w.run) for w in workers]
        for future in as_completed(futures):
            all_results.extend(future.result())

    print_summary(all_results)


def fetch_server_logs(env: RemoteEnvironment, run_dir: Path) -> None:
    """Save vLLM server logs from container to local run directory."""
    print("\nSaving vLLM server logs...")
    stdout, _, rc = env.exec("cat /tmp/vllm.log")
    if rc == 0:
        local_log_path = run_dir / "vllm.log"
        with open(local_log_path, "w") as f:
            f.write(stdout)
        print(f"Server logs saved to: {local_log_path}")
    else:
        print("Warning: Could not retrieve vLLM server logs")


def main():
    parser = argparse.ArgumentParser(
        description="BatchBench Harness - Automated GPU instance provisioning and benchmarking"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/harness.yaml",
        help="Path to config file or directory of configs",
    )
    parser.add_argument(
        "--resume",
        metavar="POD_ID",
        help="Resume using an existing pod (single config only)",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep instance alive after completion (single config only)",
    )
    parser.add_argument(
        "--include-spot",
        action="store_true",
        help="Include spot instances when selecting GPUs",
    )
    parser.add_argument(
        "--save-server-logs",
        action="store_true",
        help="Save vLLM server logs to run directory",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Path not found: {args.config}")
        sys.exit(1)

    is_single = config_path.is_file()
    if args.resume and not is_single:
        print("ERROR: --resume only supported for single config files")
        sys.exit(1)
    if args.keep_alive and not is_single:
        print("ERROR: --keep-alive only supported for single config files")
        sys.exit(1)

    run_queues(
        config_path,
        resume_pod_id=args.resume,
        include_spot=args.include_spot,
        save_server_logs=args.save_server_logs,
        keep_alive=args.keep_alive,
    )


if __name__ == "__main__":
    main()
