#!/usr/bin/env python3
"""
BatchBench Harness - Automated benchmarking orchestration via Prime Intellect.

This script:
1. Provisions a GPU instance via Prime Intellect API
2. Connects via SSH
3. Pulls the batchbench Docker image
4. Runs the container, clones repo, and runs nvidia-smi
"""

import argparse
import os
import sys
import time
from pathlib import Path

import paramiko
import requests
import yaml


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

            # Run container: clone repo and run nvidia-smi
            print("\nRunning container...")
            repo_url = "https://github.com/doublewordai/batchbench.git"
            stdout, stderr, exit_code = ssh.run(
                f'{docker_cmd} run --rm --gpus all {docker_image} bash -c "echo Successfully inside batchbench container! && git clone {repo_url} && nvidia-smi"',
                stream=verbose,
            )

            if exit_code == 0:
                print("\n" + "=" * 60)
                print("Container output:")
                print("=" * 60)
                print(stdout)
            else:
                print(f"Container run failed with exit code {exit_code}")
                if stderr:
                    print(f"stderr: {stderr}")

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
