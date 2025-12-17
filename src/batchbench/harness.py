"""Minimal benchmarking harness: start server, generate data, run benchmark, record results."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

import yaml


RUNS_DIR = Path("runs")


def ensure_source_data(config: dict) -> Path:
    """Download or verify source dataset exists. Returns path to the data."""
    source = config.get("source", {})
    source_type = source.get("type", "local")
    path = Path(source.get("path", "data/source.jsonl"))

    if path.exists():
        print(f"Source data already exists: {path}")
        return path

    if source_type == "local":
        raise RuntimeError(f"Source file not found: {path}")

    if source_type == "wildchat":
        # Ensure pip and required packages are installed
        subprocess.call([sys.executable, "-m", "ensurepip", "--upgrade"], stderr=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "tqdm", "matplotlib"])

        # Call the existing download script
        cmd = [
            sys.executable, "examples/wildchat/download_wildchat.py",
            "--output", str(path),
            "--num-samples", str(source.get("num_samples", 50000)),
            "--seed", str(source.get("seed", 42)),
        ]
        if source.get("min_turns"):
            cmd.extend(["--min-turns", str(source["min_turns"])])
        if source.get("max_turns"):
            cmd.extend(["--max-turns", str(source["max_turns"])])

        print(f"Downloading source data: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError("Source data download failed")
        return path

    raise RuntimeError(f"Unknown source type: {source_type}")


def fit_distribution(source_path: Path, stats_path: Path, plot_path: Path, source_config: dict) -> dict:
    """Fit lognormal distribution to source data token lengths."""
    # Call the existing plot_conversation_lengths script with --output-stats
    cmd = [
        sys.executable, "examples/wildchat/plot_conversation_lengths.py",
        str(source_path),
        "--output-stats", str(stats_path),
        "--output", str(plot_path),
    ]

    # Pass through tokenization options from source config
    if source_config.get("tokenizer"):
        cmd.extend(["--tokenizer", source_config["tokenizer"]])
    if source_config.get("use_chars"):
        cmd.append("--use-chars")
    if source_config.get("no_chat_template"):
        cmd.append("--no-chat-template")
    if source_config.get("bins"):
        cmd.extend(["--bins", str(source_config["bins"])])

    print(f"Fitting distribution: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Distribution fitting failed")

    # Parse the JSON output
    with open(stats_path) as f:
        stats = json.load(f)

    print(f"  Fitted median: {stats['dist_median']:.0f}")
    print(f"  Fitted sigma: {stats['dist_sigma']:.3f}")
    print(f"  P99 (max): {stats['dist_max']}")

    return stats


def load_config(path: Path) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def start_server(config: dict, log_file: Path, verbose: bool = False) -> subprocess.Popen:
    """Start vLLM server and wait for it to be healthy."""
    server = config["server"]

    cmd = ["vllm", "serve", server["model"]]
    for key, value in server.get("args", {}).items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            # If False, skip the flag entirely
        else:
            cmd.extend([f"--{key}", str(value)])

    env = os.environ.copy()
    env.update({k: str(v) for k, v in server.get("env", {}).items()})

    print(f"Starting server: {' '.join(cmd)}")
    if server.get("env"):
        print(f"Environment: {server['env']}")

    log_handle = open(log_file, "w")

    if verbose:
        # Tee output to both terminal and log file
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

        def tee_output():
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
                log_handle.flush()

        tee_thread = threading.Thread(target=tee_output, daemon=True)
        tee_thread.start()
    else:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Wait for health
    port = server.get("args", {}).get("port", 8000)
    url = f"http://127.0.0.1:{port}/health"
    timeout = server.get("startup_timeout", 600)

    print(f"Waiting for server (timeout: {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"Server ready after {time.time() - start:.1f}s")
                    return proc
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(2)

    proc.terminate()
    raise RuntimeError(f"Server failed to start within {timeout}s")


def stop_server(proc: subprocess.Popen) -> None:
    """Stop the vLLM server."""
    print("Stopping server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, OSError):
        pass
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    print("Server stopped")


def generate_dataset(config: dict, output_path: Path, model: str) -> None:
    """Generate synthetic dataset."""
    dataset = config["dataset"]

    args = [
        sys.executable, "-m", "batchbench.generate",
        "--output", str(output_path),
        "--count", str(dataset["count"]),
        "--tokenizer-model", model,
        "--model", model,
    ]

    if dataset.get("dist_mode") == "lognormal":
        args.extend([
            "--dist-mode", "lognormal",
            "--dist-median", str(dataset["dist_median"]),
            "--dist-sigma", str(dataset.get("dist_sigma", 0.5)),
        ])
        if dataset.get("dist_max"):
            args.extend(["--dist-max", str(dataset["dist_max"])])
    else:
        args.extend([
            "--dist-mode", "fixed",
            "--approx-input-tokens", str(dataset.get("approx_input_tokens", 512)),
        ])

    if dataset.get("prefix_overlap"):
        args.extend(["--prefix-overlap", str(dataset["prefix_overlap"])])
    if dataset.get("seed") is not None:
        args.extend(["--seed", str(dataset["seed"])])

    print(f"Generating dataset: {dataset['count']} requests")
    result = subprocess.run(args)
    if result.returncode != 0:
        raise RuntimeError("Dataset generation failed")


def run_benchmark(config: dict, dataset_path: Path, results_path: Path, model: str) -> None:
    """Run the online benchmark."""
    bench = config["benchmark"]
    server = config["server"]
    port = server.get("args", {}).get("port", 8000)

    # Get path to bundled binary
    from importlib import resources
    resource = resources.files("batchbench").joinpath("bin", "batchbench")

    with resources.as_file(resource) as binary:
        cmd = [
            str(binary),
            "--jsonl", str(dataset_path),
            "--model", model,
            "--host", f"http://127.0.0.1:{port}",
            "--results-csv", str(results_path),
        ]

        for key, value in bench.items():
            if key.startswith("_"):
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        print(f"Running benchmark: {bench.get('users', 1)} users, {bench.get('requests_per_user', 1)} req/user")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError("Benchmark failed")


def run(config_path: Path, name: str, verbose: bool = False) -> Path:
    """Execute a complete benchmark run."""
    config = load_config(config_path)
    model = config["server"]["model"]

    # Create run directory first (needed for stats output)
    run_dir = RUNS_DIR / name
    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    # Auto-fit distribution if source section exists and fit_from_source is true
    dataset = config.get("dataset", {})
    source_config = config.get("source", {})
    if source_config and dataset.get("fit_from_source"):
        source_path = ensure_source_data(config)
        stats_path = run_dir / "fitted_stats.json"
        plot_path = run_dir / "source_distribution.png"
        fitted = fit_distribution(source_path, stats_path, plot_path, source_config)

        # Update config with fitted values
        config["dataset"]["dist_median"] = fitted["dist_median"]
        config["dataset"]["dist_sigma"] = fitted["dist_sigma"]
        config["dataset"]["dist_max"] = fitted["dist_max"]

        # Ensure dist_mode is lognormal when fitting
        if config["dataset"].get("dist_mode") != "lognormal":
            print("Setting dist_mode to lognormal (fit_from_source requires it)")
            config["dataset"]["dist_mode"] = "lognormal"

    # Save config (includes fitted values if auto-fitted)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    dataset_path = run_dir / "dataset.jsonl"
    results_path = run_dir / "results.csv"
    log_path = run_dir / "server.log"

    proc = None
    try:
        proc = start_server(config, log_path, verbose=verbose)
        generate_dataset(config, dataset_path, model)
        run_benchmark(config, dataset_path, results_path, model)
        print(f"\nRun complete: {run_dir}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if proc:
            stop_server(proc)

    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmarking harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a benchmark")
    run_parser.add_argument("config", type=Path, help="Path to config YAML")
    run_parser.add_argument("--name", required=True, help="Name for this run")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Show server output in terminal")

    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            run(args.config, args.name, verbose=args.verbose)
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
