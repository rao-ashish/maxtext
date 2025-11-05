"""Runs a sequence of MaxText commands one-by-one, and saves their stdout /
stderr to log files.

Run with: `python scripts/run_experiments.py`
"""

from typing import Any

import os
import sys
import datetime
import subprocess
import multiprocessing
from dataclasses import dataclass

import shlex
import textwrap

from scripts.experiment_configs import ExperimentConfig, EXPERIMENTS


# Where to store stdout / stderr logs for each experiment.
BASE_LOG_DIR = "scripts/outputs"

# Coordinator address for distributed experiments.
JAX_COORDINATOR_ADDRESS = "127.0.0.1:12345"


def build_full_command(experiment: ExperimentConfig) -> str:
    """Build the full command string from an experiment config."""
    command_list = [experiment.base_command]

    for key, value in experiment.overrides.items():
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)
        command_list.append(f" {key}={value_str}")

    if experiment.is_multiprocess:
        command_list.append(" skip_jax_distributed_system=true")

    return " ".join(command_list)


def run_single_process(experiment: ExperimentConfig, log_dir: str):
    """Runs a single-process experiment using subprocess.run."""
    assert not experiment.is_multiprocess
    command = build_full_command(experiment)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{experiment.name}-{timestamp}.log")

    # Combine base environment with experiment-specific env vars.
    run_env = {**os.environ, **experiment.env}

    print(f"--- Running experiment: {experiment.name} ---")
    print(f"Logging output to: {log_filename}")
    print(f"Command:\n{command}\n")

    try:
        with open(log_filename, "w") as log_file:
            subprocess.run(
                command,
                shell=True,  # Needed to interpret the command string correctly
                env=run_env,  # Pass the combined environment
                stdout=log_file,  # Redirect stdout to our log file
                stderr=subprocess.STDOUT,  # Redirect stderr to the same place
                check=True,  # Raise CalledProcessError if command returns non-zero exit code
            )
        print(f"--- Successfully finished experiment: {experiment.name} ---")

    except subprocess.CalledProcessError as e:
        print(f"--- ERROR in experiment: {experiment.name} ---")
        print(f"Command failed with exit code {e.returncode}.")
        print(f"Check log for details: {log_filename}")

    except Exception as e:
        print(f"--- An unexpected error occurred: {experiment.name} ---")
        print(f"Error: {e}")
        print(f"Check log for details: {log_filename}")

    print("=" * 80 + "\n")


def _multiprocess_worker_fn(
    experiment: ExperimentConfig,
    process_idx: int,
    run_log_dir: str,
):
    """Target function for each spawned process in a multiprocess run.
    This function initializes jax.distributed and calls train.main directly.
    """
    import jax
    from MaxText import train

    assert experiment.multiprocess_kwargs is not None

    try:
        os.environ.update(experiment.env)

        log_file_path = os.path.join(run_log_dir, f"process_{process_idx}.log")
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

        num_processes = experiment.multiprocess_kwargs["num_processes"]
        devices_per_process = experiment.multiprocess_kwargs["devices_per_process"]

        print(f"Worker {process_idx}: Initializing JAX distributed...")
        jax.distributed.initialize(
            coordinator_address=JAX_COORDINATOR_ADDRESS,
            process_id=process_idx,
            num_processes=num_processes,
            local_device_ids=[
                (i + process_idx * devices_per_process)
                for i in range(devices_per_process)
            ],
            partition_index=0,  # Assuming single-slice, multi-host not handled here
        )

        print(
            f"Worker {process_idx}: jax.distributed.initialize() complete. "
            f"Local devices: {jax.local_devices()}"
        )
        print(f"Worker {process_idx}: All visible devices: {jax.devices()}")

        # Build the command and parse it into argv for train.main.
        command = build_full_command(experiment)
        # Get rid of "python3 -m"
        cmd_argv = shlex.split(textwrap.dedent(command).strip())[2:]

        print(f"Worker {process_idx}: Starting train.main with args: {cmd_argv}")
        train.main(cmd_argv)

        print(f"Worker {process_idx}: train.main finished.")

    except Exception as e:
        print(f"--- ERROR in worker {process_idx} for experiment {experiment.name} ---")
        print(f"Error: {e}")
        # Re-raise to ensure the process exits with a non-zero code.
        raise e


def run_multiprocess(experiment: ExperimentConfig, log_dir: str):
    """Runs a multi-process experiment by spawning workers."""
    assert experiment.is_multiprocess

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Create a dedicated directory for this run's logs.
    run_log_dir = os.path.join(log_dir, f"{experiment.name}-{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)

    print(f"--- Running multiprocess experiment: {experiment.name} ---")
    print(f"Logging outputs to directory: {run_log_dir}")

    processes = []
    num_processes = experiment.multiprocess_kwargs["num_processes"]

    for process_idx in range(num_processes):
        p = multiprocessing.Process(
            target=_multiprocess_worker_fn,
            args=(
                experiment,
                process_idx,
                run_log_dir,
            ),
        )
        p.start()
        processes.append(p)

    all_success = True
    for p in processes:
        p.join()
        if p.exitcode == 0:
            continue
        all_success = False
        print(
            f"--- ERROR: Process {p.pid} (experiment {experiment.name}) "
            f"exited with code {p.exitcode} ---"
        )

    if all_success:
        print(f"--- Successfully finished experiment: {experiment.name} ---")
    else:
        print(f"--- ERROR in experiment: {experiment.name} ---")
        print(f"One or more processes failed. Check logs in: {run_log_dir}")

    print("=" * 80 + "\n")


def run_experiments(experiments=EXPERIMENTS, log_dir=BASE_LOG_DIR):
    """Iterates through all experiments and runs them."""
    os.makedirs(log_dir, exist_ok=True)

    for experiment in experiments:
        if not experiment.is_multiprocess:
            run_single_process(experiment, log_dir)
        else:
            run_multiprocess(experiment, log_dir)


if __name__ == "__main__":
    # "spawn" is recommended for CUDA and multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    run_experiments()
