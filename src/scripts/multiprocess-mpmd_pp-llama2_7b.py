"""Runs multiprocess MaxText training after initializing a JAX distributed context.

To get a profile, run wtih 

    nsys profile \
        --output scripts/outputs/run_multiprocess.nsys-rep \
        --cpuctxsw=none \
        --trace=cublas,cuda,cudnn,cusolver,nvtx,osrt,python-gil \
        --force-overwrite true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        --python-sampling=true \
        python scripts/run_multiprocess.py
"""

import multiprocessing

import shlex
import textwrap

from MaxText import train


TOTAL_NUM_DEVICES = 8
NUM_PROCESSES = 8
DEVICES_PER_PROCESS = 1

assert TOTAL_NUM_DEVICES >= NUM_PROCESSES * DEVICES_PER_PROCESS, (
    "Not enough devices available."
)

JAX_COORDINATOR_ADDRESS = "127.0.0.1:12345"

MAXTEXT_DIR = "/opt/maxtext"

MAXTEXT_COMMAND = """
    python3 -m MaxText.train maxtext/configs/base.yml \
        run_name=logdir \
        model_name=llama2-7b \
        steps=20 \
        per_device_batch_size=2 \
        enable_checkpointing=false \
        base_output_directory=logs \
        dataset_path=local \
        dataset_type=synthetic \
        hardware=gpu \
        enable_goodput_recording=false \
        monitor_goodput=false \
        enable_checkpoint_cloud_logger=false \
        dcn_fsdp_parallelism=1 \
        ici_fsdp_parallelism=1 \
        ici_data_parallelism=2 \
        dcn_data_parallelism=1 \
        ici_tensor_parallelism=1 \
        dcn_tensor_parallelism=1 \
        ici_pipeline_parallelism=4 \
        dcn_pipeline_parallelism=1 \
        remat_policy=minimal_with_context \
        gradient_clipping_threshold=0 \
        attention=cudnn_flash_te \
        use_mpmd_pp=true \
        num_layers_per_pipeline_stage=1 \
        profiler=nsys \
        skip_first_n_steps_for_profiler=10 \
        profiler_steps=9 \
        max_segments_per_seq=32 \
        scan_layers=false
"""


def worker_fn(process_idx: int, num_processes: int):
    import os
    os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["NVTE_FUSED_ATTN"] = "1"

    import jax

    jax.distributed.initialize(
        coordinator_address=JAX_COORDINATOR_ADDRESS,
        process_id=process_idx,
        num_processes=NUM_PROCESSES,
        local_device_ids=[
            (i + process_idx * DEVICES_PER_PROCESS) for i in range(DEVICES_PER_PROCESS)
        ],
        partition_index=0,
    )

    print(
        f"After jax.distributed.initialize(), process {process_idx} "
        f"sees devices: {jax.devices()}."
    )

    # Get rid of "python3 -m".
    cmd_argv = shlex.split(textwrap.dedent(MAXTEXT_COMMAND).strip())[2:]
    train.main(cmd_argv)


def main():
    multiprocessing.set_start_method("spawn", force=True)

    processes = []
    for process_idx in range(NUM_PROCESSES):
        p = multiprocessing.Process(target=worker_fn, args=(process_idx, NUM_PROCESSES))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()