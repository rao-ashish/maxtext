from typing import Any
from dataclasses import dataclass, field


# ---- Constants common across many experiments ---- #

# Default env vars to set before running an experiment.
ENV = {
    "XLA_FLAGS": "--xla_disable_hlo_passes=rematerialization",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    "NVTE_FUSED_ATTN": "1",
}

# Default base command.
BASE_COMMAND = """python3 -m MaxText.train maxtext/configs/base.yml \
    run_name=logdir \
    model_name=llama2-7b \
    steps=10 \
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
    ici_data_parallelism=1 \
    dcn_data_parallelism=1 \
    ici_tensor_parallelism=2 \
    dcn_tensor_parallelism=1 \
    ici_pipeline_parallelism=4 \
    dcn_pipeline_parallelism=1 \
    remat_policy=minimal_with_context \
    gradient_clipping_threshold=0 \
    attention=cudnn_flash_te \
    use_mpmd_pp=true \
    mpmd_pp_optimize_for_compile_times=true \
    mpmd_pp_schedule=gpipe \
    mpmd_pp_final_layer_remat_policy=save_logits_only \
    max_segments_per_seq=32 \
    scan_layers=false"""


# ---- Main configuration dataclass ---- #


# Dataclasss bundling info for a single experiment.
@dataclass
class ExperimentConfig:
    name: str  # Name of the experiment.
    overrides: dict[str, Any]  # Flags to add to the base command.
    is_multiprocess: bool = False  # Whether to run with multiprocess.
    multiprocess_kwargs: dict | None = None  # Extra kwargs for multiprocess.
    base_command: str = BASE_COMMAND  # Command to run (before overrides).
    # Env vars for this experiment.
    env: dict[str, str] = field(default_factory=lambda: ENV)


# ---- Configured experiments ---- #

EXPERIMENTS = []

# ---- Memory usage sweeps ---- #

# Single process MPMD.
for schedule in ["gpipe", "1F1B"]:
    for num_repeats in [1, 2, 4]:
        EXPERIMENTS.append(
            ExperimentConfig(
                name=(
                    f"MEMORY-mpmd-{schedule}-num_repeats={num_repeats}"
                    "-num_mubatches=4-final_layer_remat=save_logits_only"
                ),
                overrides={
                    "num_layers_per_pipeline_stage": 8 // num_repeats,
                    "num_pipeline_microbatches": 4,
                    "use_mpmd_pp": True,
                    "ici_data_parallelism": 2,
                    "ici_tensor_parallelism": 1,
                    "mpmd_pp_schedule": schedule,
                    "mpmd_pp_optimize_for_compile_times": False,
                    "mpmd_pp_final_layer_remat_policy": "save_logits_only",
                    "mpmd_pp_print_memory_usage": True,
                    "profiler": "nsys",
                    "scan_layers": False,
                    "shard_mode": "auto",
                },
                is_multiprocess=False,
            )
        )

# Multi-process MPMD.
for num_processes in [4, 8]:
    for schedule in ["gpipe", "1F1B"]:
        for num_repeats in [1, 2, 4]:
            EXPERIMENTS.append(
                ExperimentConfig(
                    name=(
                        f"MEMORY-mpmd-{schedule}-multiprocess={num_processes}"
                        f"-num_repeats={num_repeats}"
                        "-num_mubatches=4-final_layer_remat=save_logits_only"
                    ),
                    overrides={
                        "num_layers_per_pipeline_stage": 8 // num_repeats,
                        "num_pipeline_microbatches": 4,
                        "use_mpmd_pp": True,
                        "ici_data_parallelism": 2,
                        "ici_tensor_parallelism": 1,
                        "mpmd_pp_schedule": schedule,
                        "mpmd_pp_optimize_for_compile_times": True,
                        "mpmd_pp_final_layer_remat_policy": "save_logits_only",
                        "mpmd_pp_print_memory_usage": True,
                        "profiler": "nsys",
                        "scan_layers": False,
                        "shard_mode": "auto",
                    },
                    is_multiprocess=True,
                    multiprocess_kwargs={
                        "num_processes": num_processes,
                        "devices_per_process": 8 // num_processes,
                    },
                )
            )


# ---- Performance sweeps ---- #

# Single process MPMD.
for schedule in ["gpipe", "1F1B"]:
    for num_repeats in [1, 2, 4]:
        EXPERIMENTS.append(
            ExperimentConfig(
                name=(
                    f"mpmd-{schedule}-num_repeats={num_repeats}"
                    "-num_mubatches=4-final_layer_remat=save_logits_only"
                ),
                overrides={
                    "num_layers_per_pipeline_stage": 8 // num_repeats,
                    "num_pipeline_microbatches": 4,
                    "use_mpmd_pp": True,
                    "ici_data_parallelism": 2,
                    "ici_tensor_parallelism": 1,
                    "mpmd_pp_schedule": schedule,
                    "mpmd_pp_optimize_for_compile_times": True,
                    "mpmd_pp_final_layer_remat_policy": "save_logits_only",
                    "profiler": "nsys",
                    "scan_layers": False,
                    "shard_mode": "auto",
                },
                is_multiprocess=False,
            )
        )

# Multi-process MPMD.
for num_processes in [4, 8]:
    for schedule in ["gpipe", "1F1B"]:
        for num_repeats in [1, 2, 4]:
            EXPERIMENTS.append(
                ExperimentConfig(
                    name=(
                        f"mpmd-{schedule}-multiprocess={num_processes}"
                        f"-num_repeats={num_repeats}"
                        "-num_mubatches=4-final_layer_remat=save_logits_only"
                    ),
                    overrides={
                        "num_layers_per_pipeline_stage": 8 // num_repeats,
                        "num_pipeline_microbatches": 4,
                        "use_mpmd_pp": True,
                        "ici_data_parallelism": 2,
                        "ici_tensor_parallelism": 1,
                        "mpmd_pp_schedule": schedule,
                        "mpmd_pp_optimize_for_compile_times": True,
                        "mpmd_pp_final_layer_remat_policy": "save_logits_only",
                        "profiler": "nsys",
                        "scan_layers": False,
                        "shard_mode": "auto",
                    },
                    is_multiprocess=True,
                    multiprocess_kwargs={
                        "num_processes": num_processes,
                        "devices_per_process": 8 // num_processes,
                    },
                )
            )
