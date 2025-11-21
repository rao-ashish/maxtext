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
BASE_COMMAND = """python3 -m MaxText.train MaxText/configs/base.yml \
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
    ici_data_parallelism=2 \
    dcn_data_parallelism=1 \
    ici_tensor_parallelism=1 \
    dcn_tensor_parallelism=1 \
    ici_pipeline_parallelism=4 \
    dcn_pipeline_parallelism=1 \
    remat_policy=minimal_with_context \
    gradient_clipping_threshold=0 \
    attention=cudnn_flash_te"""


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


# ---- Configured experiments. ---- #

EXPERIMENTS = [
    # ---- SPMD multi-process (4) experiments ---- #
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_1-multiprocess_4",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 4,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 4,
    #         "devices_per_process": 2,
    #     },
    # ),
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_2-multiprocess_4",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 2,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 4,
    #         "devices_per_process": 2,
    #     },
    # ),
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_4-multiprocess_4",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 1,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 4,
    #         "devices_per_process": 2,
    #     },
    # ),
    # ---- MPMD multi-process (8) experiments ---- #
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_1-multiprocess_8",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 4,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 8,
    #         "devices_per_process": 1,
    #     },
    # ),
    ExperimentConfig(
        name="spmd-mifc_256-num_repeats_2-multiprocess_8",
        overrides={
            "num_layers_per_pipeline_stage": 2,
            # "use_mmpp": False,
        },
        is_multiprocess=True,
        multiprocess_kwargs={
            "num_processes": 8,
            "devices_per_process": 1,
        },
    ),
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_4-multiprocess_8",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 1,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 8,
    #         "devices_per_process": 1,
    #     },
    # ),
    # # ---- MPMD single-process experiments ---- #
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_1",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 4,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=False,
    # ),
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_2",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 2,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=False,
    # ),
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_4",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 1,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=False,
    # ),
    # # ---- MPMD multi-process (4) experiments ---- #
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_1-multiprocess_4",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 4,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 4,
    #         "devices_per_process": 2,
    #     },
    # ),
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_2-multiprocess_4",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 2,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 4,
    #         "devices_per_process": 2,
    #     },
    # ),
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_4-multiprocess_4",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 1,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 4,
    #         "devices_per_process": 2,
    #     },
    # ),
    # # ---- MPMD multi-process (8) experiments ---- #
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_1-multiprocess_8",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 4,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 8,
    #         "devices_per_process": 1,
    #     },
    # ),
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_2-multiprocess_8",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 2,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 8,
    #         "devices_per_process": 1,
    #     },
    # ),
    # ExperimentConfig(
    #     name="mmpp-mpmd-mifc_256-num_repeats_4-multiprocess_8",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 1,
    #         "use_mmpp": True,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 8,
    #         "devices_per_process": 1,
    #     },
    # ),
]
