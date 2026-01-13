from typing import Any
from dataclasses import dataclass, field


# ---- Constants common across many experiments ---- #

# Default env vars to set before running an experiment.
ENV = {
    "XLA_FLAGS": "--xla_disable_hlo_passes=rematerialization --xla_gpu_autotune_level=0",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    "NVTE_FUSED_ATTN": "1",
    "PYTHONUNBUFFERED": "1",
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
    ici_data_parallelism=1 \
    dcn_data_parallelism=1 \
    ici_tensor_parallelism=2 \
    dcn_tensor_parallelism=1 \
    ici_pipeline_parallelism=4 \
    dcn_pipeline_parallelism=1 \
    remat_policy=minimal_with_context \
    gradient_clipping_threshold=0 \
    attention=cudnn_flash_te \
    use_mmpp=true \
    mmpp_final_layer_remat_policy=no_remat \
    mmpp_print_memory_usage=false"""


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

EXPERIMENTS = [
    # # ---- SPMD single-process experiments ---- #
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_1",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 4,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=False,
    # ),
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_2",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 2,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=False,
    # ),
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_4",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 1,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=False,
    # ),
    # # ---- SPMD multi-process (4) experiments ---- #
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
    # ---- SPMD multi-process (8) experiments ---- #
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
    # ExperimentConfig(
    #     name="mmpp-spmd-mifc_256-num_repeats_2-multiprocess_8",
    #     overrides={
    #         "num_layers_per_pipeline_stage": 2,
    #         "use_mmpp": False,
    #     },
    #     is_multiprocess=True,
    #     multiprocess_kwargs={
    #         "num_processes": 8,
    #         "devices_per_process": 1,
    #     },
    # ),
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

# Single process MPMD mubatch sweeps.
# for schedule in ["gpipe", "1F1B"]:
#     for num_mubatches in [4, 8, 16]:
#         for final_layer_remat_policy in ["no_remat", "save_logits_only"]:
#             EXPERIMENTS.append(
#                 ExperimentConfig(
#                     name=f"mpmd-{schedule}-num_repeats=1-num_mubatches={num_mubatches}-final_layer_remat={final_layer_remat_policy}",
#                     overrides={
#                         "num_layers_per_pipeline_stage": 4,
#                         "use_mmpp": True,
#                         "mmpp_schedule": schedule,
#                         "mmpp_print_memory_usage": True,
#                         "mmpp_final_layer_remat_policy": final_layer_remat_policy,
#                         "per_device_batch_size": 2 * (num_mubatches // 4),
#                         "num_pipeline_microbatches": num_mubatches,
#                         "run_name": f"mpmd-{schedule}-num_repeats_1-num_mubatches_{num_mubatches}",
#                         "profiler": "xplane",
#                         "steps": 20,
#                         "skip_first_n_steps_for_profiler": 10,
#                         "profiler_steps": 9,
#                     },
#                     is_multiprocess=False,
#                 )
#             )

# # Single process MPMD.
# for schedule in ["gpipe", "1F1B"]:
#     for num_repeats in [1, 2, 4]:
#         EXPERIMENTS.append(
#             ExperimentConfig(
#                 name=f"mpmd-{schedule}-num_repeats_{num_repeats}",
#                 overrides={
#                     "num_layers_per_pipeline_stage": 4 // num_repeats,
#                     "use_mmpp": True,
#                     "mmpp_schedule": schedule,
#                 },
#                 is_multiprocess=False,
#             )
#         )

# # Multi-process MPMD.
for num_processes in [4, 8]:
    for schedule in ["gpipe", "1F1B"]:
        for num_repeats in [1, 2, 4]:
            if not (num_processes == 8 and schedule == "1F1B" and num_repeats == 1):
                continue

            EXPERIMENTS.append(
                ExperimentConfig(
                    name=f"mpmd-{schedule}-multiprocess=8-num_repeats=1-num_mubatches=4-final_layer_remat=save_logits_only",
                    overrides={
                        "num_layers_per_pipeline_stage": 8 // num_repeats,
                        "use_mmpp": True,
                        "mmpp_schedule": schedule,
                        "mmpp_print_memory_usage": False,
                        "mmpp_final_layer_remat_policy": "save_logits_only",
                    },
                    is_multiprocess=True,
                    multiprocess_kwargs={
                        "num_processes": num_processes,
                        "devices_per_process": 8 // num_processes,
                    },
                )
            )

# # Single process SPMD.
# for num_repeats in [1, 2, 4]:
#     EXPERIMENTS.append(
#         ExperimentConfig(
#             name=f"spmd-num_repeats_{num_repeats}",
#             overrides={
#                 "num_layers_per_pipeline_stage": 4 // num_repeats,
#                 "use_mmpp": False,
#             },
#             is_multiprocess=False,
#         )
#     )

# # Multi-process SPMD.
# for num_processes in [4, 8]:
#     for num_repeats in [1, 2, 4]:
#         EXPERIMENTS.append(
#             ExperimentConfig(
#                 name=f"spmd-num_repeats_{num_repeats}-multiprocess_{num_processes}",
#                 overrides={
#                     "num_layers_per_pipeline_stage": 4 // num_repeats,
#                     "use_mmpp": False,
#                 },
#                 is_multiprocess=True,
#                 multiprocess_kwargs={
#                     "num_processes": num_processes,
#                     "devices_per_process": 8 // num_processes,
#                 },
#             )
#         )