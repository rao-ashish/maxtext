#!/bin/bash

# ---- Environment Setup ---- #

cd /opt/maxtext || exit 1

# Performance Flags.
export NCCL_IB_SL=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
export CUDA_DEVICE_MAX_CONNECTIONS=16
export NCCL_MNNVL_ENABLE=1
export NVTE_FUSED_ATTN=1

# Not very well tuned.
BASE_XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
 --xla_disable_hlo_passes=rematerialization \
 --xla_gpu_autotune_level=5 \
 --xla_gpu_enable_nccl_comm_splitting=true \
 --xla_gpu_enable_pipelined_all_gather=true \
 --xla_gpu_enable_pipelined_reduce_scatter=true \
 --xla_gpu_enable_while_loop_double_buffering=true"

export XLA_FLAGS="$XLA_FLAGS $BASE_XLA_FLAGS"

# Needed for 1 process / node.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "Node $SLURMD_NODENAME: Starting Python Training on Rank $SLURM_PROCID..."
echo "Node $SLURMD_NODENAME (Rank $SLURM_PROCID): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PROFILE_DIR="TODO"
PROFILE_BASENAME="llama2_70b_1F1B_job${SLURM_JOB_ID:-nojid}_node${SLURMD_NODENAME:-nonode}_rank${SLURM_PROCID:-norank}"
PROFILE_PATH="${PROFILE_DIR}/${PROFILE_BASENAME}"

PROFILE_CMD="nsys profile --output ${PROFILE_PATH} --cpuctxsw=none --trace=cublas,cuda,cudnn,cusolver,nvtx,osrt,python-gil --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --python-sampling=true"


# --- Run MaxText ---- #

$PROFILE_CMD python -m MaxText.train src/maxtext/configs/base.yml \
    run_name=llama2_70b-1F1B \
    model_name=llama2-70b \
    steps=20 \
    enable_checkpointing=false \
    base_output_directory=logs \
    dataset_path=local \
    dataset_type=synthetic \
    hardware=gpu_multiprocess \
    ici_fsdp_parallelism=1 \
    dcn_fsdp_parallelism=1 \
    ici_data_parallelism=1 \
    dcn_data_parallelism=1 \
    ici_tensor_parallelism=8 \
    dcn_tensor_parallelism=1 \
    ici_pipeline_parallelism=1 \
    dcn_pipeline_parallelism=4 \
    remat_policy=full \
    gradient_clipping_threshold=0 \
    attention=cudnn_flash_te \
    use_mpmd_pp=true \
    num_layers_per_pipeline_stage=4 \
    mpmd_pp_schedule=1F1B \
    mpmd_pp_final_layer_remat_policy=full_remat \
    mpmd_pp_print_memory_usage=false \
    mpmd_pp_dump_section_fn_debug_info=false \
    profiler=nsys \
    upload_all_profiler_results=true \
    skip_first_n_steps_for_profiler=10 \
    profiler_steps=9 \
    max_segments_per_seq=32 \
    max_target_length=4096 \
    per_device_batch_size=1 \
    num_pipeline_microbatches=16 \
    scan_layers=false \
    scan_layers_per_stage=false \
    --alsologtostderr \
    --stderrthreshold=info
