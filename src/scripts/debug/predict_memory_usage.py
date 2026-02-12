"""Memory usage prediction utilities for MPMD pipeline schedules."""

import pandas as pd

from MaxText.mpmd_pp.schedules import PipelineSchedule, SectionKind, make_gpipe_schedule


def activation_bytes(cfg):
    return (
        cfg["low_precision_bytes"]
        * cfg["per_device_microbatch_size"]
        * cfg["seq_len"]
        * cfg["model_dim"]
    )


def mlp_hidden_bytes(cfg):
    return (
        cfg["low_precision_bytes"]
        * cfg["per_device_microbatch_size"]
        * cfg["seq_len"]
        * cfg["mlp_dim"]
    )


def logit_bytes(cfg):
    return (
        cfg["high_precision_bytes"]
        * cfg["per_device_microbatch_size"]
        * cfg["seq_len"]
        * cfg["vocab_size"]
    )


def input_tokens_bytes(cfg):
    return 4 * cfg["seq_len"] * cfg["per_device_microbatch_size"]


def aggregate_bytes(cfg, bytes_callable_dict):
    return sum(v(cfg) for _, v in bytes_callable_dict.items())


embedding_layer_remat_policies = {
    "default": {
        "input_tokens": input_tokens_bytes,
    }
}

decoder_block_remat_policies = {
    "minimal_with_context": {
        "query_proj": activation_bytes,
        "key_proj": activation_bytes,
        "value_proj": activation_bytes,
        "softmax_aux": (
            lambda cfg: 0  # TODO.
        ),
        "pre_out_proj": activation_bytes,
        "out_proj": activation_bytes,
        "mlpwi_0": mlp_hidden_bytes,
        "mlpwi_1": mlp_hidden_bytes,
        "inputs_segmentation": input_tokens_bytes,
        "inputs_position": input_tokens_bytes,
        "input_residuals": activation_bytes,
    }
}


final_layer_remat_policies = {
    "no_remat": {
        "lm_head_inputs": activation_bytes,
        "logits": logit_bytes,
    },
    "save_logits_only": {"logits": logit_bytes},
    "full_remat": {},
}



def get_model_params_bytes(cfg, logical_stage_idx, high_precision=False):
    pre_self_attn_ln = cfg["model_dim"]
    qkvo_proj_weights = 4 * cfg["model_dim"] * cfg["model_dim"]
    post_self_attn_ln = cfg["model_dim"]
    mlpwi_0 = mlpwi_1 = mlpwo = cfg["mlp_dim"] * cfg["model_dim"]
    decoder_block = (
        pre_self_attn_ln
        + qkvo_proj_weights
        + post_self_attn_ln
        + mlpwi_0
        + mlpwi_1
        + mlpwo
    )

    blocks_per_logical_stage = cfg["num_layers"] // cfg["num_logical_stages"]
    out = blocks_per_logical_stage * decoder_block

    if logical_stage_idx == 0:
        embedding_table = cfg["vocab_size"] * cfg["model_dim"]
        out += embedding_table

    if logical_stage_idx == cfg["num_logical_stages"] - 1:
        pre_lm_head_ln = cfg["model_dim"]
        lm_head = cfg["vocab_size"] * cfg["model_dim"]
        final_layer = pre_lm_head_ln + lm_head
        out += final_layer

    precision_coeff = (
        cfg["high_precision_bytes"] if high_precision else cfg["low_precision_bytes"]
    )

    return precision_coeff * out


def get_initial_bytes(cfg, logical_stage_idx):
    """Memory present before any schedule task executes.

    This includes fp32 parameters and optimizer state (mu + nu).
    bf16 params and gradient accumulators are created during STAGE_INIT.
    """

    fp32_params = get_model_params_bytes(cfg, logical_stage_idx, high_precision=True)
    opt_state_mu = fp32_params
    opt_state_nu = fp32_params
    return fp32_params + opt_state_mu + opt_state_nu


def get_init_task_delta_bytes(cfg, logical_stage_idx):
    """Additional memory allocated by STAGE_INIT."""

    bf16_params = get_model_params_bytes(cfg, logical_stage_idx, high_precision=False)
    grads_acc = get_model_params_bytes(cfg, logical_stage_idx, high_precision=True)
    return bf16_params + grads_acc


def get_update_cleanup_delta_bytes(cfg, logical_stage_idx):
    """Memory released by UPDATE / FUSED_BACKWARD_UPDATE for a stage."""

    return -get_init_task_delta_bytes(cfg, logical_stage_idx)


def get_stashed_bytes(
    cfg,
    logical_stage_idx,
    embedding_layer_remat_policy,
    decoder_block_remat_policy,
    final_layer_remat_policy,
):
    blocks_per_logical_stage = cfg["num_layers"] // cfg["num_logical_stages"]
    decoder_block = aggregate_bytes(cfg, decoder_block_remat_policy)
    out = blocks_per_logical_stage * decoder_block
    if logical_stage_idx == 0:
        out += aggregate_bytes(cfg, embedding_layer_remat_policy)
    if logical_stage_idx == cfg["num_logical_stages"] - 1:
        out += aggregate_bytes(cfg, final_layer_remat_policy)
    return out


def execute_task(
    cfg,
    task,
    curr_memory_usages,
    embedding_layer_remat_policy,
    decoder_block_remat_policy,
    final_layer_remat_policy,
):
    _, (section_kind, logical_stage_idx) = task
    physical_stage_idx = logical_stage_idx % cfg["num_physical_stages"]

    if section_kind == SectionKind.STAGE_INIT:
        bytes_delta = get_init_task_delta_bytes(cfg, logical_stage_idx)
    elif section_kind == SectionKind.FORWARD:
        bytes_delta = get_stashed_bytes(
            cfg,
            logical_stage_idx,
            embedding_layer_remat_policy,
            decoder_block_remat_policy,
            final_layer_remat_policy,
        )
    elif section_kind == SectionKind.BACKWARD:
        bytes_delta = -get_stashed_bytes(
            cfg,
            logical_stage_idx,
            embedding_layer_remat_policy,
            decoder_block_remat_policy,
            final_layer_remat_policy,
        )
    elif section_kind == SectionKind.FUSED_BACKWARD_UPDATE:
        bytes_delta = (
            -get_stashed_bytes(
                cfg,
                logical_stage_idx,
                embedding_layer_remat_policy,
                decoder_block_remat_policy,
                final_layer_remat_policy,
            )
            + get_update_cleanup_delta_bytes(cfg, logical_stage_idx)
        )
    elif section_kind == SectionKind.UPDATE:
        bytes_delta = get_update_cleanup_delta_bytes(cfg, logical_stage_idx)
    else:
        raise ValueError(f"Unsupported SectionKind in task: {section_kind}")

    return tuple(
        b if out_physical_idx != physical_stage_idx else b + bytes_delta
        for out_physical_idx, b in enumerate(curr_memory_usages)
    )


def predict_memory_usage(
    cfg: dict,
    pipeline_schedule: PipelineSchedule,
) -> pd.DataFrame:
    """Given a pipeline schedule, return predicted memory usage by stage.

    The first element of the output gives expected memory usage before any task
    has executed.
    """

    embedding_layer_remat_policy = embedding_layer_remat_policies[
        cfg["embedding_layer_remat_policy"]
    ]
    decoder_block_remat_policy = decoder_block_remat_policies[
        cfg["decoder_block_remat_policy"]
    ]
    final_layer_remat_policy = final_layer_remat_policies[
        cfg["final_layer_remat_policy"]
    ]

    initial_memory_usage = [0 for _ in range(cfg["num_physical_stages"])]
    for logical_stage_idx in range(cfg["num_logical_stages"]):
        physical_stage_idx = logical_stage_idx % cfg["num_physical_stages"]
        initial_memory_usage[physical_stage_idx] += get_initial_bytes(
            cfg, logical_stage_idx
        )

    memory_usages = [tuple(initial_memory_usage)]

    for task in pipeline_schedule:
        memory_usages.append(
            execute_task(
                cfg,
                task,
                memory_usages[-1],
                embedding_layer_remat_policy,
                decoder_block_remat_policy,
                final_layer_remat_policy,
            )
        )

    tasks_with_initial = [
        (None, (None, -1)),
        *list(pipeline_schedule),
    ]

    df_data = []
    for task, memory_usage in zip(tasks_with_initial, memory_usages):
        microbatch_idx, section_name = task
        section_kind, logical_stage_idx = section_name
        section_kind_str = section_kind.name if section_kind is not None else "INITIAL"
        is_bwd = section_kind in {
            SectionKind.BACKWARD,
            SectionKind.FUSED_BACKWARD_UPDATE,
        }

        df_data.append(
            (
                microbatch_idx,
                logical_stage_idx,
                section_kind_str,
                is_bwd,
                *(b / 1e9 for b in memory_usage),
            )
        )

    df_cols = [
        "microbatch_idx",
        "logical_stage_idx",
        "section_kind",
        "is_bwd",
        *(f"physical_stage_{i}_gb" for i in range(cfg["num_physical_stages"])),
    ]

    return pd.DataFrame(df_data, columns=df_cols)


def main():
    pretraining_job_config = {
        "vocab_size": 32_000,
        "seq_len": 2048,
        "model_dim": 4096,
        "mlp_dim": 11008,
        "num_layers": 16,
        "num_logical_stages": 4,
        "num_physical_stages": 4,
        "num_attention_heads": 32,
        "num_mubatches": 4,
        "per_device_microbatch_size": 2,
        "low_precision_bytes": 2,
        "high_precision_bytes": 4,
        "embedding_layer_remat_policy": "default",
        "decoder_block_remat_policy": "minimal_with_context",
        "final_layer_remat_policy": "no_remat",
    }

    schedule = make_gpipe_schedule(
        pretraining_job_config["num_mubatches"],
        pretraining_job_config["num_logical_stages"],
    )

    memory_usage_predictions = predict_memory_usage(
        pretraining_job_config,
        schedule,
    )

    print(memory_usage_predictions)


if __name__ == "__main__":
    main()
