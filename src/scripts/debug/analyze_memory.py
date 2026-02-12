"""Script to plot and compare predicted vs actual memory usage for MPMD PP
MaxText runs that used mpmd_pp_log_memory_usage = True."""

import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from predict_memory_usage import predict_memory_usage
from MaxText.mpmd_pp.schedules import make_pipeline_schedule


# ==== Log data loading logic ==== #


# Where log files we need to parse are stored.
LOG_DIR = "scripts/sweeps/outputs/memory-sweeps"

# Where to save memory usage plots.
PLOT_SAVE_DIR_ACTUAL_VS_PREDICTED = "scripts/debug/plots/actual_vs_predicted"
PLOT_SAVE_DIR_1F1B_VS_GPIPE = "scripts/debug/plots/1F1B_vs_gpipe"

# We will analyze memory usage in the region of the log between START_MARKER
# and END_MARKER (exclusive of boundaries). This is only supported for single
# process runs, because for multiprocess runs, only the last process will print
# the 'loss' log lines at the end of each train step that can be usefully used
# as start/end markers.
START_MARKER = None
END_MARKER = None

# Llama2 7B config information so that we can predict memory usage for this
# model. TODO:
# - Parse this directly from the log files instead of hardcoding.
# - Support num_kv_heads != num_query_heads (this is the case for 70B).
BASE_CFG_LLAMA2_7B = {
    "vocab_size": 32_000,
    "seq_len": 2048,
    "model_dim": 4096,
    "mlp_dim": 11008,
    "num_layers": 32,
    "num_devices": 8,
    "num_physical_stages": 4,
    "num_attention_heads": 32,
    "per_device_microbatch_size": 2,
    "low_precision_bytes": 2,
    "high_precision_bytes": 4,
    "tp_factor": 1,
    "embedding_layer_remat_policy": "default",
    "decoder_block_remat_policy": "minimal_with_context",
    "final_layer_remat_policy": "save_logits_only",
}


@dataclass(frozen=True)
class RunInfo:
    """Dataclass holding metadata related to a specific run."""

    source_filename: str
    impl_type: str
    num_processes: int
    schedule_name: str
    num_repeats: int
    num_mubatches: int
    final_layer_remat_policy: str

    def __str__(self):
        return self.source_filename


def parse_log_filename(filename: str) -> RunInfo:
    """Parse a log's filename into a RunInfo object holding metadata about the
    run.

    TODO: Parsing the filename is brittle, we should parse the config dumped
    into the log file itself.
    """

    match = re.search(
        r".*(mpmd|spmd)-(1F1B|gpipe)-"
        r"(?:(?:multiprocess|num_processes)=(\d+)-)?"
        r"num_repeats=(\d+)-num_mubatches=(\d+)-"
        r"final_layer_remat=([^-/.]+)",
        filename,
    )
    if not match:
        raise ValueError(f"Could not parse run info from filename: {filename}")

    (
        impl_type,
        schedule_name,
        num_processes,
        num_repeats,
        num_mubatches,
        final_layer_remat,
    ) = match.groups()
    parsed_num_processes = int(num_processes) if num_processes is not None else 1
    num_processes = parsed_num_processes if parsed_num_processes > 1 else 1

    return RunInfo(
        source_filename=filename,
        impl_type=impl_type,
        num_processes=num_processes,
        schedule_name=schedule_name,
        num_repeats=int(num_repeats),
        num_mubatches=int(num_mubatches),
        final_layer_remat_policy=final_layer_remat,
    )


def parse_actual_memory_usage_data(
    filepath: str,
    start_marker=START_MARKER,
    end_marker=END_MARKER,
) -> pd.DataFrame:
    """Parse the contents of a log file appearing between start_marker and
    end_marker into a DataFrame of memory usage over time."""

    # Read all lines of the log.
    with open(filepath, "r") as f:
        lines = f.readlines()

    all_data = []
    column_names = None
    start_marker_seen = start_marker is None

    # Parse log lines.
    for line in lines:
        # Skip forward until we see start_marker.
        if not (start_marker_seen := start_marker_seen or start_marker in line):
            continue

        # Stop parsing the log if we see the end marker.
        if end_marker is not None and end_marker in line:
            break

        # Do not parse log lines that are not MEM logging.
        if "MEM" not in line:
            continue

        # Set column names if this is the special 'MEM,name' log line.
        if "MEM,name" in line:
            if column_names is None:
                column_names = line.strip().split(",")[1:]
            continue

        # Process the logged memory usage data at the current line into a list
        # that looks like
        # [task_name, float_data_entry_0, float_data_entry_1, ...]
        curr_data = line.split(",")[1:]
        for idx in range(1, len(curr_data)):
            curr_data[idx] = float(curr_data[idx])

        all_data.append(curr_data)

    # Make sure we parsed the column names.
    assert column_names is not None

    # Form the output DataFrame.
    out = pd.DataFrame(all_data, columns=column_names)
    if "state_total_gb" not in out.columns:
        out["state_total_gb"] = out.filter(like="state").sum(axis=1)

    # Parse task names to fill out other columns of the DataFrame.
    task_regex = re.compile(
        r"Task\(mu=(?P<microbatch>None|\d+),\s*s=(?P<logical_stage>\d+),\s*"
        r"k=SectionKind\.(?P<section_kind>[A-Z_]+)\)"
    )

    out["microbatch_idx"] = pd.NA
    out["logical_stage_idx"] = pd.NA
    out["section_kind"] = pd.NA
    out["is_bwd"] = False

    for idx, name in out["name"].items():
        normalized_name = str(name).strip()

        if normalized_name == "start":
            out.at[idx, "microbatch_idx"] = pd.NA
            out.at[idx, "logical_stage_idx"] = -1
            out.at[idx, "section_kind"] = "INITIAL"
            out.at[idx, "is_bwd"] = False
            continue

        match = task_regex.search(normalized_name)
        if not match:
            continue

        parsed = match.groupdict()
        microbatch = parsed["microbatch"]
        section_kind = parsed["section_kind"]

        out.at[idx, "microbatch_idx"] = (
            pd.NA if microbatch == "None" else int(microbatch)
        )
        out.at[idx, "logical_stage_idx"] = int(parsed["logical_stage"])
        out.at[idx, "section_kind"] = section_kind
        out.at[idx, "is_bwd"] = section_kind in {"BACKWARD", "FUSED_BACKWARD_UPDATE"}

    return out


ParsedDataDict = dict[RunInfo, dict[str, Any]]


def _get_pipeline_schedule(cfg: dict[str, Any]):
    return make_pipeline_schedule(
        num_logical_stages=cfg["num_logical_stages"],
        num_physical_stages=cfg["num_physical_stages"],
        num_microbatches=cfg["num_mubatches"],
        schedule_name=cfg["schedule_name"],
    )


def get_complete_cfg(base_cfg: dict[str, Any], run_info: RunInfo):
    """Create a copy of base_cfg where config entries have been overriden using
    the data parsed from the log filename (given as run_info.)"""

    complete_cfg = base_cfg.copy()
    complete_cfg["num_logical_stages"] = (
        run_info.num_repeats * complete_cfg["num_physical_stages"]
    )
    assert complete_cfg["num_layers"] % complete_cfg["num_logical_stages"] == 0

    complete_cfg["num_mubatches"] = run_info.num_mubatches
    complete_cfg["final_layer_remat_policy"] = run_info.final_layer_remat_policy
    complete_cfg["schedule_name"] = run_info.schedule_name

    return complete_cfg


def load_data(base_cfg: dict[str, Any], log_dir: str) -> ParsedDataDict:
    """Parse all logs in a given log_dir into a dictionary mapping RunInfo to
    its launch config, actual memory usage, and predicted memory usage.
    """

    out = {}

    for name in os.listdir(log_dir):
        run_info = parse_log_filename(name)
        cfg = get_complete_cfg(base_cfg, run_info)

        path = os.path.join(log_dir, name)

        # process_paths allows us to handle multiprocess logs.
        process_paths = [path]
        if os.path.isdir(path):
            process_paths = [
                os.path.join(path, process_filename)
                for process_filename in sorted(os.listdir(path))
            ]

        # Predict memory usage for the pipeline schedule that was run.
        predicted_memory_usage = predict_memory_usage(
            cfg=cfg,
            pipeline_schedule=_get_pipeline_schedule(cfg),
        )

        # Parse log for actual memory usage (if case is for multiprocess runs,
        # else case handles single process runs).
        if len(process_paths) > 1:
            actual_memory_usage = pd.concat(
                [
                    parse_actual_memory_usage_data(
                        process_path,
                        start_marker=START_MARKER,
                        end_marker=END_MARKER,
                    )
                    for process_path in process_paths
                ],
                axis=1,
            )
        else:
            actual_memory_usage = parse_actual_memory_usage_data(path)

        # Add to the output dict.
        out[run_info] = {
            "actual_memory_usage": actual_memory_usage,
            "predicted_memory_usage": predicted_memory_usage,
            "cfg": cfg,
        }

    return out


def _align_predicted_to_actual(
    predicted_memory_usage: pd.DataFrame,
    actual_memory_usage: pd.DataFrame,
) -> pd.DataFrame:
    """The predicted memory usage is for just one train step, whereas the
    parsed actual memory usage might span multiple train steps. This function
    'tiles' the predicted memory usage to match the actual memory usage's
    length.

    Returns: 'Tiled' predicted memory usage.
    """

    predicted_memory_usage = predicted_memory_usage.reset_index(drop=True)

    if (
        len(predicted_memory_usage) == 0
        or len(actual_memory_usage) % len(predicted_memory_usage) != 0
    ):
        raise ValueError(
            "Unable to tile predicted memory usage for "
            f"{len(predict_memory_usage)=} {len(actual_memory_usage)=}"
        )

    num_reps = len(actual_memory_usage) // len(predicted_memory_usage)
    if num_reps > 1:
        predicted_memory_usage = pd.concat(
            [predicted_memory_usage] * num_reps,
            ignore_index=True,
        )
    return predicted_memory_usage


# ==== Plotting and analysis logic ==== #


def compare_memory_usage_estimates(
    parsed_data_dict: ParsedDataDict,
    left_estimate: str = "used",  # ["used", "known_state", "predicted"]
    right_estimate: str = "predicted",  # ["used", "known_state", "predicted"]
):
    """Print DataFrames comparing the memory usage estimates specified by
    left_estimate and right_estimate."""

    for run_info, parsed_data in parsed_data_dict.items():
        cfg = parsed_data["cfg"]
        num_devices_per_physical_stage = (
            cfg["num_devices"] // cfg["num_physical_stages"]
        )
        estimate_name_to_col_name_fn = {
            "used": lambda device_id: f"device{device_id}_used_gb",
            "known_state": lambda device_id: f"device{device_id}_known_state_gb",
            "predicted": lambda device_id: "physical_stage_{}_gb".format(
                device_id // num_devices_per_physical_stage
            ),
        }

        assert left_estimate in estimate_name_to_col_name_fn
        assert right_estimate in estimate_name_to_col_name_fn

        print(f"---- Used vs predicted for {str(run_info)} ----")

        actual_memory_usage = parsed_data["actual_memory_usage"]
        predicted_memory_usage = parsed_data["predicted_memory_usage"]

        if len(actual_memory_usage) == 0 or len(predicted_memory_usage) == 0:
            print(
                f"SKIPPING {str(run_info)} as it does not have one of the "
                + "required estimate dataframes."
            )
            continue

        predicted_memory_usage = _align_predicted_to_actual(
            predicted_memory_usage,
            actual_memory_usage,
        )

        combined_memory_usage = pd.concat(
            [actual_memory_usage, predicted_memory_usage], axis=1
        )

        for device_id in range(cfg["num_devices"]):
            left_col = estimate_name_to_col_name_fn[left_estimate](device_id)
            right_col = estimate_name_to_col_name_fn[right_estimate](device_id)

            if (
                left_col not in combined_memory_usage
                or right_col not in combined_memory_usage
            ):
                print(
                    f"SKIPPING device_id_{device_id} as it does not have one "
                    + "of the required estimates."
                )
                continue

            left_gb = combined_memory_usage[left_col]
            right_gb = combined_memory_usage[right_col]

            new_df = pd.DataFrame(
                {
                    "microbatch_idx": predicted_memory_usage["microbatch_idx"],
                    "logical_stage_idx": predicted_memory_usage["logical_stage_idx"],
                    "section_kind": predicted_memory_usage["section_kind"],
                    "is_bwd": predicted_memory_usage["is_bwd"],
                    left_estimate: left_gb.values,
                    right_estimate: right_gb.values,
                    f"{left_estimate}_minus_{right_estimate}": (
                        left_gb - right_gb
                    ).values,
                }
            )

            print(f"\n[Device {device_id}] Memory Usage Comparison, {str(run_info)}:")
            print(new_df.to_string(index=False))
            print("-" * 40)


def max_deviation_of_usage_estimates(
    parsed_data_dict: ParsedDataDict,
    left_estimate: str = "used",  # ["used", "known_state", "predicted"]
    right_estimate: str = "predicted",  # ["used", "known_state", "predicted"]
):
    """Prints information about the maximum deviation in GB between predicted
    actual memory usage for every log and device in parsed_data_dict."""

    for run_info, parsed_data in parsed_data_dict.items():
        cfg = parsed_data["cfg"]
        num_devices_per_physical_stage = (
            cfg["num_devices"] // cfg["num_physical_stages"]
        )
        estimate_name_to_col_name_fn = {
            "used": lambda device_id: f"device{device_id}_used_gb",
            "known_state": lambda device_id: f"device{device_id}_known_state_gb",
            "predicted": lambda device_id: "physical_stage_{}_gb".format(
                device_id // num_devices_per_physical_stage
            ),
        }

        assert left_estimate in estimate_name_to_col_name_fn
        assert right_estimate in estimate_name_to_col_name_fn

        print(f"---- Estimate Comparison for {str(run_info)} ----")

        actual_memory_usage = parsed_data["actual_memory_usage"]
        predicted_memory_usage = parsed_data["predicted_memory_usage"]

        if len(actual_memory_usage) == 0 or len(predicted_memory_usage) == 0:
            print(
                f"SKIPPING {str(run_info)} as it does not have one of the "
                + "required estimate dataframes."
            )
            continue

        predicted_memory_usage = _align_predicted_to_actual(
            predicted_memory_usage,
            actual_memory_usage,
        )

        combined_memory_usage = pd.concat(
            [actual_memory_usage, predicted_memory_usage], axis=1
        )

        max_abs_dev = -1.0
        max_dev_info = None

        for device_id in range(cfg["num_devices"]):
            left_col = estimate_name_to_col_name_fn[left_estimate](device_id)
            right_col = estimate_name_to_col_name_fn[right_estimate](device_id)

            if (
                left_col not in combined_memory_usage
                or right_col not in combined_memory_usage
            ):
                continue

            left_series = combined_memory_usage[left_col]
            right_series = combined_memory_usage[right_col]
            diff_series = left_series - right_series
            abs_diff_series = diff_series.abs()

            curr_max_idx = abs_diff_series.idxmax()
            curr_max_val = abs_diff_series[curr_max_idx]

            if curr_max_val > max_abs_dev:
                max_abs_dev = curr_max_val
                max_dev_info = {
                    "device_id": device_id,
                    "timestep": curr_max_idx,
                    "diff": diff_series[curr_max_idx],
                    "left_val": left_series[curr_max_idx],
                    "right_val": right_series[curr_max_idx],
                    "microbatch_idx": predicted_memory_usage["microbatch_idx"].iloc[
                        curr_max_idx
                    ],
                    "logical_stage_idx": predicted_memory_usage[
                        "logical_stage_idx"
                    ].iloc[curr_max_idx],
                    "section_kind": predicted_memory_usage["section_kind"].iloc[
                        curr_max_idx
                    ],
                    "is_bwd": predicted_memory_usage["is_bwd"].iloc[curr_max_idx],
                }

        if max_dev_info is not None:
            print(f"Maximum Absolute Deviation: {max_abs_dev:.4f} GB")
            print(f"Device: {max_dev_info['device_id']}")
            print(f"Timestep: {max_dev_info['timestep']}")
            print(
                f"Context: microbatch={max_dev_info['microbatch_idx']}, "
                f"logical_stage={max_dev_info['logical_stage_idx']}, "
                f"section_kind={max_dev_info['section_kind']}, "
                f"is_bwd={max_dev_info['is_bwd']}"
            )
            print(
                f"Values: {left_estimate}={max_dev_info['left_val']:.4f}, "
                f"{right_estimate}={max_dev_info['right_val']:.4f}, "
                f"Diff ({left_estimate} - {right_estimate})={max_dev_info['diff']:.4f}"
            )
        else:
            print("No valid comparison data found across devices.")
        print("-" * 40)


def plot_actual_vs_predicted(
    parsed_data_dict: ParsedDataDict,
    save_dir: str,
):
    """Plots actual vs predicted memory usage for all devices and logs in
    parsed_data_dict. Saves plots to save_dir."""

    os.makedirs(save_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    colors = {
        "used": "red",
        "known_state": "orange",
        "predicted": "blue",
    }

    for run_info, parsed_data in parsed_data_dict.items():
        cfg = parsed_data["cfg"]
        num_devices_per_physical_stage = (
            cfg["num_devices"] // cfg["num_physical_stages"]
        )

        print(f"Plotting used vs predicted for {str(run_info)}")
        curr_dir = os.path.join(save_dir, str(run_info))
        os.makedirs(curr_dir, exist_ok=True)

        schedule_name = run_info.schedule_name
        actual_memory_usage = parsed_data["actual_memory_usage"]

        predicted_memory_usage = parsed_data["predicted_memory_usage"]

        predicted_memory_usage = _align_predicted_to_actual(
            predicted_memory_usage,
            actual_memory_usage,
        )

        for device_id in range(cfg["num_devices"]):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

            physical_stage_idx = device_id // num_devices_per_physical_stage

            predicted_gb = predicted_memory_usage[
                f"physical_stage_{physical_stage_idx}_gb"
            ]

            ax.plot(
                predicted_gb,
                color=colors["predicted"],
                label=f"{schedule_name}-predicted",
            )

            if f"device{device_id}_used_gb" in actual_memory_usage:
                used_gb = actual_memory_usage[f"device{device_id}_used_gb"]

                assert len(used_gb) == len(predicted_gb), (
                    "Mismatch in used and predicted memory data lengths."
                )

                ax.plot(
                    used_gb,
                    color=colors["used"],
                    label=f"{schedule_name}-used",
                )

            ax.set_title(f"Actual vs Predicted Memory Usage, device {device_id}")
            ax.set_xlabel("Schedule Timestep")
            ax.set_ylabel("Memory (GB)")

            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(curr_dir, f"device_id={device_id}.png"))
            plt.close(fig)


def plot_1F1B_vs_gpipe(
    parsed_data_dict: ParsedDataDict,
    save_dir: str,
):
    """Plot a comparison of 1F1B vs GPipe memory usage for all devices and logs
    inside parsed_data_dict. Save the plot to save_dir."""

    schedule_name_to_color = {
        "gpipe": "red",
        "1F1B": "blue",
    }

    # Group parsed_data_dict entries.
    grouped_run_infos = {}
    for run_info in parsed_data_dict:
        run_info_key = (
            run_info.num_processes,
            run_info.num_mubatches,
            run_info.final_layer_remat_policy,
            run_info.num_repeats,
        )
        if run_info_key not in grouped_run_infos:
            grouped_run_infos[run_info_key] = []
        grouped_run_infos[run_info_key].append(run_info)

    # Plotting logic.
    for (
        num_processes,
        num_mubatches,
        final_layer_remat_policy,
        num_repeats,
    ), run_infos in grouped_run_infos.items():
        curr_dir = os.path.join(
            save_dir,
            f"num_mubatches={num_mubatches}-"
            + f"final_layer_remat={final_layer_remat_policy}-"
            + f"num_repeats={num_repeats}-"
            + f"num_processes={num_processes}",
        )
        os.makedirs(curr_dir, exist_ok=True)

        max_num_devices = max(
            parsed_data_dict[run_info]["cfg"]["num_devices"] for run_info in run_infos
        )
        for device_id in range(max_num_devices):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

            for run_info in run_infos:
                actual_memory_usage = parsed_data_dict[run_info]["actual_memory_usage"]

                if f"device{device_id}_used_gb" not in actual_memory_usage:
                    continue

                ax.plot(
                    actual_memory_usage[f"device{device_id}_used_gb"],
                    color=schedule_name_to_color[run_info.schedule_name],
                    label=run_info.schedule_name,
                )

            ax.set_title(f"Schedule Comparison, device {device_id}")
            ax.set_xlabel("Schedule Timestep")
            ax.set_ylabel("Memory (GB)")

            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(curr_dir, f"device_id={device_id}.png"))
            plt.close(fig)


# ==== Driver logic ==== #


def main():
    loaded_data = load_data(BASE_CFG_LLAMA2_7B, LOG_DIR)

    # Uncomment to plot comparisons of predicted and actual memory use for
    # every device and log in LOG_DIR. Plots are saved to PLOT_SAVE_DIR.
    plot_actual_vs_predicted(loaded_data, PLOT_SAVE_DIR_ACTUAL_VS_PREDICTED)

    # Uncomment to print out DataFrames comparing predicted and actual memory
    # use for every device and log in LOG_DIR.
    # compare_memory_usage_estimates(
    #     loaded_data,
    #     left_estimate="predicted",
    #     right_estimate="used",
    # )

    # Uncomment to print out maximum deviations between predicted and actual
    # memory use for every device and log in LOG_DIR.
    # max_deviation_of_usage_estimates(
    #     loaded_data,
    #     left_estimate="predicted",
    #     right_estimate="used",
    # )

    # Uncomment to plot comparisons of between 1F1B and GPipe memory usage for
    # every device and log in LOG_DIR. Plots are saved to PLOT_SAVE_DIR.
    plot_1F1B_vs_gpipe(
        loaded_data,
        PLOT_SAVE_DIR_1F1B_VS_GPIPE,
    )


if __name__ == "__main__":
    main()
