import re
from pathlib import Path

import pandas as pd


LOG_DIR = "/home/asrao/workspace/maxtext/src/scripts/sweeps/outputs/performance-sweeps-no-passthrough"
TARGET_STEPS = set(range(5, 10))
IMPLEMENTATION_ORDER = ["spmd-gpipe", "mpmd-gpipe", "mpmd-1F1B"]

SINGLE_RUN_PATTERN = re.compile(
		r"^(mpmd-(?:1F1B|gpipe))-num_repeats=(\d+).*\.log$"
)
SINGLE_SPMD_PATTERN = re.compile(r"^spmd-num_repeats_(\d+).*\.log$")
MULTIPROCESS_PATTERN = re.compile(
		r"^(mpmd-(?:1F1B|gpipe))-multiprocess=(\d+)-num_repeats=(\d+).*$"
)
MULTIPROCESS_SPMD_PATTERN = re.compile(r"^spmd-num_repeats_(\d+)-multiprocess_(\d+).*$")
PROCESS_LOG_PATTERN = re.compile(r"^process_(\d+)\.log$")
STEP_METRIC_PATTERN = re.compile(
		r"completed step:\s*(\d+).*?TFLOP/s/device:\s*([0-9]*\.?[0-9]+).*?Tokens/s/device:\s*([0-9]*\.?[0-9]+)"
)


def _parse_metrics_from_log(log_path: Path) -> tuple[float, float, float, float]:
	tflops_by_step: dict[int, float] = {}
	tokens_by_step: dict[int, float] = {}

	with log_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
		for line in file_handle:
			match = STEP_METRIC_PATTERN.search(line)
			if not match:
				continue

			step = int(match.group(1))
			if step not in TARGET_STEPS:
				continue

			tflops_by_step[step] = float(match.group(2))
			tokens_by_step[step] = float(match.group(3))

	missing_steps = TARGET_STEPS.difference(tflops_by_step.keys())
	if missing_steps:
		raise ValueError(f"Missing step metrics {sorted(missing_steps)} in {log_path}")

	tflops_series = pd.Series([tflops_by_step[step] for step in sorted(TARGET_STEPS)], dtype="float64")
	tokens_series = pd.Series([tokens_by_step[step] for step in sorted(TARGET_STEPS)], dtype="float64")

	return (
			float(tflops_series.mean()),
			float(tokens_series.mean()),
			float(tflops_series.std(ddof=1)),
			float(tokens_series.std(ddof=1)),
	)


def _get_last_process_log(run_dir: Path) -> Path:
	process_logs: list[tuple[int, Path]] = []
	for child in run_dir.iterdir():
		if not child.is_file():
			continue
		match = PROCESS_LOG_PATTERN.match(child.name)
		if not match:
			continue
		process_logs.append((int(match.group(1)), child))

	if not process_logs:
		raise ValueError(f"No process logs found in multiprocess directory: {run_dir}")

	process_logs.sort(key=lambda item: item[0])
	return process_logs[-1][1]


def _collect_run_rows(log_dir: Path) -> list[dict[str, float | int | str]]:
	rows: list[dict[str, float | int | str]] = []

	for child in sorted(log_dir.iterdir(), key=lambda path: path.name):
		if child.is_file():
			single_match = SINGLE_RUN_PATTERN.match(child.name)
			spmd_single_match = SINGLE_SPMD_PATTERN.match(child.name)
			if single_match:
				implementation = single_match.group(1)
				num_repeats = int(single_match.group(2))
			elif spmd_single_match:
				implementation = "spmd-gpipe"
				num_repeats = int(spmd_single_match.group(1))
			else:
				continue

			num_processes = 1
			log_to_parse = child
		elif child.is_dir():
			multi_match = MULTIPROCESS_PATTERN.match(child.name)
			if multi_match:
				implementation = multi_match.group(1)
				num_processes = int(multi_match.group(2))
				num_repeats = int(multi_match.group(3))
			else:
				spmd_multi_match = MULTIPROCESS_SPMD_PATTERN.match(child.name)
				if not spmd_multi_match:
					continue
				implementation = "spmd-gpipe"
				num_repeats = int(spmd_multi_match.group(1))
				num_processes = int(spmd_multi_match.group(2))

			log_to_parse = _get_last_process_log(child)
		else:
			continue

		mean_tflops, mean_tokens, std_tflops, std_tokens = _parse_metrics_from_log(log_to_parse)
		rows.append(
				{
						"implementation": implementation,
						"num_processes": num_processes,
						"num_repeats": num_repeats,
						"mean-TFLOP/sec/device": mean_tflops,
						"mean-Tokens/sec/device": mean_tokens,
						"stdv-TFLOP/sec/device": std_tflops,
						"stdv-Tokens/sec/device": std_tokens,
				}
		)

	return rows


def build_summary_dataframe(log_dir: Path) -> pd.DataFrame:
	rows = _collect_run_rows(log_dir)
	if not rows:
		raise ValueError(f"No matching runs found under {log_dir}")

	dataframe = pd.DataFrame(rows)

	baseline = (
			dataframe[dataframe["implementation"] == "spmd-gpipe"][
					["num_processes", "num_repeats", "mean-TFLOP/sec/device", "mean-Tokens/sec/device"]
			]
			.rename(
					columns={
							"mean-TFLOP/sec/device": "baseline_mean_tflops",
							"mean-Tokens/sec/device": "baseline_mean_tokens",
					}
			)
			.drop_duplicates(subset=["num_processes", "num_repeats"])
	)

	dataframe = dataframe.merge(baseline, on=["num_processes", "num_repeats"], how="left")
	if dataframe[["baseline_mean_tflops", "baseline_mean_tokens"]].isna().any().any():
		missing = dataframe[
				dataframe[["baseline_mean_tflops", "baseline_mean_tokens"]].isna().any(axis=1)
		][["num_processes", "num_repeats"]].drop_duplicates()
		raise ValueError(
				"Missing spmd-gpipe baseline for some runs: "
				+ ", ".join(
						f"num_processes={int(row.num_processes)}-num_repeats={int(row.num_repeats)}"
						for row in missing.itertuples(index=False)
				)
		)

	dataframe["mean_TFLOPs-percent_SPMD"] = (
			dataframe["mean-TFLOP/sec/device"] / dataframe["baseline_mean_tflops"]
	) * 100.0
	dataframe["mean_tokens/sec-percent_SPMD"] = (
			dataframe["mean-Tokens/sec/device"] / dataframe["baseline_mean_tokens"]
	) * 100.0

	dataframe = dataframe[
			[
					"implementation",
					"num_processes",
					"num_repeats",
					"mean_TFLOPs-percent_SPMD",
					"mean-TFLOP/sec/device",
					"mean-Tokens/sec/device",
					"stdv-TFLOP/sec/device",
					"stdv-Tokens/sec/device",
					"mean_tokens/sec-percent_SPMD",
			]
	]

	dataframe["implementation"] = pd.Categorical(
			dataframe["implementation"],
			categories=IMPLEMENTATION_ORDER,
			ordered=True,
	)

	dataframe = dataframe.sort_values(
			by=["num_processes", "implementation", "num_repeats"],
			ascending=[True, True, True],
			ignore_index=True,
	)

	dataframe["implementation"] = dataframe["implementation"].astype(str)

	return dataframe


def main() -> None:
	log_dir = Path(LOG_DIR)
	summary_dataframe = build_summary_dataframe(log_dir)

	pd.set_option("display.max_rows", None)
	pd.set_option("display.max_columns", None)
	pd.set_option("display.width", 200)
	pd.set_option("display.float_format", "{:.3f}".format)
	print(summary_dataframe)

	output_path = log_dir / "summary.csv"
	summary_dataframe.to_csv(output_path, index=False)
	print(f"\nSaved summary CSV to: {output_path}")


if __name__ == "__main__":
	main()