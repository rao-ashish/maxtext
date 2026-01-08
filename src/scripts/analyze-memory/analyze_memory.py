import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys


def parse_log(
    filepath, start_marker="completed step: 7", end_marker="completed step: 8"
) -> pd.DataFrame:
    # Use the same parse_log implementation from previous steps
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=["name"])
    with open(filepath, "r") as f:
        lines = f.readlines()
    all_data = []
    column_names = None
    start_marker_seen = start_marker is None
    for line in lines:
        if not (start_marker_seen := start_marker_seen or start_marker in line):
            continue
        if end_marker is not None and end_marker in line:
            break
        if "MEM" not in line:
            continue
        if "MEM,name" in line:
            if column_names is None:
                column_names = line.split(",")[1:]
            continue
        curr_data = line.split(",")[1:]
        for idx in range(len(curr_data)):
            if idx == 0:
                continue
            try:
                curr_data[idx] = float(curr_data[idx])
            except:
                pass
        all_data.append(curr_data)
    out = pd.DataFrame(all_data, columns=column_names)
    cols_to_numeric = [c for c in out.columns if c != "name"]
    out[cols_to_numeric] = out[cols_to_numeric].apply(pd.to_numeric, errors="coerce")
    if "state_total_gb" not in out.columns:
        out["state_total_gb"] = out.filter(like="state").sum(axis=1)
    return out


def adjust_lightness(color, amount=0.7):
    """Darkens or lightens a color. amount < 1.0 darkens."""
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def shorten_label(label_str):
    pattern = r"mubatch_idx=(\d+)\|stage_idx=(\d+)\|is_bwd=(True|False)"
    match = re.search(pattern, label_str)
    if match:
        mu_idx = match.group(1)
        stage_idx = match.group(2)
        direction = "B" if match.group(3) == "True" else "F"
        return f"(mu={mu_idx},s={stage_idx},{direction})"
    return label_str


def make_plot(
    name_to_df, outputs_dir, device_id=0, num_devices=8, only_plot_stage_tasks=True
):
    plt.figure(dpi=100, figsize=(12, 6))
    plt.style.use("ggplot")

    mode_str = "Stage Tasks Only" if only_plot_stage_tasks else "All Events"
    plt.title(f"Total Memory Usage (Device {device_id}) - {mode_str}")
    plt.ylabel("Memory (GB)")
    plt.xlabel("Step Index" if only_plot_stage_tasks else "Global Event Index")
    plt.grid(True)

    for name, df in name_to_df.items():
        if df.empty:
            continue

        if only_plot_stage_tasks:
            mask = df["name"].str.contains(f"stage_idx={device_id // 2}", na=False)
            subset = df.loc[mask]

            y_values_used_gb = subset[f"device{device_id}_used_gb"]
            y_values_known_state_gb = subset[f"device{device_id}_known_state_gb"]

            labels_raw = subset["name"]
            x_values = range(len(y_values_used_gb))
        else:
            y_values_used_gb = df[f"device{device_id}_used_gb"]
            y_values_known_state_gb = df[f"device{device_id}_known_state_gb"]

            labels_raw = df["name"]
            x_values = df.index

        (line_used_gb,) = plt.plot(
            x_values,
            y_values_used_gb,
            label=name,
            marker="o",
            markersize=4,
            linestyle="-",
        )
        line_color = line_used_gb.get_color()

        plt.plot(
            x_values,
            y_values_known_state_gb,
            label=name,
            marker="o",
            markersize=4,
            linestyle="--",
        )

        text_color = adjust_lightness(line_color, amount=0.6)

        # Iterate over the explicit x_values, y_values, and labels
        for x, y_used_gb, y_known_state_gb, raw_label in zip(
            x_values, y_values_used_gb, y_values_known_state_gb, labels_raw
        ):
            short_label = shorten_label(raw_label)
            plt.text(
                x,
                y_used_gb,
                short_label,
                color=text_color,
                fontsize=8,
                ha="left",
                va="bottom",
                rotation=0,
            )
            plt.text(
                x,
                y_known_state_gb,
                short_label,
                color=text_color,
                fontsize=8,
                ha="left",
                va="bottom",
                rotation=0,
            )

    plt.legend()
    plt.tight_layout()
    os.makedirs(outputs_dir, exist_ok=True)
    output_filename = os.path.join(outputs_dir, f"memory_plot_device_{device_id}.png")
    plt.savefig(output_filename)
    plt.close()


def make_plots():
    name_to_df = {
        "gpipe": parse_log("gpipe.log"),
        "1F1B": parse_log("1F1B.log"),
    }

    for device_id in range(8):
        make_plot(
            name_to_df,
            "plots/stage_tasks_only",
            device_id=device_id,
            num_devices=8,
            only_plot_stage_tasks=True,
        )
        make_plot(
            name_to_df,
            "plots/all_events",
            device_id=device_id,
            num_devices=8,
            only_plot_stage_tasks=False,
        )


def main():
    make_plots()


if __name__ == "__main__":
    main()
