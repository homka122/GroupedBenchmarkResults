from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cfpq")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


GROUPED_ROOT = Path("results2/results/homka_bgef_depth")
NOT_GROUPED_ROOT = Path("results2/results/homka_bgef_depth_g0")
DEFAULT_OUTPUT = Path("analysis2/png/homka_bgef_depth_grouped_speedup_table.png")
DEFAULT_CSV = Path("analysis2/tables/homka_bgef_depth_grouped_speedup_table.csv")
TIMEOUT_SECONDS = 1200.0
BENCH_ORDER = [
    "basic",
    "collections",
    "cornerCases",
    "generalJava",
    "reactor",
    "org_jivesoftware_openfire",
    "com_fasterxml_jackson",
    "org_apache_jackrabbit",
]


@dataclass(frozen=True)
class ResultRow:
    variant: str
    depth: int
    contexts: int
    bench: str
    time_sec: float | None
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grouped/not-grouped speedup heatmap table for homka_bgef_depth."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the output PNG (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to the long-format CSV table (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=TIMEOUT_SECONDS,
        help=f"Time to use for OOT lower-bound speedups (default: {TIMEOUT_SECONDS}).",
    )
    return parser.parse_args()


def parse_result_file(path: Path, variant: str, group: int) -> ResultRow | None:
    match = re.search(rf"D(\d+) C(\d+).*G{group}", path.parent.name)
    if not match:
        return None

    depth, contexts = map(int, match.groups())
    bench = path.stem.replace("_grammar_1_1_1", "")

    try:
        df = pd.read_csv(path)
    except Exception:
        return ResultRow(variant, depth, contexts, bench, None, "BAD")

    if df.empty:
        return ResultRow(variant, depth, contexts, bench, None, "EMPTY")

    time_sec = pd.to_numeric(df.get("time_sec"), errors="coerce")
    ram_kb = pd.to_numeric(df.get("ram_kb"), errors="coerce")
    s_edges = pd.to_numeric(df.get("s_edges"), errors="coerce")
    ok = time_sec.notna() & ram_kb.notna() & s_edges.notna()

    if len(df) == 3 and ok.all():
        return ResultRow(variant, depth, contexts, bench, float(time_sec.mean()), "OK")

    failed = df.loc[~ok]
    if not failed.empty:
        failure = str(failed.iloc[0].get("time_sec") or failed.iloc[0].get("s_edges"))
        return ResultRow(variant, depth, contexts, bench, None, failure)

    return ResultRow(variant, depth, contexts, bench, None, "PARTIAL")


def load_results() -> pd.DataFrame:
    rows: list[ResultRow] = []
    for path in GROUPED_ROOT.glob("*/*.csv"):
        row = parse_result_file(path, "grouped", 1)
        if row is not None:
            rows.append(row)
    for path in NOT_GROUPED_ROOT.glob("*/*.csv"):
        row = parse_result_file(path, "not_grouped", 0)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No homka_bgef_depth result files found.")

    return pd.DataFrame([row.__dict__ for row in rows])


def ordered_benches(df: pd.DataFrame) -> list[str]:
    benches = [bench for bench in BENCH_ORDER if bench in set(df["bench"].unique())]
    benches.extend(sorted(set(df["bench"].unique()) - set(benches)))
    return benches


def format_speedup(value: float, lower_bound: bool = False, upper_bound: bool = False) -> str:
    if value >= 100:
        text = f"{value:.0f}x"
    elif value >= 10:
        text = f"{value:.1f}x"
    else:
        text = f"{value:.2g}x"
    if lower_bound:
        return f"{text}+"
    if upper_bound:
        return f"<{text}"
    return text


def select_result(cell: pd.DataFrame) -> tuple[float | None, str]:
    if cell.empty:
        return None, "-"
    row = cell.iloc[0]
    if row["status"] == "OK" and pd.notna(row["time_sec"]):
        return float(row["time_sec"]), "OK"
    return None, str(row["status"])


def choose_cell(cell: pd.DataFrame, timeout_seconds: float) -> tuple[float | None, str, str]:
    grouped_time, grouped_status = select_result(cell[cell["variant"] == "grouped"])
    not_grouped_time, not_grouped_status = select_result(cell[cell["variant"] == "not_grouped"])

    if grouped_status == "-" and not_grouped_status == "-":
        return None, "-", "missing"
    if grouped_status == "OOM" and not_grouped_status == "OOM":
        return None, "OOM", "oom"
    if grouped_status == "OOM" or not_grouped_status == "OOM":
        return None, "-", "one_oom"

    if grouped_time is not None and not_grouped_time is not None:
        speedup = not_grouped_time / grouped_time
        return speedup, format_speedup(speedup), "exact"

    if grouped_time is not None and not_grouped_status == "OOT":
        speedup = timeout_seconds / grouped_time
        return speedup, format_speedup(speedup, lower_bound=True), "lower_bound"

    if not_grouped_time is not None and grouped_status == "OOT":
        speedup = not_grouped_time / timeout_seconds
        return speedup, format_speedup(speedup, upper_bound=True), "upper_bound"

    if grouped_status == "OOT" and not_grouped_status == "OOT":
        return None, "OOT", "oot"

    return None, "-", "unavailable"


def build_table(df: pd.DataFrame, timeout_seconds: float) -> pd.DataFrame:
    contexts = [1, 2, 3, 5, 10]
    depths = [1, 2, 3, 4]

    rows = []
    for bench in ordered_benches(df):
        for depth in depths:
            for context in contexts:
                cell = df[
                    (df["bench"] == bench)
                    & (df["depth"] == depth)
                    & (df["contexts"] == context)
                ]
                speedup, label, status = choose_cell(cell, timeout_seconds)
                rows.append(
                    {
                        "bench": bench,
                        "depth": depth,
                        "contexts": context,
                        "speedup_not_grouped_over_grouped": speedup,
                        "cell": label,
                        "status": status,
                    }
                )

    return pd.DataFrame(rows)


def plot_table(df: pd.DataFrame, output: Path, timeout_seconds: float) -> None:
    contexts = [1, 2, 3, 5, 10]
    depths = [1, 2, 3, 4]
    columns = [(depth, context) for depth in depths for context in contexts]
    benches = ordered_benches(df)

    output.parent.mkdir(parents=True, exist_ok=True)

    value_matrix = np.full((len(benches), len(columns)), np.nan)
    color_matrix = np.full((len(benches), len(columns)), np.nan)
    labels = [["" for _ in columns] for __ in benches]
    statuses = [["" for _ in columns] for __ in benches]

    for row_idx, bench in enumerate(benches):
        for col_idx, (depth, context) in enumerate(columns):
            cell = df[
                (df["bench"] == bench)
                & (df["depth"] == depth)
                & (df["contexts"] == context)
            ]
            speedup, label, status = choose_cell(cell, timeout_seconds)
            labels[row_idx][col_idx] = label
            statuses[row_idx][col_idx] = status
            if speedup is not None and np.isfinite(speedup) and speedup > 0:
                value_matrix[row_idx, col_idx] = speedup
                color_matrix[row_idx, col_idx] = np.log2(speedup)

    finite = color_matrix[np.isfinite(color_matrix)]
    if len(finite) == 0:
        raise SystemExit("No comparable grouped/not-grouped speedup values found.")
    max_abs = max(1.0, float(np.nanmax(np.abs(finite))))

    fig_height = max(6.0, 1.05 + 0.42 * len(benches))
    fig, ax = plt.subplots(1, 1, figsize=(17.0, fig_height), constrained_layout=False)

    cmap = LinearSegmentedColormap.from_list("speedup", ["#dc2626", "#f8fafc", "#16a34a"]).copy()
    cmap.set_bad("#e5e7eb")
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    image = ax.imshow(np.ma.masked_invalid(color_matrix), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(
        range(len(columns)),
        labels=[f"D{depth}\nC{context}" for depth, context in columns],
        fontsize=8,
    )
    for depth_idx, depth in enumerate(depths):
        start = depth_idx * len(contexts)
        end = start + len(contexts) - 1
        center = (start + end) / 2
        ax.text(
            center,
            -0.9,
            f"D{depth}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#334155",
        )
        if depth_idx > 0:
            ax.axvline(start - 0.5, color="#0f172a", linewidth=2.0, alpha=0.55)

    ax.set_yticks(range(len(benches)), labels=benches, fontsize=8)
    ax.set_xlabel("depth / contexts")
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(benches), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(len(benches)):
        for col_idx in range(len(columns)):
            text = labels[row_idx][col_idx]
            if not text:
                continue
            color = "#111827"
            if np.isfinite(color_matrix[row_idx, col_idx]) and abs(color_matrix[row_idx, col_idx]) > 0.85 * max_abs:
                color = "#f8fafc"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=6.6,
                color=color,
            )

    fig.suptitle("Grouped speedup over not-grouped", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.13, right=0.9, top=0.88, bottom=0.2)
    cbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.015)
    ticks = [x for x in [0.25, 0.5, 1, 2, 4, 8, 16, 32, 128] if -max_abs <= np.log2(x) <= max_abs]
    cbar.set_ticks([np.log2(x) for x in ticks])
    cbar.set_ticklabels([format_speedup(x) for x in ticks])
    cbar.set_label("speedup = not-grouped time / grouped time")
    fig.text(
        0.5,
        0.06,
        f"Values >1x mean grouped is faster. OOT is treated as {timeout_seconds:.0f}s and shown as a lower bound with '+'. One-sided OOM is '-'; both OOM is 'OOM'.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_results()

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    build_table(df, args.timeout_seconds).to_csv(args.csv_output, index=False)

    plot_table(df, args.output, args.timeout_seconds)
    print(args.output)
    print(args.csv_output)


if __name__ == "__main__":
    main()
