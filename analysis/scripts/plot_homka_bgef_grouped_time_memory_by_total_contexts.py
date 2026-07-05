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
from matplotlib.colors import LogNorm


RESULT_ROOT = Path("results/homka_bgef_depth")
DEFAULT_TIME_OUTPUT = Path("analysis/png/homka_bgef_depth_grouped_time_by_total_contexts.png")
DEFAULT_MEMORY_OUTPUT = Path("analysis/png/homka_bgef_depth_grouped_memory_by_total_contexts.png")
DEFAULT_CSV = Path("analysis/tables/homka_bgef_depth_grouped_time_memory_by_total_contexts.csv")
CONTEXTS = [1, 2, 3, 5, 10]
DEPTHS = [1, 2, 3, 4]
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
    depth: int
    contexts: int
    total_contexts: int
    bench: str
    time_sec: float | None
    ram_kb: float | None
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped time and memory heatmaps sorted by total context count."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=RESULT_ROOT,
        help=f"Path to grouped result root (default: {RESULT_ROOT}).",
    )
    parser.add_argument(
        "--time-output",
        type=Path,
        default=DEFAULT_TIME_OUTPUT,
        help=f"Path to the time output PNG (default: {DEFAULT_TIME_OUTPUT}).",
    )
    parser.add_argument(
        "--memory-output",
        type=Path,
        default=DEFAULT_MEMORY_OUTPUT,
        help=f"Path to the memory output PNG (default: {DEFAULT_MEMORY_OUTPUT}).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to the long-format CSV table (default: {DEFAULT_CSV}).",
    )
    return parser.parse_args()


def total_contexts(num: int, depth: int) -> int:
    return sum(num**i for i in range(1, depth + 1))


def parse_result_file(path: Path) -> ResultRow | None:
    match = re.search(r"D(\d+) C(\d+).*G1", path.parent.name)
    if not match:
        return None

    depth, contexts = map(int, match.groups())
    bench = path.stem.replace("_grammar_1_1_1", "")
    total = total_contexts(contexts, depth)

    try:
        df = pd.read_csv(path)
    except Exception:
        return ResultRow(depth, contexts, total, bench, None, None, "BAD")

    if df.empty:
        return ResultRow(depth, contexts, total, bench, None, None, "EMPTY")

    time_sec = pd.to_numeric(df.get("time_sec"), errors="coerce")
    ram_kb = pd.to_numeric(df.get("ram_kb"), errors="coerce")
    s_edges = pd.to_numeric(df.get("s_edges"), errors="coerce")
    ok = time_sec.notna() & ram_kb.notna() & s_edges.notna()

    if len(df) == 3 and ok.all():
        return ResultRow(
            depth,
            contexts,
            total,
            bench,
            float(time_sec.mean()),
            float(ram_kb.tail(2).mean()),
            "OK",
        )

    if ok.sum() >= 2:
        return ResultRow(
            depth,
            contexts,
            total,
            bench,
            float(time_sec[ok].mean()),
            float(ram_kb[ok].tail(2).mean()),
            "OK",
        )

    failed = df.loc[~ok]
    if not failed.empty:
        failure = str(failed.iloc[0].get("time_sec") or failed.iloc[0].get("s_edges"))
        return ResultRow(depth, contexts, total, bench, None, None, failure)

    return ResultRow(depth, contexts, total, bench, None, None, "PARTIAL")


def load_results(root: Path) -> pd.DataFrame:
    rows = [row for path in root.glob("*/*.csv") if (row := parse_result_file(path))]
    if not rows:
        raise SystemExit(f"No grouped result files found in {root}.")
    return pd.DataFrame([row.__dict__ for row in rows])


def ordered_benches(df: pd.DataFrame) -> list[str]:
    benches = [bench for bench in BENCH_ORDER if bench in set(df["bench"].unique())]
    benches.extend(sorted(set(df["bench"].unique()) - set(benches)))
    return benches


def ordered_columns(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    columns = sorted(
        {
            (int(row.depth), int(row.contexts), int(row.total_contexts))
            for row in df[["depth", "contexts", "total_contexts"]].itertuples(index=False)
        },
        key=lambda item: (item[2], item[0], item[1]),
    )
    return columns


def format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 10:
        return f"{seconds:.2g}s"
    if seconds < 100:
        return f"{seconds:.1f}s"
    return f"{seconds:.0f}s"


def format_memory(kb: float) -> str:
    mib = kb / 1024
    gib = mib / 1024
    if gib >= 10:
        return f"{gib:.0f}GiB"
    if gib >= 1:
        return f"{gib:.1f}GiB"
    if mib >= 100:
        return f"{mib:.0f}MiB"
    return f"{mib:.1f}MiB"


def choose_cell(cell: pd.DataFrame, value_column: str) -> tuple[float | None, str]:
    if cell.empty:
        return None, "-"

    successful = cell[(cell["status"] == "OK") & cell[value_column].notna()]
    if not successful.empty:
        value = float(successful.iloc[0][value_column])
        label = format_time(value) if value_column == "time_sec" else format_memory(value)
        return value, label

    statuses = [str(status) for status in cell["status"].dropna()]
    priority = ["OOM", "OOT", "EMPTY", "BAD", "PARTIAL"]
    selected = [status for status in priority if status in statuses]
    if selected:
        return None, "/".join(selected)

    return None, statuses[0] if statuses else "-"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bench in ordered_benches(df):
        for depth, contexts, total in ordered_columns(df):
            cell = df[
                (df["bench"] == bench)
                & (df["depth"] == depth)
                & (df["contexts"] == contexts)
            ]
            time_sec, time_label = choose_cell(cell, "time_sec")
            ram_kb, ram_label = choose_cell(cell, "ram_kb")
            rows.append(
                {
                    "bench": bench,
                    "depth": depth,
                    "contexts": contexts,
                    "total_contexts": total,
                    "time_sec": time_sec,
                    "time_cell": time_label,
                    "ram_kb_last2_mean": ram_kb,
                    "memory_cell": ram_label,
                }
            )

    return pd.DataFrame(rows)


def fill_matrices(
    df: pd.DataFrame,
    benches: list[str],
    columns: list[tuple[int, int, int]],
    value_column: str,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    value_matrix = np.full((len(benches), len(columns)), np.nan)
    color_matrix = np.full((len(benches), len(columns)), np.nan)
    labels = [["" for _ in columns] for __ in benches]

    for row_idx, bench in enumerate(benches):
        for col_idx, (depth, contexts, _total) in enumerate(columns):
            cell = df[
                (df["bench"] == bench)
                & (df["depth"] == depth)
                & (df["contexts"] == contexts)
            ]
            value, label = choose_cell(cell, value_column)
            labels[row_idx][col_idx] = label
            if value is not None:
                value_matrix[row_idx, col_idx] = value

    for row_idx in range(len(benches)):
        row_values = value_matrix[row_idx]
        finite = row_values[np.isfinite(row_values)]
        if len(finite) == 0:
            continue
        if float(finite.min()) == float(finite.max()):
            color_matrix[row_idx, np.isfinite(row_values)] = 0.5
            continue
        row_norm = LogNorm(vmin=max(1e-6, float(finite.min())), vmax=float(finite.max()))
        color_matrix[row_idx, np.isfinite(row_values)] = row_norm(finite)

    return value_matrix, color_matrix, labels


def draw_heatmap(
    ax,
    df: pd.DataFrame,
    benches: list[str],
    columns: list[tuple[int, int, int]],
    value_column: str,
    title: str,
    cmap_name: str,
) -> object:
    _value_matrix, color_matrix, labels = fill_matrices(df, benches, columns, value_column)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e5e7eb")
    image = ax.imshow(np.ma.masked_invalid(color_matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_yticks(range(len(benches)), labels=benches, fontsize=8)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"D{depth}\nC{contexts}" for depth, contexts, _total in columns], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(benches), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(len(benches)):
        for col_idx in range(len(columns)):
            text = labels[row_idx][col_idx]
            if text:
                color = "#111827"
                if not np.isnan(color_matrix[row_idx, col_idx]):
                    color = "#f8fafc" if color_matrix[row_idx, col_idx] > 0.65 else "#111827"
                ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=6.4, color=color)

    return image


def plot_metric(
    df: pd.DataFrame,
    output: Path,
    value_column: str,
    title: str,
    colorbar_label: str,
    note: str,
) -> None:
    benches = ordered_benches(df)
    columns = ordered_columns(df)

    if df.loc[df["status"] == "OK", value_column].dropna().empty:
        raise SystemExit(f"No successful {value_column} values found.")

    output.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(17.0, 0.72 * len(columns))
    fig_height = max(6.4, 2.0 + 0.42 * len(benches))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), constrained_layout=False)

    image = draw_heatmap(ax, df, benches, columns, value_column, title, "magma_r")

    bottom_labels = [str(total) for _depth, _contexts, total in columns]
    secondary = ax.secondary_xaxis("bottom")
    secondary.set_xticks(range(len(columns)))
    secondary.set_xticklabels(bottom_labels, fontsize=7, rotation=45, ha="right")
    secondary.spines["bottom"].set_position(("outward", 34))
    secondary.set_xlabel("total contexts")

    fig.suptitle(title + " by total context count", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.13, right=0.9, top=0.88, bottom=0.29)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.015)
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["row min", "row mid", "row max"])
    colorbar.set_label(colorbar_label)

    fig.text(
        0.5,
        0.075,
        r"Total contexts: $\sum_{i=1}^{d} n^i$, where $n$ is the number of contexts and $d$ is depth.",
        ha="center",
        fontsize=10,
        color="#334155",
    )
    fig.text(
        0.5,
        0.025,
        note,
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_results(args.root)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    build_table(df).to_csv(args.csv_output, index=False)

    plot_metric(
        df,
        args.time_output,
        "time_sec",
        "Grouped running time",
        "within-benchmark time scale",
        "Cells show mean successful runtime.",
    )
    plot_metric(
        df,
        args.memory_output,
        "ram_kb",
        "Grouped memory usage",
        "within-benchmark memory scale",
        "Cells show mean RAM over the last 2 successful runs.",
    )
    print(args.time_output)
    print(args.memory_output)
    print(args.csv_output)


if __name__ == "__main__":
    main()
