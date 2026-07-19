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


RESULT_ROOT = Path("results2/results/homka_bgef_depth")
DEFAULT_OUTPUT = Path("analysis2/png/homka_bgef_depth_grouped_memory_table.png")
DEFAULT_CSV = Path("analysis2/tables/homka_bgef_depth_grouped_memory_table.csv")
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
    bench: str
    ram_kb: float | None
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grouped memory-usage heatmap table for homka_bgef_depth."
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
    return parser.parse_args()


def parse_result_file(path: Path) -> ResultRow | None:
    match = re.search(r"D(\d+) C(\d+).*G1", path.parent.name)
    if not match:
        return None

    depth, contexts = map(int, match.groups())
    bench = path.stem.replace("_grammar_1_1_1", "")

    try:
        df = pd.read_csv(path)
    except Exception:
        return ResultRow(depth, contexts, bench, None, "BAD")

    if df.empty:
        return ResultRow(depth, contexts, bench, None, "EMPTY")

    time_sec = pd.to_numeric(df.get("time_sec"), errors="coerce")
    ram_kb = pd.to_numeric(df.get("ram_kb"), errors="coerce")
    s_edges = pd.to_numeric(df.get("s_edges"), errors="coerce")
    ok = time_sec.notna() & ram_kb.notna() & s_edges.notna()

    if ok.sum() >= 2:
        return ResultRow(depth, contexts, bench, float(ram_kb[ok].tail(2).mean()), "OK")

    failed = df.loc[~ok]
    if not failed.empty:
        failure = str(failed.iloc[0].get("time_sec") or failed.iloc[0].get("s_edges"))
        return ResultRow(depth, contexts, bench, None, failure)

    return ResultRow(depth, contexts, bench, None, "PARTIAL")


def load_results(root: Path) -> pd.DataFrame:
    rows = [row for path in root.glob("*/*.csv") if (row := parse_result_file(path))]
    if not rows:
        raise SystemExit(f"No grouped result files found in {root}.")
    return pd.DataFrame([row.__dict__ for row in rows])


def ordered_benches(df: pd.DataFrame) -> list[str]:
    benches = [bench for bench in BENCH_ORDER if bench in set(df["bench"].unique())]
    benches.extend(sorted(set(df["bench"].unique()) - set(benches)))
    return benches


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


def choose_cell(cell: pd.DataFrame) -> tuple[float | None, str]:
    if cell.empty:
        return None, "-"

    successful = cell[(cell["status"] == "OK") & cell["ram_kb"].notna()]
    if not successful.empty:
        value = float(successful.iloc[0]["ram_kb"])
        return value, format_memory(value)

    statuses = [str(status) for status in cell["status"].dropna()]
    priority = ["OOM", "OOT", "EMPTY", "BAD", "PARTIAL"]
    selected = [status for status in priority if status in statuses]
    if selected:
        return None, "/".join(selected)

    return None, statuses[0] if statuses else "-"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
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
                ram_kb, label = choose_cell(cell)
                rows.append(
                    {
                        "bench": bench,
                        "depth": depth,
                        "contexts": context,
                        "ram_kb_last2_mean": ram_kb,
                        "cell": label,
                    }
                )

    return pd.DataFrame(rows)


def plot_table(df: pd.DataFrame, output: Path) -> None:
    contexts = [1, 2, 3, 5, 10]
    depths = [1, 2, 3, 4]
    columns = [(depth, context) for depth in depths for context in contexts]
    benches = ordered_benches(df)

    ok_values = df.loc[df["status"] == "OK", "ram_kb"].dropna()
    if ok_values.empty:
        raise SystemExit("No successful memory values found.")

    output.parent.mkdir(parents=True, exist_ok=True)

    fig_height = max(6.0, 1.05 + 0.42 * len(benches))
    fig, ax = plt.subplots(1, 1, figsize=(17.0, fig_height), constrained_layout=False)

    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#e5e7eb")

    value_matrix = np.full((len(benches), len(columns)), np.nan)
    color_matrix = np.full((len(benches), len(columns)), np.nan)
    labels = [["" for _ in columns] for __ in benches]

    for row_idx, bench in enumerate(benches):
        for col_idx, (depth, context) in enumerate(columns):
            cell = df[
                (df["bench"] == bench)
                & (df["depth"] == depth)
                & (df["contexts"] == context)
            ]
            ram_kb, label = choose_cell(cell)
            labels[row_idx][col_idx] = label
            if ram_kb is not None:
                value_matrix[row_idx, col_idx] = ram_kb

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

    image = ax.imshow(np.ma.masked_invalid(color_matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto")
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
            if text:
                color = "#111827"
                if not np.isnan(color_matrix[row_idx, col_idx]):
                    color = "#f8fafc" if color_matrix[row_idx, col_idx] > 0.65 else "#111827"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color=color,
                )

    fig.suptitle("Grouped memory usage", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.13, right=0.9, top=0.88, bottom=0.2)
    cbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.015)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["row min", "row mid", "row max"])
    cbar.set_label("within-benchmark memory scale")
    fig.text(
        0.5,
        0.06,
        "Cells show mean RAM over the last 2 successful grouped runs. '-' means the benchmark was not run in this configuration.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_results(RESULT_ROOT)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    build_table(df).to_csv(args.csv_output, index=False)

    plot_table(df, args.output)
    print(args.output)
    print(args.csv_output)


if __name__ == "__main__":
    main()
