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


GROUPED_ROOT = Path("results/homka_bgef_depth")
NOT_GROUPED_ROOT = Path("results/homka_bgef_depth_g0")
GRAPH_SUMMARY = Path("analysis/graphs/summary.md")
DEFAULT_OUTPUT = Path("analysis/png/homka_bgef_depth_speedup_by_context_field_ratio.png")
DEFAULT_CSV = Path("analysis/tables/homka_bgef_depth_speedup_by_context_field_ratio.csv")
TIMEOUT_SECONDS = 600.0

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
        description="Create a speedup trend plot over total-contexts/fields ratio."
    )
    parser.add_argument(
        "--grouped-root",
        type=Path,
        default=GROUPED_ROOT,
        help=f"Path to grouped result root (default: {GROUPED_ROOT}).",
    )
    parser.add_argument(
        "--not-grouped-root",
        type=Path,
        default=NOT_GROUPED_ROOT,
        help=f"Path to not-grouped result root (default: {NOT_GROUPED_ROOT}).",
    )
    parser.add_argument(
        "--graph-summary",
        type=Path,
        default=GRAPH_SUMMARY,
        help=f"Path to graph summary markdown (default: {GRAPH_SUMMARY}).",
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
        help=f"Path to the output CSV with plotted points (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=TIMEOUT_SECONDS,
        help=f"Time to use for OOT lower-bound speedups (default: {TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--include-lower-bounds",
        action="store_true",
        help="Include speedup lower bounds when not-grouped timed out and grouped completed.",
    )
    return parser.parse_args()


def total_contexts(num: int, depth: int) -> int:
    return sum(num**i for i in range(1, depth + 1))


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


def load_results(grouped_root: Path, not_grouped_root: Path) -> pd.DataFrame:
    rows: list[ResultRow] = []
    for path in grouped_root.glob("*/*.csv"):
        row = parse_result_file(path, "grouped", 1)
        if row is not None:
            rows.append(row)
    for path in not_grouped_root.glob("*/*.csv"):
        row = parse_result_file(path, "not_grouped", 0)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No grouped/not-grouped result files found.")

    return pd.DataFrame([row.__dict__ for row in rows])


def load_fields(summary_path: Path) -> dict[str, int]:
    if not summary_path.exists():
        raise SystemExit(f"Graph summary does not exist: {summary_path}")

    rows = []
    with open(summary_path, "r", encoding="utf-8") as summary_file:
        for line in summary_file:
            stripped = line.strip()
            if not stripped.startswith("|") or "---" in stripped or stripped.startswith("| graph"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) < 8:
                continue
            rows.append(cells)

    fields_by_graph = {}
    for cells in rows:
        graph = cells[0]
        fields = int(cells[7])
        fields_by_graph[graph] = fields

    if not fields_by_graph:
        raise SystemExit(f"No graph fields found in {summary_path}")

    return fields_by_graph


def select_result(cell: pd.DataFrame) -> tuple[float | None, str]:
    if cell.empty:
        return None, "-"
    row = cell.iloc[0]
    if row["status"] == "OK" and pd.notna(row["time_sec"]):
        return float(row["time_sec"]), "OK"
    return None, str(row["status"])


def build_speedup_points(
    df: pd.DataFrame,
    fields_by_graph: dict[str, int],
    timeout_seconds: float,
    include_lower_bounds: bool,
) -> pd.DataFrame:
    rows = []
    keys = df[["bench", "depth", "contexts"]].drop_duplicates().itertuples(index=False)
    for bench, depth, contexts in keys:
        fields = fields_by_graph.get(bench)
        if fields is None or fields <= 0:
            continue

        cell = df[(df["bench"] == bench) & (df["depth"] == depth) & (df["contexts"] == contexts)]
        grouped_time, grouped_status = select_result(cell[cell["variant"] == "grouped"])
        not_grouped_time, not_grouped_status = select_result(cell[cell["variant"] == "not_grouped"])

        bound = "exact"
        if grouped_time is not None and not_grouped_time is not None:
            speedup = not_grouped_time / grouped_time
        elif include_lower_bounds and grouped_time is not None and not_grouped_status == "OOT":
            speedup = timeout_seconds / grouped_time
            bound = "lower_bound"
        else:
            continue

        if not np.isfinite(speedup) or speedup <= 0:
            continue

        total = total_contexts(int(contexts), int(depth))
        rows.append(
            {
                "bench": bench,
                "depth": int(depth),
                "contexts": int(contexts),
                "total_contexts": total,
                "fields": fields,
                "context_field_ratio": total / fields,
                "speedup_not_grouped_over_grouped": speedup,
                "bound": bound,
                "grouped_status": grouped_status,
                "not_grouped_status": not_grouped_status,
            }
        )

    if not rows:
        raise SystemExit("No comparable speedup points found.")

    return pd.DataFrame(rows).sort_values(["context_field_ratio", "speedup_not_grouped_over_grouped", "bench"])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] == 0:
        return float("nan")
    return float(np.interp(quantile * cumulative[-1], cumulative, sorted_values))


def smooth_profile(x: np.ndarray, y: np.ndarray, grid_x: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.full_like(grid_x, np.nan, dtype=float)
    median = np.full_like(grid_x, np.nan, dtype=float)
    upper = np.full_like(grid_x, np.nan, dtype=float)

    for idx, center in enumerate(grid_x):
        weights = np.exp(-0.5 * ((x - center) / bandwidth) ** 2)
        if weights.sum() < 1e-6:
            continue
        lower[idx] = weighted_quantile(y, weights, 0.25)
        median[idx] = weighted_quantile(y, weights, 0.50)
        upper[idx] = weighted_quantile(y, weights, 0.75)

    return lower, median, upper


def format_ratio_tick(value: float) -> str:
    if value >= 1:
        return f"{value:g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def plot_speedup(points: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    x = np.log10(points["context_field_ratio"].to_numpy(dtype=float))
    y = np.log2(points["speedup_not_grouped_over_grouped"].to_numpy(dtype=float))
    x_margin = max(0.08, 0.04 * (float(x.max()) - float(x.min())))
    y_margin = max(0.18, 0.06 * (float(y.max()) - float(y.min())))
    x_min = float(x.min()) - x_margin
    x_max = float(x.max()) + x_margin
    y_min = float(y.min()) - y_margin
    y_max = float(y.max()) + y_margin

    fig, ax = plt.subplots(1, 1, figsize=(11.8, 7.2), constrained_layout=False)
    ax.set_facecolor("#f8fafc")
    ax.axhspan(0, y_max, color="#16a34a", alpha=0.055, zorder=0)
    ax.axhspan(y_min, 0, color="#dc2626", alpha=0.045, zorder=0)

    ax.scatter(
        x,
        y,
        s=34,
        marker="o",
        color="#2563eb",
        edgecolors="#ffffff",
        linewidths=0.55,
        alpha=0.72,
        zorder=3,
    )

    profile_x = np.linspace(float(np.quantile(x, 0.03)), float(np.quantile(x, 0.97)), 140)
    _lower, median, _upper = smooth_profile(x, y, profile_x, bandwidth=0.32)
    finite_profile = np.isfinite(median)
    ax.plot(profile_x[finite_profile], median[finite_profile], color="#0f172a", linewidth=2.6, zorder=5)

    x_ticks = np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300], dtype=float)
    x_ticks = x_ticks[(np.log10(x_ticks) >= x_min) & (np.log10(x_ticks) <= x_max)]
    ax.set_xticks(np.log10(x_ticks))
    ax.set_xticklabels([format_ratio_tick(tick) for tick in x_ticks], rotation=25, ha="right")

    y_ticks = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=float)
    y_ticks = y_ticks[(np.log2(y_ticks) >= y_min) & (np.log2(y_ticks) <= y_max)]
    ax.set_yticks(np.log2(y_ticks))
    ax.set_yticklabels([f"{tick:g}x" for tick in y_ticks])
    ax.axhline(0, color="#334155", linewidth=1.4, alpha=0.9, zorder=6)

    ax.set_xlabel("total contexts / fields")
    ax.set_ylabel("speedup = not-grouped time / grouped time")
    ax.set_title("Grouped speedup by total-contexts/fields ratio", fontsize=16, fontweight="bold", pad=12)
    ax.grid(color="#94a3b8", alpha=0.28, linewidth=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.text(
        0.5,
        0.025,
        r"Total contexts: $\sum_{i=1}^{d} n^i$, where $n$ is the number of contexts and $d$ is depth. Black line is a smoothed median.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.17)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = load_results(args.grouped_root, args.not_grouped_root)
    fields_by_graph = load_fields(args.graph_summary)
    points = build_speedup_points(results, fields_by_graph, args.timeout_seconds, args.include_lower_bounds)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.csv_output, index=False)

    plot_speedup(points, args.output)
    print(args.output)
    print(args.csv_output)


if __name__ == "__main__":
    main()
