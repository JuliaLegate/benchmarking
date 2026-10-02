#!/usr/bin/env python3
"""Condensed cuNumeric comparisons from the existing LOC summary (PNG/PDF)."""

import argparse
import csv
from pathlib import Path

from plot_model_loc import (
    BENCHMARK_ORDER, COLORS, METRICS, PLOT_VARIANT_LABEL, PLOT_VARIANT_ORDER,
    plot_rows, read_rows,
)

LABELS = {"sloc": "Source Lines of Code", "uloc": "Unique Lines of Code",
          "complexity": "Cyclomatic Complexity"}
HATCHES = {"sloc": "", "uloc": "///", "complexity": "..."}


def comparisons(rows, benchmarks):
    values = {(r["benchmark"], r["variant"]): r for r in rows}
    results = []
    for variant in PLOT_VARIANT_ORDER:
        if variant == "cunumeric":
            continue
        shared = [b for b in benchmarks if (b, "cunumeric") in values and (b, variant) in values]
        if not shared:
            continue
        for metric, (field, _) in METRICS.items():
            subject = [int(values[b, "cunumeric"][field]) for b in shared]
            reference = [int(values[b, variant][field]) for b in shared]
            if any(v < 0 for v in subject + reference):
                raise ValueError("Metric values must be nonnegative")
            change = 100 * (sum(reference) / sum(subject) - 1) if sum(subject) else None
            results.append(dict(variant=variant, metric=metric, benchmark_count=len(shared),
                                benchmarks=";".join(shared), cunumeric_total=sum(subject),
                                reference_total=sum(reference), change_relative_to_cunumeric_pct=change))
    return results


def group_label(variant):
    return "CUDA C++\n(NAS benchmarks)" if variant == "cuda" else PLOT_VARIANT_LABEL[variant]


def draw_bars(results):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    variants = list(dict.fromkeys(r["variant"] for r in results))
    lookup = {(r["variant"], r["metric"]): r for r in results}
    field = "change_relative_to_cunumeric_pct"
    metrics = list(METRICS)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    width = 0.72 / len(metrics)
    for i, variant in enumerate(variants):
        for j, metric in enumerate(metrics):
            x = i + (j - (len(metrics) - 1) / 2) * width
            value = lookup[variant, metric][field]
            if value is None:
                ax.annotate("N/A", (x, 0), xytext=(0, 5), textcoords="offset points",
                            ha="center", color=COLORS[variant], fontsize=9)
                continue
            bars = ax.bar(x, value, width, color=COLORS[variant], edgecolor="white",
                          linewidth=1, hatch=HATCHES[metric])
            ax.bar_label(bars, labels=[f"{value:+.1f}%"], padding=5, fontsize=9, rotation=90)
    ax.set_xticks(range(len(variants)), [group_label(v) for v in variants],
                  fontsize=12, fontweight="bold", color="black")
    ax.set_ylabel("Percent Difference in Total Counts vs cuNumeric.jl", fontsize=14)
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    available = [r[field] for r in results if r["metric"] in metrics and r[field] is not None]
    low, high = min([0, *available]), max([0, *available])
    pad = max(8, (high - low) * 0.18)
    ax.set_ylim(low - pad, high + pad)
    handles = [Patch(facecolor="#777777", edgecolor="white", hatch=HATCHES[m], label=LABELS[m]) for m in metrics]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.985),
              fontsize=13, labelspacing=0.7, frameon=False)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results" / "summary.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARK_ORDER, default=BENCHMARK_ORDER)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = comparisons(plot_rows(read_rows(args.input)), args.benchmarks)
    if not results:
        parser.error("No benchmarks shared by cuNumeric.jl and another backend")
    out = args.output_dir or args.input.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    fig = draw_bars(results)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"comparison_pooled.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote comparison_pooled (PNG/PDF) and comparisons.csv to {out}")


if __name__ == "__main__":
    main()
