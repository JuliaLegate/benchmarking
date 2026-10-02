#!/usr/bin/env python3
"""Plot grouped bars from analyze_model_loc.py's summary.csv (PNG and PDF)."""

import argparse
import csv
from pathlib import Path

from analyze_model_loc import BENCHMARK_ORDER, VARIANT_LABEL, VARIANT_ORDER

METRICS = {"sloc": ("scc_code", "Source lines of code"),
           "uloc": ("scc_uloc", "Unique source lines"),
           "complexity": ("scc_complexity", "scc complexity estimate")}
# Match plot_results.jl's benchmark palette.
COLORS = {"cunumeric": "#2a78d6", "cupynumeric": "#eb6834", "cudajl": "#1a7f37",
          "jacc": "#8e44ad", "dagger": "#c49a00", "cuda": "#444444"}
PLOT_VARIANT_ORDER = tuple(v for v in VARIANT_ORDER if v != "cuda_cufft")
PLOT_VARIANT_LABEL = {**VARIANT_LABEL, "cuda": "CUDA C++ (FT: cuFFT)"}
BENCHMARK_LABEL = {"gemm": "GEMM", "montecarlo": "Monte Carlo", "grayscott": "Gray–Scott",
                   "cg": "CG", "nas_ep": "NAS EP", "nas_ft": "NAS FT", "nas_mg": "NAS MG"}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        key = row["benchmark"], row["variant"]
        if key in seen:
            raise ValueError(f"Duplicate benchmark/variant in CSV: {key}")
        if key[0] not in BENCHMARK_ORDER or key[1] not in VARIANT_ORDER:
            raise ValueError(f"Unknown benchmark/variant: {key}")
        seen.add(key)
    return rows


def plot_rows(rows):
    """Use CUDA EP/MG and cuFFT FT as one C++ series; retain the raw CSV."""
    return [{**r, "variant": "cuda" if r["variant"] == "cuda_cufft" else r["variant"]}
            for r in rows if (r["benchmark"], r["variant"]) != ("nas_ft", "cuda")]


def draw_chart(rows, metric, benchmarks, variants):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    field, label = METRICS[metric]
    values = {(r["benchmark"], r["variant"]): int(r[field]) for r in rows}
    if any(v < 0 for v in values.values()):
        raise ValueError("Metric values must be nonnegative")
    fig, ax = plt.subplots(figsize=(max(8, len(benchmarks) * 1.8), 5.5))
    width = 0.8 / len(variants)
    for index, variant in enumerate(variants):
        present = [(i, values[b, variant]) for i, b in enumerate(benchmarks) if (b, variant) in values]
        positions = [i + (index - (len(variants) - 1) / 2) * width for i, _ in present]
        bars = ax.bar(positions, [v for _, v in present], width, label=PLOT_VARIANT_LABEL[variant],
                      color=COLORS[variant], edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, padding=3, fontsize=8, rotation=90 if len(variants) > 4 else 0)
        for i, benchmark in enumerate(benchmarks):
            if (benchmark, variant) not in values:
                position = i + (index - (len(variants) - 1) / 2) * width
                ax.annotate("×", (position, 0), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", color=COLORS[variant], fontsize=12,
                            fontweight="bold")
    ax.set_xticks(range(len(benchmarks)), [BENCHMARK_LABEL[b] for b in benchmarks])
    ax.set_ylabel(label)
    ax.set_title(f"Reference implementations · {label}", loc="left", pad=58, fontweight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=min(4, len(variants)), frameon=False, fontsize=9)
    ax.set_ylim(bottom=0, top=max(1, max(values.values())) * 1.28)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results" / "summary.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=["sloc", "uloc"])
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARK_ORDER, default=BENCHMARK_ORDER)
    parser.add_argument("--variants", nargs="+", choices=PLOT_VARIANT_ORDER, default=PLOT_VARIANT_ORDER,
                        help="Models to plot; cuda uses EP/MG CUDA refs and the FT cuFFT ref.")
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [r for r in plot_rows(read_rows(args.input)) if r["benchmark"] in args.benchmarks and r["variant"] in args.variants]
        if not rows:
            raise ValueError("No results match the requested selection")
        benchmarks = [b for b in BENCHMARK_ORDER if any(r["benchmark"] == b for r in rows)]
        variants = [v for v in PLOT_VARIANT_ORDER if any(r["variant"] == v for r in rows)]
        out = args.output_dir or args.input.parent / "plots"
        out.mkdir(parents=True, exist_ok=True)
        for metric in dict.fromkeys(args.metrics):
            fig = draw_chart(rows, metric, benchmarks, variants)
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"{metric}.{suffix}", dpi=180, bbox_inches="tight")
            plt.close(fig)
        print(f"Wrote {', '.join(dict.fromkeys(args.metrics))} charts (PNG and PDF) to {out}")
    except (ImportError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\nInstall plotting dependencies with python -m pip install -r loc-analysis/requirements.txt\n")


if __name__ == "__main__":
    main()
