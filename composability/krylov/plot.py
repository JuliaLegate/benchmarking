#!/usr/bin/env python3
"""Plot mean synchronized CG times with standard errors."""

import argparse
import csv
import math
import statistics

import matplotlib.pyplot as plt


def samples(row):
    values = [float(value) for value in row["samples_ms"].split(";")]
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Every plotted point needs at least two positive timing samples")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("single", "weak"))
    parser.add_argument("csv", nargs="+", help="One or more run.sh results.csv files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = (["CUDA", "Dagger", "cuNumeric", "cuNumeric local"]
                if args.experiment == "single" else
                ["Dagger", "cuNumeric", "cuNumeric local"])
    colors = {"CUDA": "#4c78a8", "Dagger": "#f58518",
              "cuNumeric": "#54a24b", "cuNumeric local": "#b279a2"}
    rows = []
    for path in args.csv:
        with open(path, newline="", encoding="utf-8") as stream:
            rows.extend(row for row in csv.DictReader(stream)
                        if row["experiment"] == args.experiment and row["solver"] == "cg")
    if not rows:
        parser.error("No CG results for the requested experiment")
    if len({row["eltype"] for row in rows}) != 1:
        parser.error("Plot one element type at a time")
    base_ns = {row["base_n"] for row in rows} if args.experiment == "weak" else set()
    if len(base_ns) > 1:
        parser.error("Plot one weak-scaling base N at a time")

    series = {key: {} for key in expected}
    for row in rows:
        key = row["backend"]
        if key not in series:
            parser.error(f"Unexpected backend: {key}")
        x = int(row["n"] if args.experiment == "single" else row["gpus"])
        if x in series[key]:
            parser.error(f"Duplicate {key} point at {x}")
        try:
            series[key][x] = samples(row)
        except ValueError as exc:
            parser.error(f"{key} at {x}: {exc}")

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for key, points in series.items():
        if not points:
            continue
        xs = sorted(points)
        ax.errorbar(xs, [points[x][0] for x in xs],
                    yerr=[points[x][1] for x in xs], marker="o", capsize=3,
                    label=key, color=colors[key])
        if args.experiment == "weak" and 1 in points:
            ax.axhline(points[1][0], linestyle="--", linewidth=1,
                       alpha=0.45, color=colors[key])
    ax.set_xlabel("Matrix dimension N" if args.experiment == "single" else "GPUs")
    ax.set_ylabel("Mean complete solve time (ms) ± standard error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if args.experiment == "single":
        dimensions = sorted({x for points in series.values() for x in points})
        ax.set_xticks(dimensions, [str(x) for x in dimensions])
    else:
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.grid(alpha=0.25)
    ax.legend()
    subtitle = next(iter({row["eltype"] for row in rows}))
    if args.experiment == "weak":
        subtitle += f", N(1)={next(iter(base_ns))}; N(G)≈N(1)√G"
    ax.set_title(f"CG — {args.experiment} GPU scaling — {subtitle}")
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
