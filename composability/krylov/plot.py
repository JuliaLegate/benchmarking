#!/usr/bin/env python3
"""Plot synchronized Krylov solve times from run.sh CSV files."""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("experiment", choices=("single", "weak"))
parser.add_argument("csv", nargs="+", help="One or more run.sh results.csv files")
parser.add_argument("--output", required=True, help="Output PNG, PDF, or SVG path")
args = parser.parse_args()

expected = (["CUDA", "Dagger", "cuNumeric", "cuNumeric local"]
            if args.experiment == "single" else
            ["Dagger", "cuNumeric", "cuNumeric local"])
rows = []
for path in args.csv:
    with open(path, newline="", encoding="utf-8") as stream:
        rows.extend(row for row in csv.DictReader(stream)
                    if row["experiment"] == args.experiment)
if not rows:
    parser.error(f"No {args.experiment} results in the supplied CSV files")
eltypes = {row["eltype"] for row in rows}
if len(eltypes) != 1:
    parser.error("Plot each element type separately")
base_ns = {row["base_n"] for row in rows} if args.experiment == "weak" else set()
if len(base_ns) > 1:
    parser.error("Plot each weak-scaling base N separately")

colors = {"CUDA": "#4c78a8", "Dagger": "#f58518",
          "cuNumeric": "#54a24b", "cuNumeric local": "#b279a2"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
for ax, solver in zip(axes, ("cg", "bicgstab")):
    series = defaultdict(dict)
    for row in rows:
        if row["solver"] != solver:
            continue
        x = int(row["n"] if args.experiment == "single" else row["gpus"])
        key = row["backend"]
        if x in series[key]:
            parser.error(f"Duplicate {solver} {key} point at {x}")
        series[key][x] = float(row["median_ms"])
    if set(series) != set(expected):
        parser.error(f"{solver}: expected {expected}, found {list(series)}")
    points = set(series[expected[0]])
    if any(set(series[key]) != points for key in expected):
        parser.error(f"{solver}: backend series have different x values")
    for key in expected:
        xy = sorted(series[key].items())
        ax.plot([x for x, _ in xy], [y for _, y in xy], marker="o",
                label=key, color=colors[key])
    ax.set_title("CG" if solver == "cg" else "BiCGSTAB")
    ax.set_xlabel("Matrix dimension N" if args.experiment == "single" else "GPUs")
    ax.set_xscale("log", base=2)
    if args.experiment == "weak":
        ax.set_xticks(sorted(points), [str(x) for x in sorted(points)])
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Median solve time (ms)")
axes[0].set_yscale("log")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
           bbox_to_anchor=(0.5, -0.08), ncol=len(expected))
subtitle = f"{next(iter(eltypes))}"
if args.experiment == "weak":
    subtitle += f", N(1)={next(iter(base_ns))}; N(G)≈N(1)√G"
fig.suptitle(("Single GPU" if args.experiment == "single" else "Weak scaling") +
             f" — {subtitle}")
fig.savefig(args.output, dpi=180, bbox_inches="tight")
print(args.output)
