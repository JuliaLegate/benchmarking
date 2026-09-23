#!/usr/bin/env python3
"""Plot synchronized Krylov solve times from run.sh CSV files."""

import argparse
import csv
import math
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
solvers = [solver for solver in ("cg", "bicgstab")
           if any(row["solver"] == solver for row in rows)]
if not solvers or any(row["solver"] not in solvers for row in rows):
    parser.error("Expected CG and/or BiCGSTAB results")
fig, axes = plt.subplots(1, len(solvers), figsize=(5 * len(solvers), 4),
                         sharey=True, constrained_layout=True, squeeze=False)
for ax, solver in zip(axes[0], solvers):
    series = defaultdict(dict)
    for row in rows:
        if row["solver"] != solver:
            continue
        x = int(row["n"] if args.experiment == "single" else row["gpus"])
        key = row["backend"]
        if key not in expected:
            parser.error(f"Unexpected backend: {key}")
        if x in series[key]:
            parser.error(f"Duplicate {solver} {key} point at {x}")
        series[key][x] = float(row["median_ms"])
    points = sorted({x for values in series.values() for x in values})
    for key in expected:
        if key in series:
            # NaNs mark failed/missing cases without connecting across them.
            ax.plot(points, [series[key].get(x, math.nan) for x in points],
                    marker="o", label=key, color=colors[key])
    ax.set_title("CG" if solver == "cg" else "BiCGSTAB")
    ax.set_xlabel("Matrix dimension N" if args.experiment == "single" else "GPUs")
    ax.set_xscale("log", base=2)
    if args.experiment == "weak":
        ax.set_xticks(points, [str(x) for x in points])
    ax.grid(alpha=0.25)
axes[0][0].set_ylabel("Median solve time (ms)")
axes[0][0].set_yscale("log")
handles = {}
for ax in axes[0]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles[label] = handle
fig.legend([handles[key] for key in expected if key in handles],
           [key for key in expected if key in handles], loc="lower center",
           bbox_to_anchor=(0.5, -0.08), ncol=len(handles))
subtitle = f"{next(iter(eltypes))}"
if args.experiment == "weak":
    subtitle += f", N(1)={next(iter(base_ns))}; N(G)≈N(1)√G"
fig.suptitle(("Single GPU" if args.experiment == "single" else "Weak scaling") +
             f" — {subtitle}")
fig.savefig(args.output, dpi=180, bbox_inches="tight")
print(args.output)
