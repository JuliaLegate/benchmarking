#!/usr/bin/env python3
"""Compare lines of code across the Julia/Python/CUDA GPU programming models.

Counts the curated refs/<benchmark>/<model>.<ext>.ref files (see README.md)
with scc v4, then writes summary.csv, overall_metrics.csv, and report.md.
Refs are first normalized (temporary copies) with JuliaFormatter / black at an
unbounded margin, so every statement sits on one line in both languages and
counts do not depend on wrapping style. CUDA C++ refs are preformatted separately
and copied unchanged.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import namedtuple
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REFS_DIR = HERE / "refs"
DEFAULT_OUTPUT_DIR = HERE / "results"
VENDORED_SCC = REPO_ROOT / "opt" / "scc" / "bin" / "scc"

BENCHMARK_ORDER = ("gemm", "montecarlo", "grayscott", "cg", "nas_ep", "nas_ft", "nas_mg")
VARIANT_ORDER = ("cunumeric", "cupynumeric", "cudajl", "jacc", "dagger", "cuda", "cuda_cufft")
VARIANT_LABEL = {
    "cunumeric": "cuNumeric.jl",
    "cupynumeric": "cuPyNumeric",
    "cudajl": "CUDA.jl",
    "jacc": "JACC.jl",
    "dagger": "Dagger.jl",
    "cuda": "CUDA C++",
    "cuda_cufft": "CUDA C++ with cuFFT",
}
VARIANT_SUFFIX = {v: "py" if v == "cupynumeric" else "jl" for v in VARIANT_ORDER}
SCC_LANGUAGE = {v: "Python" if v == "cupynumeric" else "Julia" for v in VARIANT_ORDER}
for _variant in ("cuda", "cuda_cufft"):
    VARIANT_SUFFIX[_variant] = "cu"
    SCC_LANGUAGE[_variant] = "Cuda"
VARIANT_BENCHMARKS = {v: set(BENCHMARK_ORDER) for v in VARIANT_ORDER}
VARIANT_BENCHMARKS["cuda"] = {"nas_ep", "nas_ft", "nas_mg"}
VARIANT_BENCHMARKS["cuda_cufft"] = {"nas_ft"}
SINGLE_GPU_ONLY = {"cudajl", "cuda", "cuda_cufft"}
FORMAT_MARGIN = 10_000  # effectively unbounded: one statement per line

# (subject, reference): percent less code in subject than in reference.
COMPARISONS = (
    ("cunumeric", "cudajl"),
    ("cunumeric", "jacc"),
    ("cunumeric", "dagger"),
    ("cunumeric", "cupynumeric"),
    ("cunumeric", "cuda"),
    ("cunumeric", "cuda_cufft"),
)

Column = namedtuple("Column", "scc_key field label")
SCC_COLUMNS = (
    Column("Code", "scc_code", "scc SLOC"),
    Column("Lines", "scc_lines", "Lines"),
    Column("Comment", "scc_comment", "Comment"),
    Column("Blank", "scc_blank", "Blank"),
    Column("Complexity", "scc_complexity", "Complexity"),
    Column("Cognitive", "scc_cognitive", "Cognitive"),
    Column("Uloc", "scc_uloc", "ULOC"),
)

MetricView = namedtuple("MetricView", "field label prefix")
VIEWS = (
    MetricView("scc_code", "scc SLOC", "scc_code_"),
    MetricView("scc_uloc", "scc ULOC", "scc_uloc_"),
)


def ref_path(benchmark: str, variant: str) -> Path:
    return REFS_DIR / benchmark / f"{variant}.{VARIANT_SUFFIX[variant]}.ref"


def scc_version(binary: str) -> str:
    out = subprocess.run([binary, "--version"], check=True, capture_output=True, text=True)
    return out.stdout.strip()


JULIA_FORMAT = f"""
using Pkg; Pkg.instantiate(; io=devnull)
using JuliaFormatter
println("JuliaFormatter ", pkgversion(JuliaFormatter))
foreach(f -> format_file(f; margin={FORMAT_MARGIN}, join_lines_based_on_source=false), ARGS)
"""


def format_copies(targets: list[tuple[str, Path]], workdir: Path) -> tuple[dict[Path, Path], str]:
    """Format temporary Julia/Python copies; preserve preformatted CUDA C++ bytes."""
    copies = {}
    for variant, path in targets:
        dest = workdir / path.parent.name / f"{variant}.{VARIANT_SUFFIX[variant]}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        copies[path] = dest
    jl = [str(p) for p in copies.values() if p.suffix == ".jl"]
    py = [str(p) for p in copies.values() if p.suffix == ".py"]
    versions = []
    if jl:
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        julia = os.environ.get("CUNUMERIC_BENCH_JULIA", "julia")
        out = subprocess.run([julia, "--startup-file=no", f"--project={HERE}", "-e", JULIA_FORMAT, *jl],
                             check=True, capture_output=True, text=True, env=env)
        versions.append(out.stdout.strip().splitlines()[-1])
    if py:
        black = [sys.executable, "-m", "black", "-q", "-l", str(FORMAT_MARGIN),
                 "--skip-magic-trailing-comma"]
        subprocess.run([*black, *py], check=True)
        out = subprocess.run([sys.executable, "-m", "black", "--version"],
                             check=True, capture_output=True, text=True)
        versions.append(out.stdout.split(" (")[0].replace(", ", " "))
    return copies, " and ".join(versions) + " (one statement per line)"


def run_scc(binary: str, targets: list[tuple[str, Path]]) -> dict[Path, dict]:
    """Run scc once per language over `targets` ((variant, path) pairs)."""
    groups: dict[str, list[Path]] = {}
    for variant, path in targets:
        groups.setdefault(SCC_LANGUAGE[variant], []).append(path)
    results = {}
    for language, group in groups.items():
        cmd = [binary, "--by-file", "--format", "json", "--no-cocomo", "--uloc",
               "--cognitive", *(str(p) for p in group)]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        for entry in json.loads(out.stdout):
            for file in entry.get("Files", []):
                results[Path(file["Location"]).resolve()] = file
    return results


def summarize(benchmarks, variants, scc_binary: str, workdir: Path) -> tuple[list[dict], str]:
    targets = [(v, ref_path(b, v)) for b in benchmarks for v in variants if b in VARIANT_BENCHMARKS[v]]
    if not targets:
        raise SystemExit("No references cover the requested benchmark/variant selection.")
    missing = [str(p) for _, p in targets if not p.is_file()]
    if missing:
        raise SystemExit("Missing ref files:\n  " + "\n  ".join(missing))
    copies, formatter = format_copies(targets, workdir)
    scc = run_scc(scc_binary, [(v, copies[p]) for v, p in targets])
    rows = []
    for benchmark in benchmarks:
        for variant in variants:
            if benchmark not in VARIANT_BENCHMARKS[variant]:
                continue
            path = ref_path(benchmark, variant)
            file = scc.get(copies[path].resolve(), {})
            row = {
                "benchmark": benchmark,
                "variant": variant,
                "path": str(path.relative_to(HERE)),
            }
            row.update({c.field: int(file.get(c.scc_key, 0)) for c in SCC_COLUMNS})
            rows.append(row)
    return rows, formatter


def pct_reduction(reference: int, subject: int) -> float:
    return 100.0 * (1.0 - (subject / reference)) if reference else float("nan")


def aggregate(rows: list[dict], benchmarks, variants, view: MetricView) -> dict:
    value = {(r["benchmark"], r["variant"]): r[view.field] for r in rows}
    metrics = {f"{view.prefix}benchmark_count": len(benchmarks)}
    for variant in variants:
        available = [b for b in benchmarks if (b, variant) in value]
        metrics[f"{view.prefix}{variant}_benchmark_count"] = len(available)
        metrics[f"{view.prefix}total_{variant}_loc"] = sum(value[b, variant] for b in available)
    for subject, reference in COMPARISONS:
        if subject not in variants or reference not in variants:
            continue
        shared = [b for b in benchmarks if (b, subject) in value and (b, reference) in value]
        if not shared:
            continue
        total_ref = sum(value[b, reference] for b in shared)
        total_sub = sum(value[b, subject] for b in shared)
        key = f"{subject}_vs_{reference}_pct"
        metrics[f"{view.prefix}{subject}_vs_{reference}_benchmarks"] = ";".join(shared)
        metrics[f"{view.prefix}total_{key}"] = round(pct_reduction(total_ref, total_sub), 2)
        metrics[f"{view.prefix}mean_{key}"] = round(mean(
            pct_reduction(value[b, reference], value[b, subject]) for b in shared
        ), 2)
    return metrics


def metric_table(title: str, metrics: dict, view: MetricView, variants) -> str:
    p = view.prefix
    lines = [f"### {title}", "", "| Model | Available benchmarks | Total |", "|---|---:|---:|"]
    for variant in variants:
        count = metrics[f"{p}{variant}_benchmark_count"]
        if count:
            lines.append(f"| {VARIANT_LABEL[variant]} | {count} | {metrics[f'{p}total_{variant}_loc']} |")
    lines += ["", "Totals cover each model's available refs; reductions use only shared benchmarks.", "",
              "| cuNumeric.jl vs | Shared benchmarks | Pooled reduction | Mean per-benchmark reduction |",
              "|---|---|---:|---:|"]
    for subject, reference in COMPARISONS:
        key = f"{subject}_vs_{reference}_pct"
        if f"{p}total_{key}" in metrics:
            shared = metrics[f"{p}{subject}_vs_{reference}_benchmarks"]
            lines.append(f"| {VARIANT_LABEL[reference]} | {shared} | {metrics[f'{p}total_{key}']:.1f}% "
                         f"| {metrics[f'{p}mean_{key}']:.1f}% |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows, benchmarks, variants, version, formatter) -> str:
    parts = [
        "# Programming-model LOC analysis",
        "",
        (f"Counted with {version} after formatting with {formatter}."
         if any(VARIANT_SUFFIX[v] in ("jl", "py") for v in variants)
         else f"Counted with {version}."),
        "Refs follow the inclusion rules in `README.md`.",
        "CUDA.jl and both CUDA C++ variants are single-GPU references.",
        "CUDA C++ refs are preformatted separately and counted unchanged.",
        "",
    ]
    for view in VIEWS:
        metrics = aggregate(rows, benchmarks, variants, view)
        parts += [metric_table(f"Overall ({view.label})", metrics, view, variants), ""]
    parts += ["## Per benchmark", ""]
    for benchmark in benchmarks:
        if not any(r["benchmark"] == benchmark for r in rows):
            continue
        parts += [f"### {benchmark}", "",
                  "| Model | scc SLOC | scc ULOC | scc Complexity | scc Cognitive | File |",
                  "|---|---:|---:|---:|---:|---|"]
        for r in rows:
            if r["benchmark"] == benchmark:
                parts.append(
                    f"| {VARIANT_LABEL[r['variant']]} | {r['scc_code']} | {r['scc_uloc']} "
                    f"| {r['scc_complexity']} | {r['scc_cognitive']} | `{r['path']}` |"
                )
        parts.append("")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARK_ORDER, default=BENCHMARK_ORDER)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER, default=VARIANT_ORDER)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scc-bin", default=str(VENDORED_SCC) if VENDORED_SCC.is_file() else "scc")
    args = parser.parse_args()

    benchmarks = [b for b in BENCHMARK_ORDER if b in args.benchmarks]
    variants = [v for v in VARIANT_ORDER if v in args.variants]
    version = scc_version(args.scc_bin)
    if not re.search(r"\bv?([4-9]|\d{2,})\.", version):
        raise SystemExit(f"{version} lacks --uloc/--cognitive; run ./install_scc.sh (v4.0.0).")

    with tempfile.TemporaryDirectory() as workdir:
        rows, formatter = summarize(benchmarks, variants, args.scc_bin, Path(workdir))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", rows, list(rows[0]))
    overall = {}
    for view in VIEWS:
        overall.update(aggregate(rows, benchmarks, variants, view))
    write_csv(out / "overall_metrics.csv", [overall], list(overall))
    (out / "report.md").write_text(build_report(rows, benchmarks, variants, version, formatter))
    print(f"Wrote {out / 'summary.csv'}, {out / 'overall_metrics.csv'}, {out / 'report.md'}")


if __name__ == "__main__":
    main()
