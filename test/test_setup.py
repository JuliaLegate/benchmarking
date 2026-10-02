"""Exercise launcher failures without downloading Julia packages or using GPUs."""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASH = os.environ.get("BENCH_TEST_BASH") or shutil.which("bash") or "bash"


def shell_path(path):
    # Match Git Bash mounts (including its /tmp alias) on Windows.
    path = path.resolve()
    if os.name == "nt":
        return subprocess.check_output(
            [BASH, "-c", '/usr/bin/cygpath -u "$1"', "bash", path.as_posix()],
            text=True).strip()
    return str(path)


class SetupTests(unittest.TestCase):
    def test_krylov_solver_selection_and_isolation(self):
        script = Path(__file__).resolve().parents[1] / "composability/krylov/run.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            (project / "Manifest.toml").write_text("# test manifest\n")
            shims = root / "bin"
            shims.mkdir()
            commands = {
                "git": 'echo test-commit\n',
                "nvidia-smi": 'echo 123\n',
                "timeout": 'shift 3\nexec "$@"\n',
                "julia": '''for arg in "$@"; do
    case "$arg" in
        *plot_results.jl)
            for option in "$@"; do
                case "$option" in --solver=*) solver=${option#--solver=} ;; esac
            done
            echo "$solver" >> "$BENCH_OUTPUT/plots-called.txt"
            [[ $solver != "$BENCH_TEST_PLOT_FAILURE" ]] || exit 1
            touch "${@: -1}"
            exit 0 ;;
        --version) echo 'julia version test'; exit 0 ;;
        -e) exit 0 ;;
    esac
done
backend=${@: -4:1}; solver=${@: -3:1}; mode=${@: -2:1}; n=${@: -1}
if [[ $solver == cg && ( $BENCH_TEST_FAILURE == all_cg ||
      ( $BENCH_TEST_FAILURE == partial && $backend == Dagger && ( $n == 32 || $BENCH_GPUS == 2 ) ) ) ]]; then
    echo 'simulated solver failure' >&2
    exit 7
fi
label=$backend
[[ $backend != CuArray ]] || label=CUDA
[[ $mode != local ]] || label='cuNumeric local'
echo "RESULT,$label,$solver,$mode,Float32,$BENCH_GPUS,$n,2,1,0.1,1,1,1,0.001,1;1"
''',
            }
            for name, body in commands.items():
                shim = shims / name
                shim.write_text("#!/bin/bash\n" + body, newline="\n")
                shim.chmod(0o755)
            env = os.environ.copy()
            env.update(BENCH_PROJECT=shell_path(project), JULIA=shell_path(shims / "julia"),
                       BENCH_DRY_RUN="0", BENCH_ELTYPE="Float32")
            env.pop("CUDA_VISIBLE_DEVICES", None)
            command = [BASH, "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"',
                       "bash", shell_path(shims), shell_path(script)]
            scenarios = [(selection, mode, "", "", include_local)
                         for selection in ("cg", "bicgstab", "cg,bicgstab")
                         for mode in ("single", "weak") for include_local in (False, True)]
            scenarios += [("cg,bicgstab", mode, "partial", "", True) for mode in ("single", "weak")]
            scenarios += [("bicgstab,cg", "single", "all_cg", "", True),
                          ("cg,bicgstab", "single", "", "cg", True)]
            for index, (selection, mode, failure, plot_failure, include_local) in enumerate(scenarios):
                with self.subTest(selection=selection, mode=mode, failure=failure,
                                  plot_failure=plot_failure, include_local=include_local):
                    output = root / f"run-{index}"
                    env.update(BENCH_OUTPUT=shell_path(output), BENCH_SOLVERS=selection,
                               BENCH_TEST_FAILURE=failure, BENCH_TEST_PLOT_FAILURE=plot_failure)
                    if include_local:
                        env["BENCH_LOCAL"] = "1"
                    else:
                        env.pop("BENCH_LOCAL", None)  # Exercise the shell launcher's default.
                    args = [mode, "16", "32"] if mode == "single" else [mode, "16", "1", "2"]
                    result = subprocess.run(command + args, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, int(bool(failure or plot_failure)), result.stderr)
                    solvers = selection.split(",")
                    labels = (["CUDA"] if mode == "single" else []) + ["Dagger", "cuNumeric"]
                    if include_local:
                        labels.append("cuNumeric local")
                    points = [(1, 16), (1, 32)] if mode == "single" else [(1, 16), (2, 23)]
                    cases = {(label, solver, str(g), str(n)) for label in labels
                             for solver in solvers for g, n in points}
                    expected = {case for case in cases if not (case[1] == "cg" and
                                (failure == "all_cg" or (failure == "partial" and case[0] == "Dagger"
                                 and (case[2] == "2" or case[3] == "32"))))}
                    with (output / "results.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), len(expected))
                    self.assertEqual({(r["backend"], r["solver"], r["gpus"], r["n"]) for r in rows}, expected)
                    for name in ("planned-cases.csv", "memory.csv"):
                        with (output / name).open() as stream:
                            records = list(csv.DictReader(stream))
                        self.assertEqual(len(records), len(cases))
                        self.assertEqual({r["solver"] for r in records}, set(solvers))
                    self.assertEqual(len(list(output.glob("gpu-memory-*.log"))), len(cases))
                    plotted = [solver for solver in solvers if any(r["solver"] == solver for r in rows)]
                    self.assertEqual((output / "plots-called.txt").read_text().splitlines(), plotted)
                    for solver in solvers:
                        image = "timings.png" if len(solvers) == 1 else f"timings-{solver}.png"
                        self.assertEqual((output / image).exists(), solver in plotted and solver != plot_failure)
                    self.assertIn(f"BENCH_SOLVERS={selection}", (output / "environment.txt").read_text())
                    self.assertIn(f"BENCH_LOCAL={int(include_local)}", (output / "environment.txt").read_text())

            for include_local in (0, 1):
                output = root / f"preview-{include_local}"
                env.update(BENCH_OUTPUT=shell_path(output), BENCH_SOLVERS="cg,bicgstab", BENCH_DRY_RUN="1",
                           BENCH_LOCAL=str(include_local))
                result = subprocess.run(command + ["weak", "16", "1", "2"], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len((output / "planned-cases.csv").read_text().splitlines()), 1 + 4 * (2 + include_local))
                self.assertEqual(len((output / "results.csv").read_text().splitlines()), 1)
                self.assertFalse((output / "plots-called.txt").exists())
            for index, selection in enumerate(("", "cg,cg", "gmres", "cg,", "cg,,bicgstab")):
                output = root / f"invalid-{index}"
                env.update(BENCH_OUTPUT=shell_path(output), BENCH_SOLVERS=selection)
                result = subprocess.run(command + ["single", "16"], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(output.exists())
            for index, value in enumerate(("", "2", "true")):
                output = root / f"invalid-local-{index}"
                env.update(BENCH_OUTPUT=shell_path(output), BENCH_SOLVERS="cg", BENCH_LOCAL=value)
                result = subprocess.run(command + ["single", "16"], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(output.exists())

    def test_sweeps_continue_after_backend_failure(self):
        composability = Path(__file__).resolve().parents[1] / "composability"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            (project / "Manifest.toml").write_text("# test manifest\n")
            shims = root / "bin"
            shims.mkdir()
            commands = {
                "git": 'echo test-commit\n',
                "nvidia-smi": 'echo 123\n',
                "timeout": 'shift 3\nexec "$@"\n',
                "julia": '''for arg in "$@"; do
    case "$arg" in
        *plot_results.jl) touch "$BENCH_TEST_OUTPUT/plot-called"; exit 0 ;;
        --version) echo 'julia version test'; exit 0 ;;
        -e) exit 0 ;;
    esac
done
if [[ $BENCH_TEST_WORKLOAD == krylov ]]; then
    backend=${@: -4:1}; mode=${@: -2:1}
else
    backend=${@: -2:1}; mode=stock
fi
n=${@: -1}
label=$backend
[[ $backend != CuArray ]] || label=CUDA
[[ $mode != local ]] || label='cuNumeric local'
if [[ $BENCH_TEST_BACKEND == all_sizes ||
      ( ( $label == "$BENCH_TEST_BACKEND" || $BENCH_TEST_BACKEND == all ) && ( $n == 32 || $n == 64 ) ) ]]; then
    if [[ $BENCH_TEST_FAILURE == exit ]]; then
        echo 'simulated backend allocation failure' >&2
        exit 7
    fi
    echo 'simulated missing result'
    exit 0
fi
case "$BENCH_TEST_WORKLOAD" in
    krylov) echo "RESULT,$label,cg,$mode,Float32,1,$n,2,1,0.1,1,1,1,0.001,1;1" ;;
    ordinarydiffeq) echo "RESULT,$label,Float32,1,$n,20,1,0.1,1,1,1,0.001,1;1" ;;
    integrals_optimization) echo "RESULT,$label,Float32,1,$n,4,12,80,32,1,0.1,1,1,1,1,0,0.001,1;1" ;;
esac
''',
            }
            for name, body in commands.items():
                shim = shims / name
                shim.write_text("#!/bin/bash\n" + body, newline="\n")
                shim.chmod(0o755)
            env = os.environ.copy()
            env.update(JULIA=shell_path(shims / "julia"), BENCH_SOLVERS="cg", BENCH_LOCAL="1")
            env.pop("CUDA_VISIBLE_DEVICES", None)
            for workload, prefix, launcher in (
                ("krylov", "BENCH", "run.sh"),
                ("ordinarydiffeq", "ODE", "run_benchmark.sh"),
                ("integrals_optimization", "INTOPT", "run_benchmark.sh"),
            ):
                script = composability / workload / launcher
                labels = ["CUDA", "Dagger", "cuNumeric"]
                if workload == "krylov":
                    labels.append("cuNumeric local")
                scenarios = labels + (["all", "all_sizes"] if workload == "ordinarydiffeq" else [])
                for failed_backend in scenarios:
                    for failure in (("exit",) if failed_backend in ("all", "all_sizes") else ("exit", "missing_result")):
                        with self.subTest(workload=workload, backend=failed_backend, failure=failure):
                            output = root / f"{workload}-{failed_backend}-{failure}"
                            env.update({f"{prefix}_PROJECT": shell_path(project),
                                        f"{prefix}_OUTPUT": shell_path(output),
                                        f"{prefix}_DRY_RUN": "0", f"{prefix}_ELTYPE": "Float32",
                                        f"{prefix}_BACKENDS": "CuArray Dagger cuNumeric"})
                            env.update(BENCH_TEST_OUTPUT=shell_path(output), BENCH_TEST_WORKLOAD=workload,
                                       BENCH_TEST_FAILURE=failure, BENCH_TEST_BACKEND=failed_backend)
                            result = subprocess.run(
                                [BASH, "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"',
                                 "bash", shell_path(shims), shell_path(script), "single", "16", "32", "48", "64"],
                                env=env, capture_output=True, text=True)
                            self.assertEqual(result.returncode, 1, result.stderr)
                            with (output / "results.csv").open() as stream:
                                rows = list(csv.DictReader(stream))
                            expected = {(backend, str(n)) for backend in labels for n in (16, 32, 48, 64)
                                        if failed_backend != "all_sizes"
                                        and not (failed_backend in (backend, "all") and n in (32, 64))}
                            self.assertEqual(len(rows), len(expected))
                            size_key = "n" if workload == "krylov" else "N"
                            self.assertEqual({(row["backend"], row[size_key]) for row in rows}, expected)
                            # Only the plume launcher records the largest fully passing baseline.
                            if workload != "integrals_optimization":
                                self.assertFalse((output / "base_n.txt").exists())
                            elif expected:
                                self.assertEqual((output / "base_n.txt").read_text().strip(), "48")
                            else:
                                self.assertFalse((output / "base_n.txt").exists())
                            self.assertEqual((output / "plot-called").is_file(), bool(expected))
                            for name in ("planned-cases.csv", "memory.csv"):
                                self.assertEqual(len((output / name).read_text().splitlines()), 1 + 4 * len(labels))
                            backend = ("CuArray" if failed_backend in ("CUDA", "all", "all_sizes")
                                       else failed_backend.split()[0])
                            mode = "local" if failed_backend == "cuNumeric local" else "stock"
                            log = (f"Float32-{backend}-cg-{mode}-1-64.log" if workload == "krylov"
                                   else f"{backend}-1-64.log")
                            self.assertIn("simulated", (output / log).read_text())
                            self.assertIn("continuing the sweep", result.stderr)

    def test_krylov_collects_results_before_plotting(self):
        script = Path(__file__).resolve().parents[1] / "composability/krylov/run.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            (project / "Manifest.toml").write_text("# test manifest\n")
            shims = root / "bin"
            shims.mkdir()
            commands = {
                "git": 'echo test-commit\n',
                "nvidia-smi": 'echo 123\n',
                "timeout": 'shift 3\nexec "$@"\n',
                "julia": '''for arg in "$@"; do
    case "$arg" in *plot_results.jl) echo 'simulated plotting error' >&2; exit 1 ;; esac
done
case "$1" in
--version) echo 'julia version test' ;;
--startup-file=no) exit 0 ;;
*) echo 'RESULT,Dagger,cg,stock,Float32,1,16,2,1,0.1,1,1,1,0.001,1;1' ;;
esac
''',
            }
            for name, body in commands.items():
                shim = shims / name
                shim.write_text("#!/bin/bash\n" + body, newline="\n")
                shim.chmod(0o755)
            env = os.environ.copy()
            env.update(BENCH_PROJECT=shell_path(project), JULIA=shell_path(shims / "julia"),
                       BENCH_DRY_RUN="0", BENCH_ELTYPE="Float32", BENCH_SOLVERS="cg", BENCH_LOCAL="0")
            env.pop("CUDA_VISIBLE_DEVICES", None)
            for mode, args in (("single", ["16"]), ("weak", ["16", "1"])):
                with self.subTest(mode=mode):
                    output = root / mode
                    env["BENCH_OUTPUT"] = shell_path(output)
                    # Set PATH inside Bash so this also works with Git Bash on Windows.
                    result = subprocess.run(
                        [BASH, "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"',
                         "bash", shell_path(shims), shell_path(script), mode, *args],
                        env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn("Plot generation failed", result.stderr)
                    self.assertIn("Running Dagger", result.stdout)
                    self.assertGreater(len((output / "results.csv").read_text().splitlines()), 1)
                    self.assertGreater(len((output / "planned-cases.csv").read_text().splitlines()), 1)
