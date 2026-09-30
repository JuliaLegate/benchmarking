"""Exercise setup and launcher failures without downloading Julia packages or using GPUs."""
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
            env.update(JULIA=shell_path(shims / "julia"))
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
                            # Later complete success advances the baseline; partial success does not.
                            if expected:
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
                       BENCH_DRY_RUN="0", BENCH_ELTYPE="Float32")
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

    def test_source_selection(self):
        script = Path(__file__).resolve().parents[1] / "instantiate_projects.sh"
        for standalone in (False, True):
            with self.subTest(standalone=standalone), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "cuNumeric checkout"
                (source / "lib/CNPreferences").mkdir(parents=True)
                (source / "Project.toml").touch()
                (source / "lib/CNPreferences/Project.toml").touch()
                bench = root / "benchmarking" if standalone else source / "benchmark"
                bench.mkdir()
                (bench / script.name).write_text(script.read_text(), newline="\n")
                project = bench / "environments/composability/Project.toml"
                project.parent.mkdir(parents=True)
                project.write_text((script.parent / "environments/composability/Project.toml").read_text(), newline="\n")
                log = root / "args"
                fake = root / "julia"
                fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$SETUP_TEST_LOG"\n', newline="\n")
                fake.chmod(0o755)
                env = {k: v for k, v in os.environ.items()
                       if k != "CUNUMERIC_SOURCE"}
                env.update(CUNUMERIC_BENCH_JULIA=shell_path(fake), SETUP_TEST_LOG=shell_path(log))
                if standalone:
                    # Relative overrides are relative to the caller, not the script.
                    env["CUNUMERIC_SOURCE"] = source.name
                result = subprocess.run([BASH, shell_path(bench / script.name)], cwd=root,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = log.read_text().splitlines()
                self.assertEqual(args[-2:],
                                 [shell_path(source), shell_path(source / "lib/CNPreferences")])
                self.assertEqual(args.count("--project=environments/composability"), 1)
                self.assertFalse(any("setup.jl" in arg for arg in args))
                self.assertFalse((bench / "environments/krylov").exists())
                log.unlink()
                env["CUNUMERIC_SOURCE"] = shell_path(root / "missing")
                result = subprocess.run([BASH, shell_path(bench / script.name)], cwd=root,
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CUNUMERIC_SOURCE", result.stderr)
                self.assertFalse(log.exists())
