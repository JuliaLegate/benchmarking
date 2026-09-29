"""Exercise setup and launcher failures without downloading Julia packages or using GPUs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def shell_path(path):
    # Match Git Bash mounts (including its /tmp alias) on Windows.
    path = path.resolve()
    if os.name == "nt":
        return subprocess.check_output(
            ["bash", "-c", '/usr/bin/cygpath -u "$1"', "bash", path.as_posix()],
            text=True).strip()
    return str(path)


class SetupTests(unittest.TestCase):
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
                        ["bash", "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"',
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
                result = subprocess.run(["bash", shell_path(bench / script.name)], cwd=root,
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
                result = subprocess.run(["bash", shell_path(bench / script.name)], cwd=root,
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CUNUMERIC_SOURCE", result.stderr)
                self.assertFalse(log.exists())
