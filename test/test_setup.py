"""Exercise standalone/submodule setup without downloading Julia packages."""
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
