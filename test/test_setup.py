"""Exercise standalone/submodule setup without downloading Julia packages."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


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
                shutil.copy(script, bench / script.name)
                log = root / "args"
                fake = root / "julia"
                fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$SETUP_TEST_LOG"\n')
                fake.chmod(0o755)
                env = {k: v for k, v in os.environ.items() if k != "CUNUMERIC_SOURCE"}
                env.update(CUNUMERIC_BENCH_JULIA=str(fake), SETUP_TEST_LOG=str(log))
                if standalone:
                    # Relative overrides are relative to the caller, not the script.
                    env["CUNUMERIC_SOURCE"] = source.name
                result = subprocess.run(["bash", str(bench / script.name)], cwd=root,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(log.read_text().splitlines()[-2:],
                                 [str(source), str(source / "lib/CNPreferences")])
                log.unlink()
                env["CUNUMERIC_SOURCE"] = str(root / "missing")
                result = subprocess.run(["bash", str(bench / script.name)], cwd=root,
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CUNUMERIC_SOURCE", result.stderr)
                self.assertFalse(log.exists())
