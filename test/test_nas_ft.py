"""NAS FT adapter checks with NumPy as an independent CPU execution backend."""
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_timing import SOURCE, load

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is required for CPU FT verification")
class NASFTTests(unittest.TestCase):
    def test_reference_checksums_match_julia_for_every_class(self):
        with patch.dict("sys.modules", {
            "cupynumeric": np,
            "core": SimpleNamespace(register_benchmark=lambda *args: None),
        }):
            ft = load(SOURCE / "benchmarks" / "nas" / "ft.py")
        # Verify the shared tables without running large classes.
        source = (SOURCE.parent / "nas" / "ft.jl").read_text()
        for cls, checksums in ft.CHECKSUMS.items():
            with self.subTest(cls=cls):
                block = re.search(
                    rf'"{cls}" => ComplexF64\[(.*?)\n    \]', source, re.S
                ).group(1)
                reference = [complex(float(re_part), float(im_part))
                             for re_part, im_part in re.findall(
                                 r"([\d.e+-]+) \+ ([\d.e+-]+)im", block)]
                self.assertEqual(len(checksums), ft.CLASSES[cls][3])
                self.assertEqual(checksums, reference)
        self.assertEqual(ft.CHECKSUMS["B"][15],
                         512.6064276004 + 511.4218460548j)

    def test_sparse_checksum_and_official_verification(self):
        with patch.dict("sys.modules", {
            "cupynumeric": np,
            "core": SimpleNamespace(register_benchmark=lambda *args: None),
        }):
            ft = load(SOURCE / "benchmarks" / "nas" / "ft.py")
        for cls in ("S", "W"):
            with self.subTest(cls=cls):
                nx, ny, nz, _ = ft.CLASSES[cls]
                b = ft.NASFourierTransform(np.float64, nx, ny, **{"class": cls})
                s = b.initialize()
                self.assertEqual(s["indices"].shape, (1024,))
                self.assertNotIn("mask", s)
                # Independent coordinate indexing checks flattening order and
                # multiplicities, especially the non-cubic W grid.
                values = np.arange(nx*ny*nz).reshape(nz, ny, nx)
                expected = [values[(5*j) % nz, (3*j) % ny, j % nx]
                            for j in range(1, 1025)]
                np.testing.assert_array_equal(np.take(values, s["indices"]), expected)
                got = b.run(s)
                self.assertTrue(all(abs((x-r)/r) <= 1e-12
                                    for x, r in zip(got, ft.CHECKSUMS[cls])))
