# CUDA reference provenance and license

The four CUDA C++ comparison excerpts below have their comments removed for
source measurement. Keep this notice with the excerpts when copying or sharing
them. Their inclusion rules and local adaptations are documented in the
[LOC analysis README](../README.md).

Upstream: GMAP/NPB-GPU, commit
`3f12d84920ee315ab00ef283717c1e74b68f4d00`:

- `nas_ep/cuda.cu.ref`: [CUDA/EP/ep.cu](https://github.com/GMAP/NPB-GPU/blob/3f12d84920ee315ab00ef283717c1e74b68f4d00/CUDA/EP/ep.cu).
- `nas_ft/cuda.cu.ref`: [CUDA/FT/ft.cu](https://github.com/GMAP/NPB-GPU/blob/3f12d84920ee315ab00ef283717c1e74b68f4d00/CUDA/FT/ft.cu).
- `nas_ft/cuda_cufft.cu.ref`: a local cuFFT adaptation of the same FT source,
  not an implementation provided by upstream.
- `nas_mg/cuda.cu.ref`: [CUDA/MG/mg.cu](https://github.com/GMAP/NPB-GPU/blob/3f12d84920ee315ab00ef283717c1e74b68f4d00/CUDA/MG/mg.cu).

FT's complex type and arithmetic helpers were adapted from upstream
[`CUDA/common/npb-CPP.hpp`](https://github.com/GMAP/NPB-GPU/blob/3f12d84920ee315ab00ef283717c1e74b68f4d00/CUDA/common/npb-CPP.hpp).
Launch defaults come from the same revision's `CUDA/config/gpu.config`.

## Attribution

Parallel Applications Modelling Group — GMAP, [gmap.pucrs.br](https://gmap.pucrs.br),
Pontifical Catholic University of Rio Grande do Sul (PUCRS),
Av. Ipiranga, 6681, Porto Alegre — Brazil, 90619-900.

The original NPB 3.4 Fortran code belongs to
[NASA NAS Parallel Benchmarks](https://www.nas.nasa.gov/Software/NPB/).
Original authors credited by the upstream sources:

- EP: P. O. Frederickson, D. H. Bailey, A. C. Woo.
- FT: D. Bailey, W. Saphir.
- MG: E. Barszcz, P. Frederickson, A. Woo, M. Yarrow.

The [serial C++ translation](https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER)
credits Dalvan Griebler (`dalvangriebler@gmail.com`),
Gabriell Araujo (`hexenoften@gmail.com`), and Júnior Löff (`loffjh@gmail.com`).
The CUDA implementations credit Gabriell Araujo (`hexenoften@gmail.com`).

## MIT License

Copyright (c) 2021 Parallel Applications Modelling Group - GMAP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
