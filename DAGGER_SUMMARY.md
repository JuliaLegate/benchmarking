# Dagger.jl vs cuNumeric.jl: multi-GPU findings

Machine: 4x NVIDIA L40S (48 GB), PCIe behind a host bridge (`nvidia-smi topo`: PHB),
**no peer access between any pair of GPUs**. One Julia process drives all GPUs.
Dagger: `/opt/Dagger-aot` (branch `aot-schedulers-rebased`, uncommitted changes listed below).

## Results at a glance (mean of 5 trials; every configuration passes correctness)

| Benchmark | GPUs | cuNumeric | Dagger before | Dagger now |
|---|---:|---:|---:|---:|
| GEMM (`gemm.toml`) | 1 / 2 / 4 | 2404 / 2897 / 4813 ms | 2468 / 6610 / **crash** ms | **2444 / 2474 / 3387 ms** |
| Gray-Scott (`grayscott-dagger.toml`) | 1 / 2 / 4 | 24.9 / 26.2 / 25.9 ms | 30.4 / 34.1 / 36.4 ms¹ | **17.8 / 20.9 / 24.1 ms** |
| Monte Carlo (`montecarlo-dagger.toml`) | 1 / 2 / 4 | 25.8 / 26.1 / 26.6 ms | 27.1 / **OOM** / **OOM** | **27.1 / 26.9 / 27.5 ms** |
| CG (`cg-dagger.toml`) | 1 / 2 / 4 | 383 / 393 / 396 ms² | 172 / 185 / 5802 ms | **172 / 191 / 201 ms** |

¹ Same auto-sized N as cuNumeric, measured with every Gray-Scott fix except the
buffer swap (`results/20261001-201104-j7jKZm`). With none of them, at the
original config sizes: 46.6 / 52.1 / 75.9 ms, and Dagger's 4-GPU run ran out of
memory on a later trial (see Gray-Scott).
² Not like-for-like; see the CG caveat.

Weak scaling: lower time at more GPUs is better; flat is ideal. Result
directories: GEMM `results/20261001-183051-di7msZ`, Gray-Scott
`results/20261001-202100-rOmntZ`, Monte Carlo `results/20261001-192255-uIz8RI`,
CG `results/20261001-192731-N8Iks1`.

## GEMM (`gemm.toml`, Float32, weak scaling, `C = A * B`)

| GPUs | N | cuNumeric | Dagger before | Dagger after |
|---:|---:|---:|---:|---:|
| 1 | 35744 | 2404 ms | 2468 ms | 2444 ms |
| 2 | 45032 | 2897 ms | 6610 ms | **2474 ms** |
| 4 | 56736 | 4813 ms | crash (segfault) | **3387 ms** |

Mean of 5 trials, all correctness checks `pass`. "Before" is
`results/20261001-171932-9Zxs5w`; "after" is `results/20261001-183051-di7msZ`.
Ideal weak scaling (1-GPU rate on every GPU) is ~2.4 s at every scale: Dagger is
now at ideal on 2 GPUs and 1.4x slower than ideal on 4. cuNumeric is 1.2x and 2.0x
slower than ideal.

### What was wrong, and the fixes

1. **The 4-GPU crash: host memory unregistered from a GC finalizer that could not
   run.** Datadeps' cross-device remainder copies staged through host buffers
   page-locked with `CUDA.pin`, whose finalizer takes a `ReentrantLock`. When the
   lock was contended the finalizer threw `task switch not allowed from inside gc
   finalizer`, so the unregistration was lost. GC freed the memory anyway, leaving
   freed ranges registered with the driver. The worker then segfaulted, with a
   stack overflow while printing the backtrace hiding the cause.
   *Fix:* `pin_buffer!` (CUDAExt) registers buffers itself. Its finalizer only
   `trylock`s, and on contention re-arms itself, which keeps the buffer alive until
   a later GC. In addition, same-process GPU-to-GPU copies no longer go through the
   host at all (next item).

2. **Cross-GPU copies ran at 1.4 GB/s.** Without peer access, CUDA.jl's `copyto!`
   between devices stages through a fresh *pageable* `Vector` with a synchronous
   device-to-host copy. `cuMemcpyPeerAsync` works without peer access (the driver
   pipelines it through its own pinned buffers) and measured **21.7 GB/s** on this
   machine.
   *Fix:* every same-process device-to-device path (in-place `move!`, the
   out-of-place `move`, and Datadeps remainder copies through a new core hook
   `device_remainder_copy!`) now uses peer copies. Strided spans are packed on the
   source and unpacked on the destination, so a copy isn't thousands of tiny
   transfers. This alone took 4 GPUs from 27 s per call to ~4.3 s.

3. **Round-robin placement only matched owner-computes by coincidence.** The
   default Datadeps scheduler (RoundRobin) placed each `C[m,n]` update on its
   owner only because the row-major task order happened to line up with the
   column owners. With any other order, every task copied A, B *and* C, then wrote
   C back: 48 copies per call instead of 12.
   *Fix (benchmark):* the Dagger GEMM worker now uses `Dagger.GreedyScheduler()`,
   whose cost model keeps each update on its tile's owner and moves only A's
   panels. It is selectable with the `scheduler` kwarg (`"greedy"`, the default,
   or `"roundrobin"`).

4. **Every GPU wanted the same A panel at once.** In row-major order, all column
   owners need row panel `A[m,:]` simultaneously. The copies queue behind the
   panel owner's kernels while the other GPUs idle; on 2 GPUs one GPU idled for a
   whole tile (3.86 s, versus 2.7 s once fixed).
   *Fix (Dagger `gemm_dagger!`):* visit C's tiles by wrapped diagonals. Step 0 is
   all-local, and each later step needs a different panel per GPU (a ring), so
   transfers spread out and overlap. Each `C[m,n]`'s own `k` sequence is
   unchanged.

5. **Copies could not overlap compute.** Each GPU had one stream, so incoming
   copies serialized with that GPU's kernels. The copy also waited for every
   kernel queued on the *source* GPU, including the ones only *reading* the panel
   being sent.
   *Fix:* peer copies run on a per-device copy stream. They wait only on per-
   allocation events (`BUFFER_EVENTS`): the source's last writer, and the
   destination's last writer and reader. `execute!` records those events after
   each GPU task, using a new `writes` task option that Datadeps fills with the
   positions of the arguments a task writes. Copies are host-synchronized on
   completion, so every pre-existing synchronization point stays valid. Anything
   untracked falls back to a whole-stream wait. Effect at 4 GPUs: ~3.9 s → ~3.1–3.4 s.

6. **Tuning.** With Greedy, `blocks_per_gpu = 1` is best at every scale
   (4 GPUs: bpg 1 ≈ 3.1–3.6 s; bpg 2/3/4 ≈ 4.7–5.4 s). Every GPU needs all of A
   regardless, so smaller tiles add tasks and copies and make each GEMM less
   efficient. Larger problem sizes were not needed: at these sizes each call takes
   seconds, so scheduling overhead is negligible.

Remaining gap at 4 GPUs (~3.4 s vs ~2.4 s ideal): each GPU still pulls 3 × 3.2 GB
of A per call through the host bridge, and the four transfers share it. Datadeps
frees the replicas at region end, so every call re-copies A. Legate keeps valid
replicas across operations. Caching read-only replicas across Datadeps regions is
the next lever.

### Pre-existing bugs found by the new tests (fixed)

- `aliasing(::SubArray)` called `pointer(parent)`, which in CUDA.jl takes stream
  ownership for the *active* device and throws for a view on another non-peer
  GPU. New core hook `data_address` (CUDA overrides it with the raw address).
- `Dagger.aliasing(::CuArray)` added `x.offset` (an element count) to the pointer
  as bytes.
- Host-to-device `move` of a chunk labelled with a host processor but holding a
  `CuArray` (Datadeps rebuilding a view of a host array on a GPU) tried to
  page-lock device memory.
- Host↔device `move!` of strided device views fell back to scalar indexing.

### Dagger changes (`/opt/Dagger-aot`, uncommitted)

| File | Change |
|---|---|
| `ext/CUDAExt.jl` | finalizer-safe `pin_buffer!`; peer copies (`peer_copy_spans!`, packing, `peer_move!`, `_peer_copy_out`, `device_remainder_copy!`); per-device copy streams and `BUFFER_EVENTS` tracking (`_record_task!` in `execute!`, `_note_alloc!` in `alloc_uninit`); `data_address`; `aliasing(::CuArray)` offset fix; device-chunk HtoD fix; strided-view HtoD/DtoH |
| `src/datadeps/remainders.jl` | `device_remainder_copy!` hook, used for same-process device-to-device remainders |
| `src/datadeps/queue.jl` | `written_positions`; sets the `writes` option on every Datadeps task |
| `src/options.jl` | new `writes` option |
| `src/array/mul.jl` | diagonal traversal in `gemm_dagger!` |
| `src/memory-spaces.jl` | `data_address` hook; SubArray aliasing computes the view address from the parent's |
| `src/datadeps/scheduling.jl` | schedule cache stores only the matching key (`_schedule_cache_key`), not tasks/arguments; `DATADEPS_SCHEDULE_CACHE_FULL_SPECS` opt-in for tests that take fixtures from the cache |
| `test/datadeps/scheduling.jl` | regression test that cached plans hold no tasks; enables full specs for the fixture-harvesting tests |
| `test/gpu.jl` | "Cross-GPU copies, one process" testset (RoundRobin and Greedy): alternating writers, writer vs. readers of the previous value, strided views, repeated tiled GEMM |
| `AGENTS.md` | lessons 74–77 |

Tests: `test/gpu.jl` on 2 GPUs with 3 workers: CUDA 306/306, distributed shared-GPU
48/48, multi-GPU-in-one-process 2/2. `test/datadeps.jl` with 1 worker: 1855 pass;
2 errors from my harness lacking `using Random` (the official runner provides
it). `test/datadeps/scheduling.jl` with 1 worker, after the schedule-cache
change: 542 pass, 1 broken (pre-existing).

## Monte Carlo (`montecarlo-dagger.toml`, Float32, auto-sized)

**Error (2 and 4 GPUs): out of GPU memory during initialization.** The samples were
built as `T(10) .* rand(...)`, which materializes a second full-size DArray. Its
broadcast tasks were scoped to *all* GPUs, so some ran off their chunk's owner and
pulled a third copy there. At the auto-sized N (~16.9 GiB per GPU) that is out of
memory. *Fix (benchmark):* generate the samples, then scale them in place with one
task pinned to each chunk's GPU. Dagger now scales perfectly (26.9–27.5 ms) and is
within ~4% of cuNumeric. The small remaining gap is per-call task-launch and
reduction overhead.

## CG (`cg-dagger.toml`, Float64, 10 iterations to convergence)

**Performance bug (4 GPUs, 30x slowdown):** each in-place DArray broadcast
(`x .+= alpha .* p` etc.) is a Datadeps region, placed by the default RoundRobin
scheduler. On 2 GPUs the round-robin order happened to match each chunk's owner
(3.8 ms per broadcast). On 4 it did not, so every chunk was copied to another
GPU, updated and copied back (170 ms per broadcast). *Fix (benchmark):* all
Datadeps-based Dagger benchmarks (GEMM, CG, Gray-Scott) now select
`GreedyScheduler` through a shared helper (`dagger_datadeps_scheduler` in
`src/dagger/common.jl`, `scheduler` kwarg: `"greedy"` or `"roundrobin"`).
Result: 5802 → 201 ms at 4 GPUs.

**Caveat:** the two CG implementations are not like-for-like. cuNumeric stores
the tridiagonal matrix as three vectors and applies it in three shifted passes.
The Dagger version is a matrix-free, constant-coefficient `@stencil` that reads
only `p`. Both run the same 10 iterations to the same tolerance, but Dagger moves
much less memory per iteration, so its ~2x lead mostly reflects the formulation.
I left both as they were; if you want a fair comparison, one of them should be
changed to match the other.

## Gray-Scott (`grayscott-dagger.toml`, Float32, 50 steps per trial)

1. **Sizes:** the `configs/multi_gpu/grayscott.toml` sizes (N=28000 on 1 GPU) do
   not fit cuNumeric on these 48 GB GPUs: Legate keeps about 15 full-size arrays
   per GPU, and the auto-sizer underestimates this, failing even at
   `mem_frac = 0.75`. `grayscott-dagger.toml` auto-sizes at `mem_frac = 0.5`.
   This is a cuNumeric/auto-sizer issue I worked around, not fixed.
2. **Dagger out of memory at 4 GPUs across trials, and trials after the first
   ~40% slower.** Two causes:
   - The AOT schedulers' process-wide plan cache stored each region's full
     `DAGSpec`, including the tasks' specs and therefore their array arguments.
     Every cached plan pinned one trial's arrays for the life of the process.
     *Fix (Dagger):* cache only the part plans are matched on
     (`_schedule_cache_key`).
   - A dropped DArray is not freed by the collection itself: finalizers queue
     releases on MemPool's work queue, which drains later. The worker collected
     and immediately allocated the next trial, so at large N the previous
     trial's arrays were still resident and the near-full pool slowed every
     later trial (71 → 101 ms/step). *Fix (benchmark):* a `model_release_memory`
     hook in `model_trial`. Dagger's version collects until MemPool's queue is
     idle.
3. **Layout:** square tiles of side N/gpus gave each GPU a whole column of
   tiles at 4 GPUs: 4x the tasks of one strip, plus strided halos between a
   GPU's own tiles every step. *Fix (benchmark):* column strips (`Blocks(N, N/gpus)`),
   whose halos are contiguous columns. 4 GPUs: 66 → 50 ms/step at the larger N.
4. **Unequal work:** the Dagger step copied `Un → U` and `Vn → V` (two extra full
   passes), while the array-backend workers swap buffers. *Fix (benchmark):*
   swap. 1 GPU: 30.4 → 17.8 ms.
5. **Tuning:** `blocks_per_gpu = 1` is best (strips: bpg 2 ≈ 53 ms vs 50 ms at
   4 GPUs; with square tiles bpg 2/4 were 108/245 ms). RoundRobin and Greedy
   perform the same once the layout is fixed.

Remaining: 17.8 → 24.1 ms from 1 to 4 GPUs. Per-step halo exchange (host-synced
peer copies plus one Datadeps region per step) is the main cost; overlapping
interior compute with halo transfers would close most of it.

### Benchmark repo changes

- `src/dagger/common.jl`: `dagger_datadeps_scheduler` / `with_dagger_scheduler`
  (Greedy by default); `dagger_release_memory`.
- `src/model_worker.jl`: `model_release_memory` hook, called at the start of
  each trial (defaults to `GC.gc(true)`, the previous behavior).
- `src/dagger/single.jl`: Dagger's `model_release_memory`.
- `src/dagger/benchmarks/gemm.jl`, `cg.jl`, `grayscott.jl`: use the selected
  scheduler. In `grayscott.jl`, also the strip layout and buffer swap.
- `src/dagger/benchmarks/montecarlo.jl`: in-place, owner-pinned sample scaling.
- New configs: `grayscott-dagger.toml`, `montecarlo-dagger.toml`,
  `cg-dagger.toml` (cuNumeric vs Dagger, 1/2/4 GPUs).
- `environments/dagger/Project.toml` still points at `/opt/Dagger-aot` (your
  local change; I left it as-is).

## Recommendations for Dagger

- **RoundRobin is a hazardous default for GPU Datadeps.** It ignored data
  placement in two of these four benchmarks, and the damage only appeared at
  4 GPUs. Make the default owner-aware (Greedy, or RoundRobin with an
  owner-computes preference for written arguments). At minimum, elementwise
  in-place broadcasts should pin each task to its destination chunk's owner.
- **Cache read-only replicas across Datadeps regions.** GEMM re-copies every A
  panel on every call; Legate does not.
- **ROCm/oneAPI/Metal/OpenCL extensions** share the patterns fixed here for CUDA
  (finalizer-based pinning, host-staged cross-device copies) but were not
  changed or tested.
