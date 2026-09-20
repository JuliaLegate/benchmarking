using CUDA, Profile

arr = CUDA.ones(Float32, 10_000_000)
Profile.take_heap_snapshot()
Base.summarysize(arr) # 184