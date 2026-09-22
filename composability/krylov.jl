using LinearAlgebra, Statistics, Krylov, cuNumeric, CUDA

length(ARGS) == 4 || error("Usage: julia krylov.jl {cuNumeric|CuArray} {cg|bicgstab} {stock|plain|fused} N")
const BACKEND, SOLVER, MODE = ARGS[1:3]
const N = parse(Int, ARGS[4])
BACKEND in ("cuNumeric", "CuArray") || error("Unknown backend: $BACKEND")
SOLVER in ("cg", "bicgstab") || error("Unknown solver: $SOLVER")
MODE in ("stock", "plain", "fused") || error("Unknown mode: $MODE")
N > 1 || error("N must exceed 1")
const T = get(ENV, "BENCH_ELTYPE", "Float32") == "Float64" ? Float64 : Float32
const TOL = T === Float32 ? T(1e-5) : T(1e-8)
BLAS.set_num_threads(1)
CUDA.allowscalar(false)
cuNumeric.allowscalar(false)

make_array(a) = BACKEND == "cuNumeric" ? NDArray(a) : CuArray(a)
sync() = BACKEND == "cuNumeric" ? cuNumeric.issue_execution_fence(; block=true) : CUDA.synchronize()

# These small bodies are outside the solver loop because @accelerate does not
# accept control flow. The plain bodies mirror Krylov's vector recurrences.
cg_plain!(x, r, p, Ap, α) = (axpy!(α, p, x); axpy!(-α, Ap, r))
@accelerate function cg_fused!(x, r, p, Ap, α)
    x .= x .+ α .* p
    r .= r .- α .* Ap
end

bicg_start_plain!(s, r, v, α) = (copyto!(s, r); axpy!(-α, v, s))
@accelerate function bicg_start_fused!(s, r, v, α)
    s .= r .- α .* v
end

bicg_finish_plain!(x, r, p, s, t, α, ω) =
    (axpy!(α, p, x); axpy!(ω, s, x); copyto!(r, s); axpy!(-ω, t, r))
@accelerate function bicg_finish_fused!(x, r, p, s, t, α, ω)
    x .= x .+ α .* p .+ ω .* s
    r .= s .- ω .* t
end

bicg_direction_plain!(p, r, v, β, ω) =
    (axpy!(-ω, v, p); axpby!(one(eltype(p)), r, β, p))
@accelerate function bicg_direction_fused!(p, r, v, β, ω)
    p .= r .+ β .* (p .- ω .* v)
end

function local_workspace(b)
    fields = SOLVER == "cg" ? (:x, :r, :p, :Ap) : (:x, :r, :p, :v, :s, :t)
    return NamedTuple{fields}(map(_ -> similar(b), fields))
end

function local_cg!(w, A, b, fused)
    x, r, p, Ap = w
    fill!(x, zero(T)); copyto!(r, b); copyto!(p, r)
    γ = dot(r, r)
    limit = TOL * sqrt(γ)
    for k in 1:200
        mul!(Ap, A, p)
        α = γ / dot(p, Ap)
        fused ? cg_fused!(x, r, p, Ap, α) : cg_plain!(x, r, p, Ap, α)
        γnext = dot(r, r)
        sqrt(γnext) <= limit && return x, k, true
        β = γnext / γ
        if fused
            p .= r .+ β .* p
        else
            axpby!(one(T), r, β, p)
        end
        γ = γnext
    end
    return x, 200, false
end

function local_bicgstab!(w, A, b, fused)
    x, r, p, v, s, t = w
    fill!(x, zero(T)); copyto!(r, b); copyto!(p, r)
    rhat = b # Fixed shadow residual; initial guess is zero.
    limit = TOL * norm(r)
    ρ = dot(rhat, r)
    for k in 1:200
        mul!(v, A, p)
        α = ρ / dot(rhat, v)
        fused ? bicg_start_fused!(s, r, v, α) : bicg_start_plain!(s, r, v, α)
        mul!(t, A, s)
        ω = dot(t, s) / dot(t, t)
        fused ? bicg_finish_fused!(x, r, p, s, t, α, ω) :
                bicg_finish_plain!(x, r, p, s, t, α, ω)
        norm(r) <= limit && return x, k, true
        ρnext = dot(rhat, r)
        β = (ρnext / ρ) * (α / ω)
        fused ? bicg_direction_fused!(p, r, v, β, ω) :
                bicg_direction_plain!(p, r, v, β, ω)
        ρ = ρnext
    end
    return x, 200, false
end

lower = fill(T(SOLVER == "cg" ? -0.5 : -0.3), N - 1)
upper = fill(T(SOLVER == "cg" ? -0.5 : -0.8), N - 1)
diagonal = T.(range(2.0, 4.0; length=N))
reference = Tridiagonal(lower, diagonal, upper)
Ah = Matrix(reference) # The timed operator is dense for both backends.
bh = T[sin(i) + 1 for i in 1:N]
const A, b = make_array(Ah), make_array(bh)
const w = MODE == "stock" ?
    (SOLVER == "cg" ? Krylov.CgWorkspace(A, b) : Krylov.BicgstabWorkspace(A, b)) :
    local_workspace(b)

function solve!()
    if MODE == "stock"
        if SOLVER == "cg"
            Krylov.cg!(w, A, b; atol=zero(T), rtol=TOL, itmax=200)
        else
            Krylov.bicgstab!(w, A, b; atol=zero(T), rtol=TOL, itmax=200)
        end
        return w.x, w.stats.niter, w.stats.solved
    end
    return SOLVER == "cg" ? local_cg!(w, A, b, MODE == "fused") :
                            local_bicgstab!(w, A, b, MODE == "fused")
end

permitted_solve!() = @allowpromotion @allowautofetch solve!()

function checked_solve!()
    x, iterations, solved = permitted_solve!()
    sync()
    solved || error("$SOLVER did not converge")
    residual = norm(reference * Float64.(Array(x)) - Float64.(bh)) / norm(Float64.(bh))
    residual <= TOL || error("Relative residual $residual exceeds $TOL")
    return iterations, residual
end

for _ in 1:2
    checked_solve!()
end
samples = Float64[]
for _ in 1:5
    GC.gc(); sync()
    start = time_ns()
    permitted_solve!()
    sync()
    push!(samples, (time_ns() - start) / 1e6)
end
iterations, residual = checked_solve!()
println("RESULT,$BACKEND,$SOLVER,$MODE,$T,$N,$iterations,$(median(samples)),$(minimum(samples)),$(maximum(samples)),$residual,$(join(samples, ';'))")
