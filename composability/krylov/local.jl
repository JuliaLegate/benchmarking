# These updates are outside the solver loop because @accelerate does not
# accept control flow.
@accelerate function cg_update!(x, r, p, Ap, α)
    x .= x .+ α .* p
    r .= r .- α .* Ap
end

@accelerate function bicg_start!(s, r, v, α)
    s .= r .- α .* v
end

@accelerate function bicg_finish!(x, r, p, s, t, α, ω)
    x .= x .+ α .* p .+ ω .* s
    r .= s .- ω .* t
end

@accelerate function bicg_direction!(p, r, v, β, ω)
    p .= r .+ β .* (p .- ω .* v)
end

function local_workspace(b)
    fields = SOLVER == "cg" ? (:x, :r, :p, :Ap) : (:x, :r, :p, :v, :s, :t)
    return NamedTuple{fields}(map(_ -> similar(b), fields))
end

function local_cg!(w, A, b)
    x, r, p, Ap = w
    fill!(x, zero(T)); copyto!(r, b); copyto!(p, r)
    γ = dot(r, r)
    limit = TOL * sqrt(γ)
    for k in 1:200
        mul!(Ap, A, p)
        α = γ / dot(p, Ap)
        cg_update!(x, r, p, Ap, α)
        γnext = dot(r, r)
        sqrt(γnext) <= limit && return x, k, true
        β = γnext / γ
        p .= r .+ β .* p
        γ = γnext
    end
    return x, 200, false
end

function local_bicgstab!(w, A, b)
    x, r, p, v, s, t = w
    fill!(x, zero(T)); copyto!(r, b); copyto!(p, r)
    rhat = b # Fixed shadow residual; initial guess is zero.
    limit = TOL * norm(r)
    ρ = dot(rhat, r)
    for k in 1:200
        mul!(v, A, p)
        α = ρ / dot(rhat, v)
        bicg_start!(s, r, v, α)
        mul!(t, A, s)
        ω = dot(t, s) / dot(t, t)
        bicg_finish!(x, r, p, s, t, α, ω)
        norm(r) <= limit && return x, k, true
        ρnext = dot(rhat, r)
        β = (ρnext / ρ) * (α / ω)
        bicg_direction!(p, r, v, β, ω)
        ρ = ρnext
    end
    return x, 200, false
end
