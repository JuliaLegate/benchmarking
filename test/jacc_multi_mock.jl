# CPU mock of JACC.Multi (JACC 1.4, ext/CUDAExt/multi.jl): same column
# partitioning, ghost layout, and exchange order, on `ndev` simulated devices.
struct MockPart{T}
    a::Matrix{T}
    dev_id::Int
    ndev::Int
    ghost_dims::Int
end
Base.getindex(p::MockPart, i) = p.a[i]
Base.setindex!(p::MockPart, v, i) = (p.a[i] = v)
Base.length(p::MockPart) = length(p.a)
Base.size(p::MockPart) = size(p.a)

struct MockMulti{T}
    a1::Vector{MockPart{T}}
    a2::Vector{Matrix{T}}
    orig_size::Tuple{Int,Int}
    ng::Int
end

struct MockOps
    ndev::Int
end

jm_ndev(o::MockOps) = o.ndev
jm_parts(::MockOps, a::MockMulti) = a.a2

function jm_array(o::MockOps, x::Matrix{T}; ghost_dims) where {T}
    nd, total = o.ndev, size(x, 2)
    ng = nd == 1 ? 0 : ghost_dims
    partlen = cld(total, nd)
    mats = Matrix{T}[]
    for i in 1:nd
        lo = (i - 1)*partlen + 1 - (i > 1 ? ng : 0)
        hi = i == nd ? total : i*partlen + ng
        push!(mats, x[:, lo:hi])
    end
    parts = [MockPart(mats[i], i, nd, ng) for i in 1:nd]
    return MockMulti(parts, mats, size(x), ng)
end

jm_array(o::MockOps, x::Vector; ghost_dims) = jm_array(o, reshape(x, 1, :); ghost_dims)

mock_arg(x, d) = x
mock_arg(x::MockMulti, d) = x.a1[d]

function jm_for(o::MockOps, N::Integer, f, args...)
    part = cld(N, o.ndev)
    for d in 1:o.ndev
        n = d == o.ndev ? N - (o.ndev - 1)*part : part
        a = map(x -> mock_arg(x, d), args)
        for i in 1:n
            f(i, a...)
        end
    end
end

function jm_reduce(o::MockOps, N::Integer, f, args...)
    part, total = cld(N, o.ndev), 0.0
    for d in 1:o.ndev
        n = d == o.ndev ? N - (o.ndev - 1)*part : part
        a = map(x -> mock_arg(x, d), args)
        for i in 1:n
            total += f(i, a...)
        end
    end
    return total
end

function jm_sync!(o::MockOps, x::MockMulti)
    ng = x.ng
    ng == 0 && return nothing
    for i in 1:(o.ndev - 1)            # left to right
        src, cols = x.a2[i], size(x.a2[i], 2)
        x.a2[i + 1][:, 1:ng] .= src[:, (cols + 1 - 2ng):(cols - ng)]
    end
    for i in 2:o.ndev                  # right to left
        dst, cols = x.a2[i - 1], size(x.a2[i - 1], 2)
        dst[:, (cols - ng + 1):cols] .= x.a2[i][:, (1 + ng):(2ng)]
    end
    return nothing
end

function jm_to_host(o::MockOps, x::MockMulti{T}) where {T}
    out = Matrix{T}(undef, x.orig_size)
    part = cld(x.orig_size[2], o.ndev)
    for d in 1:o.ndev
        lo = (d - 1)*part + 1
        hi = d == o.ndev ? x.orig_size[2] : d*part
        skip = d > 1 ? x.ng : 0
        out[:, lo:hi] .= x.a2[d][:, (skip + 1):(skip + hi - lo + 1)]
    end
    return out
end

jm_copy!(::MockOps, dst, dd, doff, src, ds, soff, n) = copyto!(dst, doff, src, soff, n)
jm_upload!(::MockOps, part, d, host, n) = copyto!(part, 1, host, 1, n)

function jm_each_part(f, ::MockOps, a::MockMulti)
    for (d, part) in enumerate(a.a2)
        f(part, d)
    end
    return a
end
