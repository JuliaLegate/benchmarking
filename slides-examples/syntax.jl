N = 10
A = cuNumeric.rand(Float32, N, N)
B = cuNumeric.rand(Float32, N, N)
C = cuNumeric.zeros(Float32, N, N)
mul!(C, A, B)

integrand = (x) -> exp(-square(x))
N = 1_000_000
x_max = 5.0
domain = [-x_max, x_max]
Ω = domain[2] - domain[1]
samples = Ω*cuNumeric.rand(NDArray, N) - x_max
estimate = (Ω/N) * sum(integrand(samples))


N = 100
u = cuNumeric.ones((N, N))
v = cuNumeric.ones((N, N))
for i in iters
    F_u = ((u[2:(end - 1), 2:(end - 1)] .* (v[2:(end - 1), 2:(end - 1)] 
        .* v[2:(end - 1), 2:(end - 1)])) - (f+k)*v[2:(end - 1), 2:(end - 1)])
    #.... more physics
    #.... update u and v
    GC.gc()
end


N = 10
A = cuNumeric.rand(Float32, N, N)
B = cuNumeric.rand(Float32, N, N)
C = cuNumeric.zeros(Float32, N, N)
mul!(C, A, B)


A = cuNumeric.rand(FT, dim)
B = cuNumeric.rand(FT, dim)
C = sin(A) + B