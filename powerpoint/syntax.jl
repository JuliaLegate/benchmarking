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
A = cuNumeric.rand(Float64, N, N)
B = cuNumeric.rand(Float64, N, N)
C = cuNumeric.zeros(Float64, N, N)
mul!(C, A, B)


A = cuNumeric.rand(FT, dim)
B = cuNumeric.rand(FT, dim)
C = sin(A) + B