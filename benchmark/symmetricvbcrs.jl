using BlockSparseMatrices
using OhMyThreads
using LinearAlgebra
using Random

Random.seed!(1)
blocksize = 16
nblocks = 80
indices = [collect(((i - 1) * blocksize + 1):(i * blocksize)) for i in 1:nblocks]
blocks = [randn(ComplexF64, blocksize, blocksize) for _ in 1:(nblocks - 1)]
sbm = SymmetricBlockMatrix(
    Matrix{ComplexF64}[],
    Vector{Int}[],
    blocks,
    indices[1:(end - 1)],
    indices[2:end],
    (blocksize * nblocks, blocksize * nblocks),
)

old = VariableBlockCompressedRowStorage(sbm)
new = SymmetricVariableBlockCompressedRowStorage(sbm)
x = randn(ComplexF64, size(sbm, 2))
yold = similar(x)
ynew = similar(x)

mul!(yold, old, x)
mul!(ynew, new, x)
@assert isapprox(yold, ynew)

function average_time(f; repeats=100)
    elapsed = @elapsed for _ in 1:repeats
        f()
    end
    return elapsed / repeats
end

println("Old storage: ", Base.format_bytes(Base.summarysize(old)))
println("New storage: ", Base.format_bytes(Base.summarysize(new)))
println("Old MVP (s): ", average_time(() -> mul!(yold, old, x)))
println("New MVP (s): ", average_time(() -> mul!(ynew, new, x)))

# Reference: compare both storage formats with the original SymmetricBlockMatrix.
ysbm = similar(x)
mul!(ysbm, sbm, x)
@assert isapprox(ysbm, ynew)
println("SymmetricBlockMatrix storage: ", Base.format_bytes(Base.summarysize(sbm)))
println("SymmetricBlockMatrix MVP (s): ", average_time(() -> mul!(ysbm, sbm, x)))
