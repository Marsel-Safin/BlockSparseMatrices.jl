"""
    SymmetricVariableBlockCompressedRowStorage(sbm::SymmetricBlockMatrix;
        scheduler=sbm.scheduler, coloringalgorithm=coloringalgorithm)

A compressed storage format for complex-symmetric block matrices. Diagonal blocks and
one copy of each off-diagonal block are stored; the transposed contribution is applied
without storing another block. Block indices may be non-contiguous.

Blocks have a row-major and a column-major ordering. Separate conflict-free colors
protect the output rows during parallel multiplication.
"""
struct SymmetricVariableBlockCompressedRowStorage{T,D,M,P,S} <: AbstractBlockMatrix{T}
    diagonals::Vector{D}
    offdiagonals::Vector{M}
    diagonalindices::Vector{Vector{P}}
    rowindices::Vector{Vector{P}}
    colindices::Vector{Vector{P}}
    roworder::Vector{Int}
    rowptr::Vector{Int}
    colorder::Vector{Int}
    colptr::Vector{Int}
    diagonalcolors::Vector{Vector{Int}}
    offdiagonalcolors::Vector{Vector{Int}}
    transposeoffdiagonalcolors::Vector{Vector{Int}}
    size::Tuple{Int,Int}
    scheduler::S
end

function _symmetricvbcrs_checkindices(indices, n, extent)
    length(indices) == n || throw(ArgumentError("Block dimensions and indices must match"))
    isempty(indices) && throw(ArgumentError("Empty block indices are not supported"))
    all(index -> 1 <= index <= extent, indices) ||
        throw(ArgumentError("Block indices are out of bounds"))
    return nothing
end

function _symmetricvbcrs_ptr(order, indices)
    ptr = Int[1]
    for i in 2:length(order)
        if first(indices[order[i]]) != first(indices[order[i - 1]])
            push!(ptr, i)
        end
    end
    !isempty(order) && push!(ptr, length(order) + 1)
    return ptr
end

function SymmetricVariableBlockCompressedRowStorage(
    sbm::SymmetricBlockMatrix{T,D,P,M};
    scheduler=sbm.scheduler,
    coloringalgorithm=coloringalgorithm,
) where {T,D,P,M}
    size(sbm, 1) == size(sbm, 2) ||
        throw(ArgumentError("A symmetric matrix must be square"))
    diagonalindices = sbm.diagonalindices
    rowindices = sbm.rowindices
    colindices = sbm.colindices

    for i in eachindex(sbm.diagonals)
        block = sbm.diagonals[i]
        size(block, 1) == size(block, 2) ||
            throw(ArgumentError("Diagonal blocks must be square"))
        _symmetricvbcrs_checkindices(diagonalindices[i], size(block, 1), size(sbm, 1))
    end
    for i in eachindex(sbm.offdiagonals)
        block = sbm.offdiagonals[i]
        _symmetricvbcrs_checkindices(rowindices[i], size(block, 1), size(sbm, 1))
        _symmetricvbcrs_checkindices(colindices[i], size(block, 2), size(sbm, 2))
    end

    roworder = sortperm(
        eachindex(sbm.offdiagonals); by=i -> (first(rowindices[i]), first(colindices[i]))
    )
    colorder = sortperm(
        eachindex(sbm.offdiagonals); by=i -> (first(colindices[i]), first(rowindices[i]))
    )
    rowptr = _symmetricvbcrs_ptr(roworder, rowindices)
    colptr = _symmetricvbcrs_ptr(colorder, colindices)

    diagonalcolors = _symmetricblockcolors(
        sbm.diagonals, sbm.diagonalindices, scheduler, coloringalgorithm
    )
    offdiagonalcolors = _symmetricblockcolors(
        sbm.offdiagonals, sbm.rowindices, scheduler, coloringalgorithm
    )
    transposeoffdiagonalcolors = _symmetricblockcolors(
        sbm.offdiagonals, sbm.colindices, scheduler, coloringalgorithm
    )
    for color in offdiagonalcolors
        sort!(color; by=i -> (first(rowindices[i]), first(colindices[i])))
    end
    for color in transposeoffdiagonalcolors
        sort!(color; by=i -> (first(colindices[i]), first(rowindices[i])))
    end

    return SymmetricVariableBlockCompressedRowStorage{T,D,M,P,typeof(scheduler)}(
        sbm.diagonals,
        sbm.offdiagonals,
        diagonalindices,
        rowindices,
        colindices,
        roworder,
        rowptr,
        colorder,
        colptr,
        diagonalcolors,
        offdiagonalcolors,
        transposeoffdiagonalcolors,
        size(sbm),
        scheduler,
    )
end

function _symmetricvbcrs_mul!(y, A, x, α, β, conjugated)
    if iszero(β)
        fill!(y, zero(eltype(y)))
    else
        y .*= β
    end

    rowcolors = isserial(A.scheduler) ? (A.roworder,) : A.offdiagonalcolors
    for color in rowcolors
        @tasks for i in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            block = A.offdiagonals[i]
            rows = A.rowindices[i]
            cols = A.colindices[i]
            direct = conjugated ? adjoint(transpose(block)) : block
            LinearAlgebra.mul!(view(y, rows), direct, view(x, cols), α, true)
        end
    end
    colcolors = isserial(A.scheduler) ? (A.colorder,) : A.transposeoffdiagonalcolors
    for color in colcolors
        @tasks for i in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            block = A.offdiagonals[i]
            rows = A.rowindices[i]
            cols = A.colindices[i]
            reverse = conjugated ? adjoint(block) : transpose(block)
            LinearAlgebra.mul!(view(y, cols), reverse, view(x, rows), α, true)
        end
    end
    for color in A.diagonalcolors
        @tasks for i in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            block = A.diagonals[i]
            rows = A.diagonalindices[i]
            diagonal = conjugated ? adjoint(block) : block
            LinearAlgebra.mul!(view(y, rows), diagonal, view(x, rows), α, true)
        end
    end
    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector,
    A::SymmetricVariableBlockCompressedRowStorage,
    x::AbstractVector,
    α::Number,
    β::Number,
)
    return _symmetricvbcrs_mul!(y, A, x, α, β, false)
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector,
    A::LinearMaps.TransposeMap{<:Any,<:SymmetricVariableBlockCompressedRowStorage},
    x::AbstractVector,
    α::Number,
    β::Number,
)
    return _symmetricvbcrs_mul!(y, A.lmap, x, α, β, false)
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector,
    A::LinearMaps.AdjointMap{<:Any,<:SymmetricVariableBlockCompressedRowStorage},
    x::AbstractVector,
    α::Number,
    β::Number,
)
    return _symmetricvbcrs_mul!(y, A.lmap, x, α, β, true)
end

function SparseArrays.nnz(A::SymmetricVariableBlockCompressedRowStorage)
    return sum(_nnz, A.diagonals; init=0) + 2 * sum(_nnz, A.offdiagonals; init=0)
end

function rowcolvals(A::SymmetricVariableBlockCompressedRowStorage)
    rows = Int[]
    cols = Int[]
    vals = eltype(A)[]
    for i in eachindex(A.diagonals)
        block = A.diagonals[i]
        indices = A.diagonalindices[i]
        _pushblocktoarrays!(rows, cols, block, indices, indices, vals)
    end
    for i in A.roworder
        block = A.offdiagonals[i]
        r = A.rowindices[i]
        c = A.colindices[i]
        _pushblocktoarrays!(rows, cols, block, r, c, vals)
        _pushblocktoarrays!(rows, cols, transpose(block), c, r, vals)
    end
    return rows, cols, vals
end
