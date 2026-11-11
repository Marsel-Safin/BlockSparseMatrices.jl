using Test
using BlockSparseMatrices
using LinearAlgebra
using SparseArrays
using OhMyThreads

@testset "SymmetricVariableBlockCompressedRowStorage" begin
    d1 = ComplexF64[2 + im 1 + 2im; 1 + 2im 3 - im]
    d2 = reshape(ComplexF64[4 + 3im], 1, 1)
    b = reshape(ComplexF64[1 + 3im, -2 + im], 2, 1)
    x = ComplexF64[1 - im, 2 + im, -1 + 2im]

    for scheduler in (SerialScheduler(), DynamicScheduler())
        sbm = SymmetricBlockMatrix(
            [d1, d2], [[1, 2], [3]], [b], [[1, 2]], [[3]], (3, 3); scheduler=scheduler
        )
        A = SymmetricVariableBlockCompressedRowStorage(sbm; scheduler=scheduler)
        full = sparse(sbm)

        @test A.diagonals === sbm.diagonals
        @test A.offdiagonals === sbm.offdiagonals
        @test size(A) == (3, 3)
        @test nnz(A) == nnz(sbm)
        @test sparse(A) ≈ full
        @test A * x ≈ full * x
        @test transpose(A) * x ≈ transpose(full) * x
        @test adjoint(A) * x ≈ adjoint(full) * x
        @test !(transpose(A) * x ≈ adjoint(A) * x)

        for map in (A, transpose(A), adjoint(A))
            y = fill(2 + im, 3)
            expected = (2 - im) .* y .+ (3 + im) .* (map * x)
            mul!(y, map, x, 3 + im, 2 - im)
            @test y ≈ expected
        end
    end

    for scheduler in (SerialScheduler(), DynamicScheduler())
        diagonal_only = SymmetricBlockMatrix{ComplexF64}(
            [d1],
            [[1, 2]],
            Matrix{ComplexF64}[],
            Vector{Int}[],
            Vector{Int}[],
            (3, 3);
            scheduler=scheduler,
        )
        offdiagonal_only = SymmetricBlockMatrix{ComplexF64}(
            Matrix{ComplexF64}[],
            Vector{Int}[],
            [b],
            [[1, 2]],
            [[3]],
            (3, 3);
            scheduler=scheduler,
        )
        empty_matrix = SymmetricBlockMatrix{ComplexF64}(
            Matrix{ComplexF64}[],
            Vector{Int}[],
            Matrix{ComplexF64}[],
            Vector{Int}[],
            Vector{Int}[],
            (3, 3);
            scheduler=scheduler,
        )
        for sbm in (diagonal_only, offdiagonal_only, empty_matrix)
            A = SymmetricVariableBlockCompressedRowStorage(sbm; scheduler=scheduler)
            full = sparse(sbm)
            @test nnz(A) == nnz(sbm)
            @test sparse(A) == full
            @test A * x ≈ full * x
            @test transpose(A) * x ≈ transpose(full) * x
            @test adjoint(A) * x ≈ adjoint(full) * x
        end
    end

    for scheduler in (SerialScheduler(), DynamicScheduler())
        blocks = [
            reshape(ComplexF64[1 + im, 2 - im], 2, 1),
            reshape(ComplexF64[3 + 2im, 4 - 3im], 2, 1),
        ]
        sbm = SymmetricBlockMatrix(
            Matrix{ComplexF64}[],
            Vector{Int}[],
            blocks,
            [[1, 2], [2, 3]],
            [[4], [4]],
            (4, 4);
            scheduler=scheduler,
        )
        A = SymmetricVariableBlockCompressedRowStorage(sbm; scheduler=scheduler)
        full = sparse(sbm)
        x4 = ComplexF64[1 + im, 2 - im, 3 + 2im, 4 - im]
        @test A * x4 ≈ full * x4
        @test transpose(A) * x4 ≈ transpose(full) * x4
        @test adjoint(A) * x4 ≈ adjoint(full) * x4
    end

    sbm = SymmetricBlockMatrix{ComplexF64}(
        Matrix{ComplexF64}[],
        Vector{Int}[],
        Matrix{ComplexF64}[],
        Vector{Int}[],
        Vector{Int}[],
        (0, 0),
    )
    A = SymmetricVariableBlockCompressedRowStorage(sbm)
    @test size(A) == (0, 0)
    @test nnz(A) == 0
    @test sparse(A) == spzeros(ComplexF64, 0, 0)

    diagonal = ComplexF64[1 + im 2im 3; 2im 4 - im 5 + im; 3 5 + im 6]
    offdiagonal = ComplexF64[2 - im 1 + im; 3 + 2im 4 - im]
    x8 = ComplexF64[1 + im, 2 - im, 3 + im, 4 - im, 5 + im, 6 - im, 7 + im, 8 - im]
    for scheduler in (SerialScheduler(), DynamicScheduler())
        for (diagonal_indices, row_indices, col_indices) in
            (([4, 6, 8], [1, 3], [5, 7]), ([8, 4, 6], [3, 1], [7, 5]))
            sbm = SymmetricBlockMatrix(
                [diagonal],
                [diagonal_indices],
                [offdiagonal],
                [row_indices],
                [col_indices],
                (8, 8);
                scheduler=scheduler,
            )
            A = SymmetricVariableBlockCompressedRowStorage(sbm; scheduler=scheduler)
            full = sparse(sbm)
            @test A.diagonalindices === sbm.diagonalindices
            @test A.rowindices === sbm.rowindices
            @test A.colindices === sbm.colindices
            @test nnz(A) == nnz(sbm)
            @test sparse(A) == full
            @test A * x8 ≈ full * x8
            @test transpose(A) * x8 ≈ transpose(full) * x8
            @test adjoint(A) * x8 ≈ adjoint(full) * x8
        end
    end

    for scheduler in (SerialScheduler(), DynamicScheduler())
        blocks = [
            reshape(ComplexF64[1 + im, 2 - im], 2, 1),
            reshape(ComplexF64[3 + 2im, 4 - im], 2, 1),
        ]
        sbm = SymmetricBlockMatrix(
            Matrix{ComplexF64}[],
            Vector{Int}[],
            blocks,
            [[1, 3], [3, 5]],
            [[7], [7]],
            (8, 8);
            scheduler=scheduler,
        )
        A = SymmetricVariableBlockCompressedRowStorage(sbm; scheduler=scheduler)
        full = sparse(sbm)
        @test sparse(A) == full
        @test A * x8 ≈ full * x8
        @test adjoint(A) * x8 ≈ adjoint(full) * x8
    end

    out_of_bounds = SymmetricBlockMatrix{ComplexF64}(
        [diagonal], [[4, 6, 9]], Matrix{ComplexF64}[], Vector{Int}[], Vector{Int}[], (8, 8)
    )
    @test_throws ArgumentError SymmetricVariableBlockCompressedRowStorage(out_of_bounds)
end

@testset "Compare symmetric VBCRS with SymmetricBlockMatrix" begin
    diagonal = ComplexF64[
        1 + im 2 + 3im 3 - im;
        2 + 3im 4 - im 5 + 2im;
        3 - im 5 + 2im 6 + im
    ]
    offdiagonal = ComplexF64[1 + im 2 - im; -3 + 2im 4 + 3im]
    overlapblock = reshape(ComplexF64[2 + im, 3 - im], 2, 1)
    x8 = ComplexF64[1 + im, 2 - im, 3 + im, 4 - im, 5 + im, 6 - im, 7 + im, 8 - im]

    for scheduler in (SerialScheduler(), DynamicScheduler())
        matrices = (
            SymmetricBlockMatrix{ComplexF64}(
                [diagonal],
                [[4, 6, 8]],
                [offdiagonal],
                [[1, 3]],
                [[5, 7]],
                (8, 8);
                scheduler=scheduler,
            ),
            SymmetricBlockMatrix{ComplexF64}(
                [diagonal],
                [[8, 4, 6]],
                Matrix{ComplexF64}[],
                Vector{Int}[],
                Vector{Int}[],
                (8, 8);
                scheduler=scheduler,
            ),
            SymmetricBlockMatrix{ComplexF64}(
                Matrix{ComplexF64}[],
                Vector{Int}[],
                [overlapblock, overlapblock],
                [[1, 3], [3, 5]],
                [[7], [7]],
                (8, 8);
                scheduler=scheduler,
            ),
            SymmetricBlockMatrix{ComplexF64}(
                Matrix{ComplexF64}[],
                Vector{Int}[],
                Matrix{ComplexF64}[],
                Vector{Int}[],
                Vector{Int}[],
                (8, 8);
                scheduler=scheduler,
            ),
            SymmetricBlockMatrix{ComplexF64}(
                Matrix{ComplexF64}[],
                Vector{Int}[],
                Matrix{ComplexF64}[],
                Vector{Int}[],
                Vector{Int}[],
                (0, 0);
                scheduler=scheduler,
            ),
        )

        for sbm in matrices
            A = SymmetricVariableBlockCompressedRowStorage(sbm; scheduler=scheduler)
            x = x8[1:size(sbm, 2)]
            @test nnz(A) == nnz(sbm)
            @test sparse(A) == sparse(sbm)

            for operation in (identity, transpose, adjoint)
                @test operation(A) * x ≈ operation(sbm) * x
                y = fill(1 + im, size(sbm, 1))
                expected = copy(y)
                mul!(y, operation(A), x, 2 + im, -1 + 2im)
                mul!(expected, operation(sbm), x, 2 + im, -1 + 2im)
                @test y ≈ expected
            end
        end
    end
end
