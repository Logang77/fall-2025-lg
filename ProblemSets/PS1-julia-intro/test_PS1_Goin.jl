using Test
using JLD
using Random
using Distributions
using LinearAlgebra
include("PS1_Goin.jl")

@testset "PS1_Goin Tests" begin
    
    @testset "Problem 1 - Matrix Operations" begin
        # Test q1 function outputs
        A, B, C, D = q1()
        
        @test size(A) == (10, 7)  # Check matrix dimensions
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)
        
        # Test matrix properties
        @test all(-5 .<= A .<= 10)  # Check A is within bounds
        @test size(C, 1) == 5  # Check C has correct number of rows
        @test all(D .<= 0)  # Check D contains only non-positive numbers
        
        # Test file creation
        @test isfile("matrixpractice.jld")
        @test isfile("firstmatrix.jld")
        @test isfile("Cmatrix.csv")
        @test isfile("Dmatrix.dat")
    end

    @testset "Problem 2 - Matrix Addition" begin
        # Test with small matrices
        test_A = [1.0 2.0; 3.0 4.0]
        test_B = [5.0 6.0; 7.0 8.0]
        result = q2(test_A, test_B)
        @test result == [6.0 8.0; 10.0 12.0]

        # Test with zero matrix
        zeros_mat = zeros(2, 2)
        @test q2(test_A, zeros_mat) == test_A
        
        # Test with matrices of different sizes should throw error
        @test_throws DimensionMismatch q2(ones(2,2), ones(3,3))
    end

    @testset "File Operations" begin
        # Test if we can load saved matrices
        @test isfile("matrixpractice.jld")
        saved_data = load("matrixpractice.jld")
        @test haskey(saved_data, "A")
        @test haskey(saved_data, "B")
        @test haskey(saved_data, "C")
        @test haskey(saved_data, "D")
        
        # Test dimensions of loaded matrices
        @test size(saved_data["A"]) == (10, 7)
        @test size(saved_data["B"]) == (10, 7)
        @test size(saved_data["C"]) == (5, 7)
        @test size(saved_data["D"]) == (10, 7)
    end

    @testset "Matrix Properties" begin
        A, B, C, D = q1()
        
        # Test matrix E (vec operator)
        E = B[:]  # This should match what's done in q1
        @test length(E) == prod(size(B))
        @test E isa Vector
        
        # Test matrix F (3D array)
        F = cat(A, B, dims=3)
        @test size(F, 3) == 2
        @test size(F, 1) == 10
        @test size(F, 2) == 7
        
        # Test Kronecker product
        G = kron(B, C)
        @test size(G) == (size(B,1)*size(C,1), size(B,2)*size(C,2))
    end
end
