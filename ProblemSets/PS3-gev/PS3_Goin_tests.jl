using Test
using LinearAlgebra
using Random
using CSV
using HTTP
using DataFrames
using Distributions  # for Gumbel distribution
using Optim         # for optimization, GLM, FreqTables

cd(@__DIR__)

include("PS3_Goin_Source.jl")

@testset "PS3 Question 1 Tests" begin
    
    @testset "Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        
        # Test dimensions
        @test size(X, 2) == 3  # age, white, collgrad
        @test size(Z, 2) == 8  # 8 wage variables
        @test size(X, 1) == size(Z, 1) == length(y)  # same number of observations
        
        # Test data types
        @test eltype(X) <: Real
        @test eltype(Z) <: Real
        @test eltype(y) <: Integer
        
        # Test value ranges
        @test all(X[:, 2] .∈ ([0, 1]))  # white indicator
        @test all(X[:, 3] .∈ ([0, 1]))  # college grad indicator
        @test all(1 .<= y .<= 8)  # occupation codes
    end
    
    @testset "Multinomial Logit Function" begin
        # Create small test data matching the actual structure
        X_test = [25.0 1.0 0.0; 30.0 0.0 1.0; 35.0 1.0 1.0]  # 3 obs, 3 vars (age, white, collgrad)
        Z_test = [1.0 2.0 3.0 1.5 2.5 3.5 1.2 2.8;   # 3 obs, 8 wage alternatives
                  2.0 3.0 1.0 2.5 1.5 2.8 3.2 1.8;
                  3.0 1.0 2.0 3.5 2.8 1.5 2.2 3.8]
        y_test = [1, 2, 3]
        
        # Test with zero parameters (22 total: 21 alphas + 1 gamma)
        theta_zero = zeros(22)
        ll_zero = mlogit_with_Z(theta_zero, X_test, Z_test, y_test)
        @test ll_zero > 0  # negative log-likelihood should be positive
        @test !isnan(ll_zero)  # should not be NaN
        @test !isinf(ll_zero)  # should not be Inf
        
        # Test parameter structure
        @test length(theta_zero) == 22  # 3*7 + 1 = 22
        
        # Test with small positive gamma (wage coefficient)
        theta_gamma = vcat(zeros(21), 0.1)
        ll_gamma = mlogit_with_Z(theta_gamma, X_test, Z_test, y_test)
        @test ll_gamma > 0
        @test ll_gamma != ll_zero  # should be different from zero gamma
    end
    
    @testset "Optimization" begin
        # Create test data with 8 alternatives to match the real problem
        Random.seed!(1234)
        N = 100  # smaller sample for testing
        K = 3    # age, white, collgrad
        J = 8    # 8 occupations
        
        # Generate X (individual characteristics)
        X_sim = [25 .+ 10*rand(N) rand(N) .< 0.5 rand(N) .< 0.3]  # age, white, collgrad
        
        # Generate Z (alternative-specific wages)
        Z_sim = 1 .+ 2*rand(N, J)  # log wages
        
        # Create simple choices (for testing)
        y_sim = rand(1:J, N)
        
        # Test that optimization runs without error
        try
            result = optimize_mlogit(X_sim, Z_sim, y_sim)
            @test length(result) == 22  # correct number of parameters (21 alphas + 1 gamma)
            @test !any(isnan.(result))  # no NaN values
            @test !any(isinf.(result))  # no Inf values
        catch e
            @test false  # optimization should not fail
        end
    end
    
    @testset "Numerical Properties" begin
        # Create simple test data
        X_test = [25.0 1.0 0.0; 30.0 0.0 1.0]  # 2 obs, 3 vars
        Z_test = ones(2, 8)  # 2 obs, 8 alternatives
        y_test = [1, 2]
        
        # Test parameter scaling
        theta_small = 0.001 * ones(22)  # 21 alphas + 1 gamma
        ll_small = mlogit_with_Z(theta_small, X_test, Z_test, y_test)
        @test !isnan(ll_small)  # should not produce NaN
        @test !isinf(ll_small)  # should not produce Inf
        @test ll_small > 0      # negative log-likelihood should be positive
        
        # Test with moderate parameters
        theta_moderate = 0.1 * randn(22)
        ll_moderate = mlogit_with_Z(theta_moderate, X_test, Z_test, y_test)
        @test !isnan(ll_moderate)
        @test !isinf(ll_moderate)
        @test ll_moderate > 0
    end
    
    @testset "Real Data Integration" begin
        # Test with actual data to ensure compatibility
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        
        # Test that function works with real data dimensions
        theta_test = zeros(22)  # 21 alphas + 1 gamma
        ll_test = mlogit_with_Z(theta_test, X, Z, y)
        @test !isnan(ll_test)
        @test !isinf(ll_test)
        @test ll_test > 0
        
        # Test that the optimization setup is correct
        startvals = [2*rand(7*size(X,2)).-1; 0.1]
        @test length(startvals) == 22  # correct number of starting values
    end
end
