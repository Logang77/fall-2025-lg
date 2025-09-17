using Test
using Optim
using GLM
using DataFrames
using LinearAlgebra
include("PS2_Goin.jl")

@testset "PS2_Goin Tests" begin
    
    @testset "Question 1: Function Minimization" begin
        # Test the optimization result
        result = optimize(minusf, [0.0], BFGS())
        minimizer = Optim.minimizer(result)[1]
        
        # Test that we found a minimum (gradient should be near zero)
        @test Optim.converged(result)
        
        # Test function values
        @test f([minimizer]) ≈ -f([-minimizer])  # Test symmetry
        @test minusf([minimizer]) < minusf([10.0])  # Test that we found a minimum
    end

    @testset "Question 2: OLS Tests" begin
        # Create small test data
        X_test = [1.0 2.0; 1.0 3.0; 1.0 4.0]
        y_test = [2.0, 4.0, 6.0]
        
        # Test OLS function
        test_beta = [0.0, 2.0]
        test_ssr = ols(test_beta, X_test, y_test)
        @test test_ssr ≥ 0  # SSR should be non-negative
        
        # Test closed form solution
        test_bols = inv(X_test'X_test)*X_test'y_test
        @test ols(test_bols, X_test, y_test) ≤ ols(test_beta, X_test, y_test)  # Closed form should minimize SSR
    end

    @testset "Question 3: Logit Tests" begin
        # Create small test data
        X_test = [1.0 0.0; 1.0 1.0]
        y_test = [0.0, 1.0]
        
        # Test logit function
        logit_val = logit([0.0, 0.0], X_test, y_test)
        @test logit_val ≥ 0  # Negative log-likelihood should be non-negative
        
        # Test that probability predictions are between 0 and 1
        beta_test = [0.0, 1.0]
        probs = 1 ./ (1 .+ exp.(-X_test * beta_test))
        @test all(0 .≤ probs .≤ 1)
    end

    @testset "Question 5: Multinomial Logit Tests" begin
        # Create small test data
        X_test = [1.0 0.0; 1.0 1.0; 1.0 0.0]
        y_test = [1, 2, 3]
        
        # Test mlogit function with zero coefficients
        alpha_test = zeros(2 * 2)  # 2 features, 2 non-base categories
        loglike = mlogit(alpha_test, X_test, y_test)
        @test loglike ≥ 0  # Negative log-likelihood should be non-negative
        
        # Test dimensions of reshaped parameters
        alpha_mat = reshape(alpha_test, 2, 2)
        @test size(alpha_mat) == (2, 2)
    end

    @testset "Main Function Test" begin
        # Test that main() runs without errors and returns a dictionary
        results = main()
        @test results isa Dict
        
        # Test that all expected keys are present
        expected_keys = ["q1_minimizer", "q1_minimum", "q2_ols", "q2_ols_se", 
                        "q3_logit", "q4_glm", "q5_mlogit"]
        for key in expected_keys
            @test haskey(results, key)
        end
    end
end