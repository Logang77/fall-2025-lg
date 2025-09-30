using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

cd(@__DIR__)

Random.seed!(1234)

include("PS4_Goin_Source.jl")

@testset "PS4 Mixture Models Tests" begin

    # Test 1: Data Loading Function
    @testset "Data Loading Tests" begin
        println("Testing data loading...")
        
        @test_nowarn df, X, Z, y = load_data()
        df, X, Z, y = load_data()
        
        # Check data dimensions
        @test size(X, 2) == 3  # age, white, collgrad
        @test size(Z, 2) == 8  # 8 wage alternatives
        @test length(y) == size(X, 1)  # same number of observations
        @test length(y) == size(Z, 1)  # same number of observations
        
        # Check data types
        @test isa(X, Matrix)
        @test isa(Z, Matrix)
        @test isa(y, Vector)
        @test isa(df, DataFrame)
        
        # Check that choices are valid (1-8)
        @test minimum(y) >= 1
        @test maximum(y) <= 8
        @test length(unique(y)) <= 8
        
        println("✓ Data loading tests passed")
    end

    # Test 2: Multinomial Logit with Alternative-Specific Covariates
    @testset "Multinomial Logit Tests" begin
        println("Testing multinomial logit function...")
        
        # Load test data
        df, X, Z, y = load_data()
        
        # Create test parameters
        K = size(X, 2)  # 3
        J = length(unique(y))  # 8
        n_params = K * (J - 1) + 1  # 21 alphas + 1 gamma = 22
        theta_test = rand(n_params) * 0.1  # Small random parameters
        
        # Test function execution
        @test_nowarn ll = mlogit_with_Z(theta_test, X, Z, y)
        ll = mlogit_with_Z(theta_test, X, Z, y)
        
        # Check output properties
        @test isa(ll, Number)
        @test isfinite(ll)
        @test ll > 0  # Negative log-likelihood should be positive
        
        # Test with zero parameters (should give log(J) per observation)
        theta_zero = zeros(n_params)
        ll_zero = mlogit_with_Z(theta_zero, X, Z, y)
        expected_ll = length(y) * log(J)
        @test abs(ll_zero - expected_ll) < 1e-10
        
        # Test parameter sensitivity
        theta_test2 = theta_test .+ 0.01
        ll2 = mlogit_with_Z(theta_test2, X, Z, y)
        @test ll2 != ll  # Different parameters should give different likelihood
        
        println("✓ Multinomial logit tests passed")
    end

    # Test 3: Quadrature Functions
    @testset "Quadrature Tests" begin
        println("Testing quadrature functions...")
        
        # Test that quadrature functions run without error
        @test_nowarn practice_quadrature()
        @test_nowarn variance_quadrature()
        
        # Test lgwt function is available
        @test_nowarn nodes, weights = lgwt(7, -4, 4)
        nodes, weights = lgwt(7, -4, 4)
        
        @test length(nodes) == 7
        @test length(weights) == 7
        @test all(isfinite.(nodes))
        @test all(isfinite.(weights))
        @test all(weights .> 0)  # Weights should be positive
        
        println("✓ Quadrature tests passed")
    end

    # Test 4: Monte Carlo Functions
    @testset "Monte Carlo Tests" begin
        println("Testing Monte Carlo functions...")
        
        # Test Monte Carlo practice function
        @test_nowarn practice_monte_carlo()
        
        println("✓ Monte Carlo tests passed")
    end

    # Test 5: Mixed Logit with Quadrature
    @testset "Mixed Logit Quadrature Tests" begin
        println("Testing mixed logit with quadrature...")
        
        # Load test data  
        df, X, Z, y = load_data()
        
        # Create test parameters for mixed logit
        K = size(X, 2)  # 3
        J = length(unique(y))  # 8
        n_alpha = K * (J - 1)  # 21
        theta_mixed = [rand(n_alpha) * 0.1; 0.1; 0.5]  # alphas + mu_gamma + sigma_gamma
        
        # Test with small sample to speed up testing
        n_test = 100
        X_test = X[1:n_test, :]
        Z_test = Z[1:n_test, :]
        y_test = y[1:n_test]
        
        # Test quadrature nodes
        nodes, weights = lgwt(7, -4, 4)
        
        # Test function execution
        @test_nowarn ll = mixed_logit_quad(theta_mixed, X_test, Z_test, y_test, nodes)
        ll = mixed_logit_quad(theta_mixed, X_test, Z_test, y_test, nodes)
        
        # Check output properties
        @test isa(ll, Number)
        @test isfinite(ll)
        @test ll > 0  # Negative log-likelihood should be positive
        
        println("✓ Mixed logit quadrature tests passed")
    end

    # Test 6: Mixed Logit with Monte Carlo
    @testset "Mixed Logit Monte Carlo Tests" begin
        println("Testing mixed logit with Monte Carlo...")
        
        # Load test data
        df, X, Z, y = load_data()
        
        # Create test parameters
        K = size(X, 2)  # 3
        J = length(unique(y))  # 8
        n_alpha = K * (J - 1)  # 21
        theta_mixed = [rand(n_alpha) * 0.1; 0.1; 0.5]  # alphas + mu_gamma + sigma_gamma
        
        # Test with small sample and few draws
        n_test = 50
        D_test = 100  # Number of Monte Carlo draws
        X_test = X[1:n_test, :]
        Z_test = Z[1:n_test, :]
        y_test = y[1:n_test]
        
        # Test function execution
        @test_nowarn ll = mixed_logit_mc(theta_mixed, X_test, Z_test, y_test, D_test)
        ll = mixed_logit_mc(theta_mixed, X_test, Z_test, y_test, D_test)
        
        # Check output properties
        @test isa(ll, Number)
        @test isfinite(ll)
        @test ll > 0  # Negative log-likelihood should be positive
        
        println("✓ Mixed logit Monte Carlo tests passed")
    end

    # Test 7: Optimization Functions
    @testset "Optimization Tests" begin
        println("Testing optimization functions...")
        
        # Load test data
        df, X, Z, y = load_data()
        
        # Test with small sample for speed
        n_test = 100
        X_test = X[1:n_test, :]
        Z_test = Z[1:n_test, :]
        y_test = y[1:n_test]
        
        # Test multinomial logit optimization
        @test_nowarn theta_hat, se_hat = optimize_mlogit(X_test, Z_test, y_test)
        theta_hat, se_hat = optimize_mlogit(X_test, Z_test, y_test)
        
        # Check output dimensions
        K = size(X_test, 2)
        J = length(unique(y_test))
        expected_params = K * (J - 1) + 1
        @test length(theta_hat) == expected_params
        @test length(se_hat) == expected_params
        
        # Check that estimates are finite
        @test all(isfinite.(theta_hat))
        @test all(isfinite.(se_hat))
        @test all(se_hat .> 0)  # Standard errors should be positive
        
        # Test mixed logit optimization setup (these don't actually run optimization)
        nodes = [1.0, 2.0, 3.0]  # dummy nodes for testing
        @test_nowarn startvals = optimize_mixed_logit_quad(X_test, Z_test, y_test, nodes)
        @test_nowarn startvals = optimize_mixed_logit_mc(X_test, Z_test, y_test)
        
        println("✓ Optimization tests passed")
    end

    # Test 8: Parameter Validation
    @testset "Parameter Validation Tests" begin
        println("Testing parameter validation...")
        
        # Load test data
        df, X, Z, y = load_data()
        K = size(X, 2)
        J = length(unique(y))
        
        # Test with wrong parameter dimensions
        theta_wrong = rand(10)  # Wrong number of parameters
        @test_throws BoundsError mlogit_with_Z(theta_wrong, X, Z, y)
        
        # Test with correct dimensions
        theta_correct = rand(K * (J - 1) + 1)
        @test_nowarn mlogit_with_Z(theta_correct, X, Z, y)
        
        println("✓ Parameter validation tests passed")
    end

    # Test 9: Numerical Stability
    @testset "Numerical Stability Tests" begin
        println("Testing numerical stability...")
        
        # Load test data
        df, X, Z, y = load_data()
        K = size(X, 2)
        J = length(unique(y))
        
        # Test with extreme parameters
        theta_large = ones(K * (J - 1) + 1) * 10  # Large parameters
        @test_nowarn ll = mlogit_with_Z(theta_large, X, Z, y)
        ll = mlogit_with_Z(theta_large, X, Z, y)
        @test isfinite(ll)
        
        # Test with very small parameters  
        theta_small = ones(K * (J - 1) + 1) * 1e-10
        @test_nowarn ll = mlogit_with_Z(theta_small, X, Z, y)
        ll = mlogit_with_Z(theta_small, X, Z, y)
        @test isfinite(ll)
        
        println("✓ Numerical stability tests passed")
    end

    # Test 10: Integration Tests
    @testset "Integration Tests" begin
        println("Testing full workflow...")
        
        # Test the main wrapper function doesn't crash
        # Note: This will actually run optimization, so it might take time
        # Uncomment if you want to test the full workflow
        # @test_nowarn allwrap()
        
        println("✓ Integration tests completed")
    end

end

println("\n🎉 All PS4 unit tests completed successfully!")