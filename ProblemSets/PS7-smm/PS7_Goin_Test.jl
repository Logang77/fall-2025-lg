using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)
include("PS7_Goin_Source.jl")

@testset "PS7 GMM and SMM Estimation Tests" begin

    #--------------------------------------------------------------------------
    # TEST SET 1: DATA LOADING AND PREPARATION
    #--------------------------------------------------------------------------
    
    @testset "Data Loading Functions" begin
        
        @testset "load_data() tests" begin
            # Test with actual URL
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
            
            # Test basic functionality
            @test_nowarn df, X, y = load_data(url)
            
            df, X, y = load_data(url)
            
            # Test structure
            @test isa(df, DataFrame)
            @test isa(X, Matrix)
            @test isa(y, Vector)
            
            # Test dimensions
            @test nrow(df) > 0
            @test size(X, 1) == nrow(df)
            @test length(y) == nrow(df)
            @test size(X, 2) == 4  # intercept + age + race + collgrad
            
            # Test X matrix structure
            @test all(X[:, 1] .== 1)  # First column should be intercept
            @test all(X[:, 3] .∈ Ref([0, 1]))  # Race indicator should be binary
            @test all(X[:, 4] .∈ Ref([0, 1]))  # College grad should be binary
            
            # Test that y is log wage (should be positive but not huge)
            @test all(isfinite.(y))
            @test all(y .> 0)  # Log wages should be positive for this dataset
            @test maximum(y) < 10  # Reasonable upper bound for log wages
        end
        
        @testset "prepare_occupation_data() tests" begin
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
            df_orig = CSV.read(HTTP.get(url).body, DataFrame)
            
            # Test basic functionality
            @test_nowarn df, X, y = prepare_occupation_data(df_orig)
            
            df, X, y = prepare_occupation_data(df_orig)
            
            # Test structure
            @test isa(df, DataFrame)
            @test isa(X, Matrix)
            @test isa(y, Vector)
            
            # Test occupation categories (should be 1-7 after collapsing)
            unique_occs = unique(y)
            @test all(occ -> occ in 1:7, unique_occs)
            @test maximum(y) <= 7
            @test minimum(y) >= 1
            
            # Test X matrix
            @test size(X, 2) == 4  # intercept + age + white + collgrad
            @test all(X[:, 1] .== 1)  # Intercept
            @test all(X[:, 3] .∈ Ref([0, 1]))  # White indicator
            @test all(X[:, 4] .∈ Ref([0, 1]))  # College grad indicator
            
            # Test white variable creation
            @test haskey(df, :white)
            @test all(df.white .∈ Ref([0, 1]))
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 2: OLS VIA GMM (QUESTION 1)
    #--------------------------------------------------------------------------
    
    @testset "OLS via GMM Tests" begin
        
        # Create test data
        N = 100
        K = 3
        Random.seed!(123)
        X_test = [ones(N) randn(N) randn(N)]
        β_true = [1.0, 0.5, -0.3]
        y_test = X_test * β_true + 0.1 * randn(N)
        
        @testset "ols_gmm() tests" begin
            # Test basic functionality
            @test_nowarn obj = ols_gmm(β_true, X_test, y_test)
            
            obj_true = ols_gmm(β_true, X_test, y_test)
            obj_wrong = ols_gmm(zeros(K), X_test, y_test)
            
            # True parameters should give lower objective than wrong parameters
            @test obj_true < obj_wrong
            
            # Test with exact solution
            β_ols = X_test \ y_test
            obj_ols = ols_gmm(β_ols, X_test, y_test)
            
            # OLS solution should minimize objective
            @test obj_ols <= obj_true + 1e-10
            
            # Test that objective is finite and positive
            @test isfinite(obj_ols)
            @test obj_ols >= 0
            
            # Test dimension mismatch handling
            @test_throws BoundsError ols_gmm([1.0, 2.0], X_test, y_test)  # Wrong β dimension
        end
        
        @testset "OLS GMM optimization" begin
            # Test that optimization recovers true parameters
            result = optimize(b -> ols_gmm(b, X_test, y_test), 
                            randn(K), 
                            LBFGS(), 
                            Optim.Options(g_tol=1e-8))
            
            β_gmm = result.minimizer
            β_ols = X_test \ y_test
            
            # GMM should be close to OLS
            @test norm(β_gmm - β_ols) < 1e-6
            @test result.converged
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 3: MULTINOMIAL LOGIT MLE (QUESTION 2)
    #--------------------------------------------------------------------------
    
    @testset "Multinomial Logit MLE Tests" begin
        
        # Create small test data
        N = 50
        K = 3
        J = 4
        Random.seed!(456)
        X_test = [ones(N) randn(N) randn(N)]
        y_test = rand(1:J, N)
        α_test = randn(K * (J-1))
        
        @testset "mlogit_mle() tests" begin
            # Test basic functionality
            @test_nowarn ll = mlogit_mle(α_test, X_test, y_test)
            
            ll = mlogit_mle(α_test, X_test, y_test)
            
            # Log-likelihood should be finite and negative (since it's negative log-likelihood)
            @test isfinite(ll)
            @test ll > 0  # We return negative log-likelihood
            
            # Test with different parameters
            ll2 = mlogit_mle(zeros(length(α_test)), X_test, y_test)
            @test ll != ll2  # Different parameters should give different likelihoods
            
            # Test dimension consistency
            @test_throws Exception mlogit_mle(α_test[1:end-1], X_test, y_test)  # Wrong dimension
        end
        
        @testset "mlogit_gmm() tests" begin
            # Test just-identified GMM
            @test_nowarn obj = mlogit_gmm(α_test, X_test, y_test)
            
            obj = mlogit_gmm(α_test, X_test, y_test)
            
            # Objective should be finite and non-negative
            @test isfinite(obj)
            @test obj >= 0
            
            # Test that different parameters give different objectives
            obj2 = mlogit_gmm(zeros(length(α_test)), X_test, y_test)
            @test obj != obj2
        end
        
        @testset "mlogit_gmm_overid() tests" begin
            # Test over-identified GMM
            @test_nowarn obj = mlogit_gmm_overid(α_test, X_test, y_test)
            
            obj = mlogit_gmm_overid(α_test, X_test, y_test)
            
            # Objective should be finite and non-negative
            @test isfinite(obj)
            @test obj >= 0
            
            # Test parameter sensitivity
            obj2 = mlogit_gmm_overid(α_test .+ 0.1, X_test, y_test)
            @test obj != obj2
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 4: DATA SIMULATION (QUESTION 3)
    #--------------------------------------------------------------------------
    
    @testset "Data Simulation Tests" begin
        
        Random.seed!(789)
        
        @testset "sim_logit() tests" begin
            N = 1000
            J = 4
            
            # Test basic functionality
            @test_nowarn Y, X = sim_logit(N, J)
            
            Y, X = sim_logit(N, J)
            
            # Test dimensions
            @test length(Y) == N
            @test size(X, 1) == N
            @test size(X, 2) == 4  # Based on function implementation
            
            # Test choice range
            @test all(y -> y in 1:J, Y)
            @test minimum(Y) >= 1
            @test maximum(Y) <= J
            
            # Test X matrix structure
            @test all(X[:, 1] .== 1)  # First column should be intercept
            @test all(isfinite.(X))
            
            # Test that choices are distributed across alternatives
            choice_counts = [sum(Y .== j) for j in 1:J]
            @test all(count -> count > 0, choice_counts)  # Each choice should appear
        end
        
        @testset "sim_logit_w_gumbel() tests" begin
            N = 1000
            J = 4
            
            # Test basic functionality
            @test_nowarn Y, X = sim_logit_w_gumbel(N, J)
            
            Y, X = sim_logit_w_gumbel(N, J)
            
            # Test dimensions and ranges (same as regular sim_logit)
            @test length(Y) == N
            @test size(X, 1) == N
            @test all(y -> y in 1:J, Y)
            @test all(X[:, 1] .== 1)
            
            # Test that both methods give similar choice distributions (roughly)
            Y1, X1 = sim_logit(N, J)
            Y2, X2 = sim_logit_w_gumbel(N, J)
            
            freq1 = [sum(Y1 .== j)/N for j in 1:J]
            freq2 = [sum(Y2 .== j)/N for j in 1:J]
            
            # Choice frequencies should be somewhat similar (within reasonable bounds)
            @test maximum(abs.(freq1 - freq2)) < 0.2  # Allow for sampling variation
        end
        
        @testset "Simulation consistency tests" begin
            # Test that simulated data can be estimated
            N = 2000
            J = 4
            Y_sim, X_sim = sim_logit(N, J)
            
            # Test that we can run MLE on simulated data
            α_start = randn((size(X_sim, 2)) * (J-1))
            @test_nowarn mlogit_mle(α_start, X_sim, Y_sim)
            
            # The MLE should converge (though we don't test parameter recovery here for speed)
            ll = mlogit_mle(α_start, X_sim, Y_sim)
            @test isfinite(ll)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 5: SMM ESTIMATION (QUESTION 5)
    #--------------------------------------------------------------------------
    
    @testset "SMM Estimation Tests" begin
        
        # Create small test data for speed
        N = 100
        K = 3
        J = 4
        Random.seed!(101)
        X_test = [ones(N) randn(N) randn(N)]
        y_test = rand(1:J, N)
        α_test = randn(K * (J-1))
        
        @testset "mlogit_smm_overid() tests" begin
            D = 10  # Small number of draws for testing
            
            # Test basic functionality
            @test_nowarn obj = mlogit_smm_overid(α_test, X_test, y_test, D)
            
            obj = mlogit_smm_overid(α_test, X_test, y_test, D)
            
            # Objective should be finite and non-negative
            @test isfinite(obj)
            @test obj >= 0
            
            # Test with different number of simulation draws
            obj_more_draws = mlogit_smm_overid(α_test, X_test, y_test, D*2)
            @test isfinite(obj_more_draws)
            
            # Test parameter sensitivity
            obj2 = mlogit_smm_overid(α_test .+ 0.1, X_test, y_test, D)
            @test obj != obj2
            
            # Test reproducibility (should give same result with same seed)
            Random.seed!(101)
            obj_repeat = mlogit_smm_overid(α_test, X_test, y_test, D)
            # Note: Due to random seed in function, this might not be exactly equal
            @test isfinite(obj_repeat)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 6: INTEGRATION TESTS
    #--------------------------------------------------------------------------
    
    @testset "Integration Tests" begin
        
        @testset "Main function components" begin
            # Test that we can load data without errors
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
            @test_nowarn df, X_wage, y_wage = load_data(url)
            @test_nowarn df, X, y = prepare_occupation_data(df)
            
            # Test a simple optimization to ensure functions work together
            df, X_wage, y_wage = load_data(url)
            
            # Test OLS GMM optimization
            @test_nowarn result = optimize(b -> ols_gmm(b, X_wage[1:100, :], y_wage[1:100]), 
                                         randn(size(X_wage, 2)), 
                                         LBFGS(), 
                                         Optim.Options(g_tol=1e-4, iterations=100))
        end
        
        @testset "Workflow consistency" begin
            # Test that the different estimation methods are compatible
            N = 200
            J = 3
            Y_sim, X_sim = sim_logit(N, J)
            
            # Test that MLE and GMM can both be run on the same data
            α_start = randn((size(X_sim, 2)) * (J-1))
            
            @test_nowarn ll_mle = mlogit_mle(α_start, X_sim, Y_sim)
            @test_nowarn obj_gmm = mlogit_gmm(α_start, X_sim, Y_sim)
            @test_nowarn obj_gmm_overid = mlogit_gmm_overid(α_start, X_sim, Y_sim)
            
            ll_mle = mlogit_mle(α_start, X_sim, Y_sim)
            obj_gmm = mlogit_gmm(α_start, X_sim, Y_sim)
            obj_gmm_overid = mlogit_gmm_overid(α_start, X_sim, Y_sim)
            
            @test all(isfinite.([ll_mle, obj_gmm, obj_gmm_overid]))
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 7: EDGE CASES AND ROBUSTNESS
    #--------------------------------------------------------------------------
    
    @testset "Edge Cases and Robustness" begin
        
        @testset "Small sample tests" begin
            # Test with very small samples
            N_small = 10
            K = 2
            J = 3
            X_small = [ones(N_small) randn(N_small)]
            y_small = rand(1:J, N_small)
            α_small = randn(K * (J-1))
            
            # Functions should not crash with small samples
            @test_nowarn mlogit_mle(α_small, X_small, y_small)
            @test_nowarn mlogit_gmm(α_small, X_small, y_small)
            @test_nowarn mlogit_gmm_overid(α_small, X_small, y_small)
        end
        
        @testset "Boundary parameter tests" begin
            N = 50
            K = 2
            J = 3
            X_test = [ones(N) randn(N)]
            y_test = rand(1:J, N)
            
            # Test with zero parameters
            α_zero = zeros(K * (J-1))
            @test_nowarn mlogit_mle(α_zero, X_test, y_test)
            
            # Test with large parameters (should not cause overflow)
            α_large = fill(10.0, K * (J-1))
            @test_nowarn mlogit_mle(α_large, X_test, y_test)
            
            # Results should be finite
            ll_zero = mlogit_mle(α_zero, X_test, y_test)
            ll_large = mlogit_mle(α_large, X_test, y_test)
            @test isfinite(ll_zero)
            @test isfinite(ll_large)
        end
        
        @testset "Data quality tests" begin
            # Test with perfect separation (all choices are the same)
            N = 20
            K = 2
            J = 3
            X_test = [ones(N) randn(N)]
            y_uniform = fill(1, N)  # All choose alternative 1
            α_test = randn(K * (J-1))
            
            # Should handle perfect separation gracefully
            @test_nowarn ll = mlogit_mle(α_test, X_test, y_uniform)
            ll = mlogit_mle(α_test, X_test, y_uniform)
            @test isfinite(ll)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 8: PERFORMANCE AND NUMERICAL STABILITY
    #--------------------------------------------------------------------------
    
    @testset "Performance and Numerical Stability" begin
        
        @testset "Large sample performance" begin
            # Test that functions can handle reasonably large samples
            N = 5000
            K = 4
            J = 5
            Random.seed!(999)
            
            # Generate data
            Y_large, X_large = sim_logit(N, J)
            α_test = randn(K * (J-1))
            
            # Functions should complete in reasonable time and not crash
            @test_nowarn mlogit_mle(α_test, X_large, Y_large)
            @test_nowarn mlogit_gmm_overid(α_test, X_large, Y_large)
            
            # Test with timing constraint (should complete within reasonable time)
            start_time = time()
            ll = mlogit_mle(α_test, X_large, Y_large)
            elapsed = time() - start_time
            @test elapsed < 10.0  # Should complete in less than 10 seconds
            @test isfinite(ll)
        end
        
        @testset "Numerical precision tests" begin
            # Test with data that might cause numerical issues
            N = 100
            K = 3
            J = 4
            
            # Generate data with extreme values
            X_extreme = [ones(N) 100 .* randn(N) 1000 .* randn(N)]
            y_extreme = rand(1:J, N)
            α_extreme = 0.001 .* randn(K * (J-1))
            
            # Should handle extreme values without NaN/Inf
            ll = mlogit_mle(α_extreme, X_extreme, y_extreme)
            @test isfinite(ll)
            
            # Test with very small parameters
            α_tiny = 1e-10 .* randn(K * (J-1))
            ll_tiny = mlogit_mle(α_tiny, X_extreme, y_extreme)
            @test isfinite(ll_tiny)
        end
    end
    
    println("\n" * "="^60)
    println("ALL PS7 GMM AND SMM ESTIMATION TESTS COMPLETED")
    println("="^60)
    
end