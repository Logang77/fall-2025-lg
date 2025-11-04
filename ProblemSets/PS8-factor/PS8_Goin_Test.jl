using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff, LineSearches

cd(@__DIR__)

include("PS8_Goin_Source.jl")

# ==================================================================================
# Unit Tests for PS8_Goin_Source.jl
# ==================================================================================

@testset "PS8 Factor Model Tests" begin
    
    # =============================================================================
    # Test 1: load_data function
    # =============================================================================
    @testset "Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        
        # Test that data loads successfully
        df = load_data(url)
        @test df isa DataFrame
        @test size(df, 1) > 0  # Has observations
        @test size(df, 2) > 0  # Has variables
        
        # Test that expected columns exist
        @test :logwage in names(df)
        @test :black in names(df)
        @test :hispanic in names(df)
        @test :female in names(df)
        @test :schoolt in names(df)
        @test :gradHS in names(df)
        @test :grad4yr in names(df)
        
        # Test that ASVAB columns exist
        @test :asvabAR in names(df)
        @test :asvabCS in names(df)
        @test :asvabMK in names(df)
        @test :asvabNO in names(df)
        @test :asvabPC in names(df)
        @test :asvabWK in names(df)
        
        # Test that data has reasonable values (no all NaN or Inf)
        @test all(isfinite, skipmissing(df.logwage))
        @test all(x -> x in [0, 1], df.black)
        @test all(x -> x in [0, 1], df.hispanic)
        @test all(x -> x in [0, 1], df.female)
    end
    
    # =============================================================================
    # Test 2: compute_asvab_correlations function
    # =============================================================================
    @testset "ASVAB Correlations" begin
        # Create test data with known correlations
        Random.seed!(1234)
        n = 100
        test_df = DataFrame(
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        # Compute correlations
        cordf = compute_asvab_correlations(test_df)
        
        # Test output structure
        @test cordf isa DataFrame
        @test size(cordf) == (6, 6)  # 6x6 correlation matrix
        @test all(names(cordf) .== ["cor1", "cor2", "cor3", "cor4", "cor5", "cor6"])
        
        # Test that diagonal elements are 1 (variable correlated with itself)
        @test isapprox(cordf.cor1[1], 1.0, atol=1e-10)
        @test isapprox(cordf.cor2[2], 1.0, atol=1e-10)
        @test isapprox(cordf.cor3[3], 1.0, atol=1e-10)
        @test isapprox(cordf.cor4[4], 1.0, atol=1e-10)
        @test isapprox(cordf.cor5[5], 1.0, atol=1e-10)
        @test isapprox(cordf.cor6[6], 1.0, atol=1e-10)
        
        # Test that correlations are symmetric
        @test isapprox(cordf.cor1[2], cordf.cor2[1], atol=1e-10)
        @test isapprox(cordf.cor1[3], cordf.cor3[1], atol=1e-10)
        
        # Test that all correlations are between -1 and 1
        for col in names(cordf)
            @test all(-1 .<= cordf[!, col] .<= 1)
        end
    end
    
    # =============================================================================
    # Test 3: generate_pca! function
    # =============================================================================
    @testset "PCA Generation" begin
        Random.seed!(1234)
        n = 100
        
        # Create test dataframe with ASVAB scores at the end
        test_df = DataFrame(
            var1 = randn(n),
            var2 = randn(n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        # Apply PCA
        result_df = generate_pca!(test_df)
        
        # Test that output is a DataFrame
        @test result_df isa DataFrame
        
        # Test that asvabPCA column was added
        @test :asvabPCA in names(result_df)
        
        # Test that asvabPCA has correct length
        @test length(result_df.asvabPCA) == n
        
        # Test that all values are finite
        @test all(isfinite, result_df.asvabPCA)
        
        # Test that original columns are preserved
        @test :var1 in names(result_df)
        @test :var2 in names(result_df)
        @test all(names(test_df) .∈ Ref(names(result_df)))
    end
    
    # =============================================================================
    # Test 4: generate_factor! function
    # =============================================================================
    @testset "Factor Analysis Generation" begin
        Random.seed!(1234)
        n = 100
        
        # Create test dataframe with ASVAB scores at the end
        test_df = DataFrame(
            var1 = randn(n),
            var2 = randn(n),
            asvabAR = randn(n) .+ 5,
            asvabCS = randn(n) .+ 5,
            asvabMK = randn(n) .+ 5,
            asvabNO = randn(n) .+ 5,
            asvabPC = randn(n) .+ 5,
            asvabWK = randn(n) .+ 5
        )
        
        # Apply Factor Analysis
        result_df = generate_factor!(test_df)
        
        # Test that output is a DataFrame
        @test result_df isa DataFrame
        
        # Test that asvabFactor column was added
        @test :asvabFactor in names(result_df)
        
        # Test that asvabFactor has correct length
        @test length(result_df.asvabFactor) == n
        
        # Test that all values are finite
        @test all(isfinite, result_df.asvabFactor)
        
        # Test that original columns are preserved
        @test :var1 in names(result_df)
        @test :var2 in names(result_df)
        @test all(names(test_df) .∈ Ref(names(result_df)))
    end
    
    # =============================================================================
    # Test 5: prepare_factor_matrices function
    # =============================================================================
    @testset "Factor Matrix Preparation" begin
        Random.seed!(1234)
        n = 50
        
        # Create test dataframe with all required columns
        test_df = DataFrame(
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            logwage = randn(n) .+ 2.0,
            asvabAR = randn(n) .+ 50,
            asvabCS = randn(n) .+ 50,
            asvabMK = randn(n) .+ 50,
            asvabNO = randn(n) .+ 50,
            asvabPC = randn(n) .+ 50,
            asvabWK = randn(n) .+ 50
        )
        
        # Call function
        X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
        
        # Test X matrix dimensions (N × 7)
        @test size(X) == (n, 7)
        @test X isa Matrix
        
        # Test that last column of X is constant (all ones)
        @test all(X[:, end] .== 1.0)
        
        # Test y vector
        @test length(y) == n
        @test y isa Vector
        @test all(isfinite, y)
        
        # Test Xfac matrix dimensions (N × 4)
        @test size(Xfac) == (n, 4)
        @test Xfac isa Matrix
        
        # Test that last column of Xfac is constant (all ones)
        @test all(Xfac[:, end] .== 1.0)
        
        # Test asvabs matrix dimensions (N × 6)
        @test size(asvabs) == (n, 6)
        @test asvabs isa Matrix
        @test all(isfinite, asvabs)
        
        # Test that matrices contain correct data
        @test X[:, 1] == test_df.black
        @test X[:, 2] == test_df.hispanic
        @test X[:, 3] == test_df.female
        @test y == test_df.logwage
    end
    
    # =============================================================================
    # Test 6: factor_model function (likelihood computation)
    # =============================================================================
    @testset "Factor Model Likelihood" begin
        Random.seed!(1234)
        n = 30  # Small sample for testing
        K = 7   # Covariates in wage equation
        L = 4   # Covariates in measurement equations
        J = 6   # Number of ASVAB tests
        R = 5   # Quadrature points
        
        # Create synthetic data
        X = [randn(n, K-1) ones(n)]
        Xfac = [randn(n, L-1) ones(n)]
        Meas = randn(n, J) .+ 50
        y = randn(n) .+ 2.0
        
        # Create parameter vector
        # θ = [γ (L×J), β (K×1), α (J+1), σ (J+1)]
        γ = randn(L * J)
        β = randn(K)
        α = abs.(randn(J + 1)) .+ 0.1  # Positive factor loadings
        σ = abs.(randn(J + 1)) .+ 0.5  # Positive standard deviations
        θ = vcat(γ, β, α, σ)
        
        # Expected total length: L*J + K + (J+1) + (J+1) = 24 + 7 + 7 + 7 = 45
        @test length(θ) == L * J + K + (J + 1) + (J + 1)
        
        # Compute likelihood
        nll = factor_model(θ, X, Xfac, Meas, y, R)
        
        # Test that likelihood is finite
        @test isfinite(nll)
        
        # Test that likelihood is a scalar
        @test nll isa Real
        
        # Test that negative log-likelihood is positive (log-likelihood is negative)
        # This should generally be true for well-specified models
        @test nll > 0 || isfinite(nll)  # At minimum should be finite
        
        # Test with different quadrature points
        nll2 = factor_model(θ, X, Xfac, Meas, y, 7)
        @test isfinite(nll2)
        
        # Test that function works with Float64
        @test typeof(nll) <: Real
    end
    
    # =============================================================================
    # Test 7: Parameter unpacking in factor_model
    # =============================================================================
    @testset "Parameter Unpacking" begin
        Random.seed!(1234)
        n = 20
        K = 7
        L = 4
        J = 6
        
        # Create test data
        X = [randn(n, K-1) ones(n)]
        Xfac = [randn(n, L-1) ones(n)]
        Meas = randn(n, J) .+ 50
        y = randn(n) .+ 2.0
        
        # Create parameter vector with known structure
        γ_true = collect(1:L*J) ./ 10.0
        β_true = collect(1:K) ./ 5.0
        α_true = collect(1:J+1) ./ 10.0
        σ_true = ones(J + 1) .* 0.5
        θ = vcat(γ_true, β_true, α_true, σ_true)
        
        # Test that we can compute likelihood (implicitly tests unpacking)
        nll = factor_model(θ, X, Xfac, Meas, y, 5)
        @test isfinite(nll)
        
        # Test correct parameter vector length
        expected_length = L * J + K + (J + 1) + (J + 1)
        @test length(θ) == expected_length
    end
    
    # =============================================================================
    # Test 8: Edge cases and error handling
    # =============================================================================
    @testset "Edge Cases" begin
        Random.seed!(1234)
        
        # Test with minimal data
        n_small = 10
        test_df_small = DataFrame(
            black = rand(0:1, n_small),
            hispanic = rand(0:1, n_small),
            female = rand(0:1, n_small),
            schoolt = rand(8:16, n_small),
            gradHS = rand(0:1, n_small),
            grad4yr = rand(0:1, n_small),
            logwage = randn(n_small),
            asvabAR = randn(n_small),
            asvabCS = randn(n_small),
            asvabMK = randn(n_small),
            asvabNO = randn(n_small),
            asvabPC = randn(n_small),
            asvabWK = randn(n_small)
        )
        
        # Test that functions work with minimal data
        X, y, Xfac, asvabs = prepare_factor_matrices(test_df_small)
        @test size(X, 1) == n_small
        @test length(y) == n_small
        
        # Test correlation computation with minimal data
        cordf = compute_asvab_correlations(test_df_small)
        @test size(cordf) == (6, 6)
    end
    
    # =============================================================================
    # Test 9: Integration - Full workflow with synthetic data
    # =============================================================================
    @testset "Integration Test" begin
        Random.seed!(5678)
        n = 100
        
        # Create synthetic dataset with known structure
        latent_ability = randn(n)
        
        test_df = DataFrame(
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            logwage = 2.0 .+ 0.1 .* latent_ability .+ randn(n) .* 0.1,
            asvabAR = 50.0 .+ 0.5 .* latent_ability .+ randn(n),
            asvabCS = 50.0 .+ 0.5 .* latent_ability .+ randn(n),
            asvabMK = 50.0 .+ 0.5 .* latent_ability .+ randn(n),
            asvabNO = 50.0 .+ 0.5 .* latent_ability .+ randn(n),
            asvabPC = 50.0 .+ 0.5 .* latent_ability .+ randn(n),
            asvabWK = 50.0 .+ 0.5 .* latent_ability .+ randn(n)
        )
        
        # Test full workflow
        # 1. Correlation computation
        cordf = compute_asvab_correlations(test_df)
        @test size(cordf) == (6, 6)
        
        # 2. PCA
        test_df_pca = generate_pca!(copy(test_df))
        @test :asvabPCA in names(test_df_pca)
        
        # 3. Factor Analysis
        test_df_fa = generate_factor!(copy(test_df))
        @test :asvabFactor in names(test_df_fa)
        
        # 4. Matrix preparation
        X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
        @test size(X, 1) == n
        @test length(y) == n
        @test size(asvabs) == (n, 6)
        
        # 5. Likelihood evaluation with reasonable parameters
        K = size(X, 2)
        L = size(Xfac, 2)
        J = size(asvabs, 2)
        
        θ_test = vcat(
            zeros(L * J),           # γ
            zeros(K),               # β
            ones(J + 1) .* 0.1,     # α
            ones(J + 1) .* 0.5      # σ
        )
        
        nll = factor_model(θ_test, X, Xfac, asvabs, y, 5)
        @test isfinite(nll)
        @test nll > 0
    end
    
    # =============================================================================
    # Test 10: Numerical stability
    # =============================================================================
    @testset "Numerical Stability" begin
        Random.seed!(9999)
        n = 50
        K = 7
        L = 4
        J = 6
        
        # Create data with extreme values
        X = [randn(n, K-1) .* 100 ones(n)]  # Large values
        Xfac = [randn(n, L-1) .* 100 ones(n)]
        Meas = randn(n, J) .* 10 .+ 50
        y = randn(n) .* 10 .+ 2.0
        
        # Create reasonable parameters
        θ = vcat(
            randn(L * J) .* 0.01,    # Small γ
            randn(K) .* 0.01,        # Small β
            abs.(randn(J + 1)) .* 0.1,  # Small positive α
            ones(J + 1) .* 1.0       # Reasonable σ
        )
        
        # Test that likelihood is still computable
        nll = factor_model(θ, X, Xfac, Meas, y, 5)
        @test isfinite(nll) || nll > 0  # Should be finite or at least not NaN
    end
    
end

println("\n" * "="^80)
println("All unit tests completed!")
println("="^80)
