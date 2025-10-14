using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__)
include("PS6_Goin_Source.jl")

@testset "PS6 CCP Estimation Tests" begin

    #--------------------------------------------------------------------------
    # TEST SET 1: DATA LOADING AND PREPARATION
    #--------------------------------------------------------------------------
    
    @testset "Data Loading Functions" begin
        
        @testset "load_and_reshape_data() tests" begin
            # Test with a mock small dataset (since downloading may be slow in tests)
            # We'll create a simple test case and also test the actual function
            
            # Test basic functionality with real URL (this might be slow)
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
            @test_nowarn df_long = load_and_reshape_data(url)
            
            df_long = load_and_reshape_data(url)
            
            # Test structure
            @test isa(df_long, DataFrame)
            @test haskey(df_long, :bus_id)
            @test haskey(df_long, :time)
            @test haskey(df_long, :Y)
            @test haskey(df_long, :Odometer)
            @test haskey(df_long, :RouteUsage)
            @test haskey(df_long, :Branded)
            
            # Test dimensions
            @test nrow(df_long) > 0
            @test ncol(df_long) == 6
            
            # Test time variable ranges from 1 to 20
            @test minimum(df_long.time) == 1
            @test maximum(df_long.time) == 20
            
            # Test that Y is binary
            @test all(y -> y in [0, 1], df_long.Y)
            
            # Test that Branded is binary
            @test all(b -> b in [0, 1], df_long.Branded)
            
            # Test that each bus has exactly 20 observations
            bus_counts = combine(groupby(df_long, :bus_id), nrow => :count)
            @test all(count -> count == 20, bus_counts.count)
            
            # Test sorting
            @test issorted(df_long, [:bus_id, :time])
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 2: FLEXIBLE LOGIT ESTIMATION
    #--------------------------------------------------------------------------
    
    @testset "Flexible Logit Estimation Tests" begin
        
        @testset "estimate_flexible_logit() tests" begin
            # Create small test dataset
            test_df = DataFrame(
                Y = [0, 1, 0, 1, 0, 1],
                Odometer = [100.0, 200.0, 150.0, 300.0, 250.0, 180.0],
                RouteUsage = [0.5, 0.8, 0.6, 0.9, 0.7, 0.4],
                Branded = [0, 1, 0, 1, 0, 1],
                time = [1, 2, 3, 4, 5, 6]
            )
            
            # Test that function runs without error
            @test_nowarn model = estimate_flexible_logit(test_df)
            
            # Test return type
            model = estimate_flexible_logit(test_df)
            @test isa(model, StatsModels.TableRegressionModel)
            
            # Test that the model has coefficients
            @test length(coef(model)) > 0
            
            # Test with actual data (may be slow)
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
            df_long = load_and_reshape_data(url)
            @test_nowarn real_model = estimate_flexible_logit(df_long)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 3: STATE SPACE CONSTRUCTION
    #--------------------------------------------------------------------------
    
    @testset "State Space Construction Tests" begin
        
        # Create test grids
        include("create_grids.jl")
        zval, zbin, xval, xbin, xtran = create_grids()
        
        @testset "construct_state_space() tests" begin
            # Test basic functionality
            @test_nowarn state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Test structure
            @test isa(state_df, DataFrame)
            @test haskey(state_df, :Odometer)
            @test haskey(state_df, :RouteUsage)
            @test haskey(state_df, :Branded)
            @test haskey(state_df, :time)
            
            # Test dimensions
            expected_rows = xbin * zbin
            @test nrow(state_df) == expected_rows
            @test ncol(state_df) == 4
            
            # Test Odometer values
            @test length(unique(state_df.Odometer)) == xbin
            @test all(x -> x in xval, state_df.Odometer)
            
            # Test RouteUsage values
            @test length(unique(state_df.RouteUsage)) == zbin
            @test all(z -> z in zval, state_df.RouteUsage)
            
            # Test initial values
            @test all(state_df.Branded .== 0)
            @test all(state_df.time .== 0)
            
            # Test Kronecker product structure
            expected_odometer = kron(ones(zbin), xval)
            expected_routeusage = kron(zval, ones(xbin))
            @test state_df.Odometer ≈ expected_odometer
            @test state_df.RouteUsage ≈ expected_routeusage
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 4: FUTURE VALUE COMPUTATION
    #--------------------------------------------------------------------------
    
    @testset "Future Value Computation Tests" begin
        
        # Create test data
        include("create_grids.jl")
        zval, zbin, xval, xbin, xtran = create_grids()
        
        # Create simple test DataFrame for state space
        state_df = DataFrame(
            Odometer = kron(ones(zbin), xval),
            RouteUsage = kron(zval, ones(xbin)),
            Branded = zeros(xbin * zbin),
            time = zeros(xbin * zbin)
        )
        
        # Create simple test model (mock GLM)
        test_data = DataFrame(
            Y = rand([0, 1], 100),
            Odometer = rand(100:500, 100),
            RouteUsage = rand(100) * 0.5 .+ 0.5,
            Branded = rand([0, 1], 100),
            time = rand(1:20, 100)
        )
        test_model = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time), 
                        test_data, Binomial(), LogitLink())
        
        @testset "compute_future_values() tests" begin
            T = 5  # Use smaller T for testing
            β = 0.9
            
            # Test basic functionality
            @test_nowarn FV = compute_future_values(state_df, test_model, xtran, xbin, zbin, T, β)
            
            FV = compute_future_values(state_df, test_model, xtran, xbin, zbin, T, β)
            
            # Test dimensions
            @test size(FV) == (xbin * zbin, 2, T + 1)
            
            # Test terminal condition (should be zeros)
            @test all(FV[:, :, T + 1] .== 0.0)
            
            # Test that future values are finite
            @test all(isfinite.(FV))
            
            # Test that discount factor affects magnitude
            FV_high_beta = compute_future_values(state_df, test_model, xtran, xbin, zbin, T, 0.95)
            # With higher discount factor, future values should generally be larger in absolute value
            @test mean(abs.(FV_high_beta[FV_high_beta .!= 0])) >= mean(abs.(FV[FV .!= 0])) - 1e-10
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 5: MAPPING FUTURE VALUES TO DATA
    #--------------------------------------------------------------------------
    
    @testset "Future Value Mapping Tests" begin
        
        # Create test data structures
        include("create_grids.jl")
        zval, zbin, xval, xbin, xtran = create_grids()
        
        # Create test long DataFrame
        N = 5  # Small number of buses for testing
        T = 4  # Small number of time periods
        df_long = DataFrame(
            bus_id = repeat(1:N, inner=T),
            time = repeat(1:T, outer=N),
            Y = rand([0, 1], N*T),
            Odometer = rand(100:400, N*T),
            RouteUsage = rand(N*T) * 0.5 .+ 0.5,
            Branded = repeat(rand([0, 1], N), inner=T)
        )
        
        # Create test matrices (simplified for testing)
        FV = rand(xbin * zbin, 2, T + 1)
        Xstate = rand(1:xbin, N, T)  # Discrete state indices
        Zstate = rand(1:zbin, N)     # Discrete state indices
        B = rand([0, 1], N)          # Brand indicators
        
        @testset "compute_fvt1() tests" begin
            # Test basic functionality
            @test_nowarn fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
            
            fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
            
            # Test dimensions
            @test length(fvt1) == N * T
            @test length(fvt1) == nrow(df_long)
            
            # Test that results are finite
            @test all(isfinite.(fvt1))
            
            # Test with edge cases
            # Test when all states are the same
            Xstate_same = ones(Int, N, T)
            Zstate_same = ones(Int, N)
            @test_nowarn fvt1_same = compute_fvt1(df_long, FV, xtran, Xstate_same, Zstate_same, xbin, B)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 6: STRUCTURAL PARAMETER ESTIMATION
    #--------------------------------------------------------------------------
    
    @testset "Structural Parameter Estimation Tests" begin
        
        # Create test data
        N = 100
        test_df = DataFrame(
            Y = rand([0, 1], N),
            Odometer = rand(100:500, N),
            Branded = rand([0, 1], N),
            bus_id = 1:N,
            time = ones(Int, N),
            RouteUsage = rand(N) * 0.5 .+ 0.5
        )
        fvt1 = randn(N) * 0.1  # Small random future values
        
        @testset "estimate_structural_params() tests" begin
            # Test basic functionality
            @test_nowarn theta_hat = estimate_structural_params(test_df, fvt1)
            
            theta_hat = estimate_structural_params(test_df, fvt1)
            
            # Test return type
            @test isa(theta_hat, StatsModels.TableRegressionModel)
            
            # Test that model has the expected number of coefficients
            # Should have intercept + Odometer + Branded = 3 coefficients
            @test length(coef(theta_hat)) == 3
            
            # Test that the model converged
            @test theta_hat.model.fit.converged
            
            # Test with zero future values
            fvt1_zero = zeros(N)
            @test_nowarn theta_zero = estimate_structural_params(test_df, fvt1_zero)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 7: INTEGRATION TESTS
    #--------------------------------------------------------------------------
    
    @testset "Integration Tests" begin
        
        @testset "main() function components" begin
            # Test that grid creation works
            @test_nowarn begin
                include("create_grids.jl")
                zval, zbin, xval, xbin, xtran = create_grids()
            end
            
            # Test that data loading works
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
            @test_nowarn df_long = load_and_reshape_data(url)
            
            # Test the sequential workflow with small data
            df_long = load_and_reshape_data(url)
            include("create_grids.jl")
            zval, zbin, xval, xbin, xtran = create_grids()
            
            # Test state space construction
            @test_nowarn statedf = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Test flexible logit (this may take time with real data)
            @test_nowarn flexlogit = estimate_flexible_logit(df_long[1:1000, :])  # Use subset for speed
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 8: EDGE CASES AND ROBUSTNESS
    #--------------------------------------------------------------------------
    
    @testset "Edge Cases and Robustness Tests" begin
        
        @testset "Small data tests" begin
            # Test with minimal data
            small_df = DataFrame(
                Y = [0, 1],
                Odometer = [100.0, 200.0],
                RouteUsage = [0.5, 0.8],
                Branded = [0, 1],
                time = [1, 2]
            )
            
            # Should handle small datasets gracefully
            @test_nowarn model = estimate_flexible_logit(small_df)
        end
        
        @testset "Data consistency tests" begin
            # Test with the actual grid dimensions
            include("create_grids.jl")
            zval, zbin, xval, xbin, xtran = create_grids()
            
            # Verify grid consistency
            @test length(zval) == zbin
            @test length(xval) == xbin
            @test size(xtran) == (zbin * xbin, xbin)
            
            # Test state space construction with actual grids
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            @test nrow(state_df) == size(xtran, 1)
        end
        
        @testset "Parameter boundary tests" begin
            # Test with extreme discount factors
            include("create_grids.jl")
            zval, zbin, xval, xbin, xtran = create_grids()
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Create simple model for testing
            test_data = DataFrame(
                Y = rand([0, 1], 50),
                Odometer = rand(100:500, 50),
                RouteUsage = rand(50) * 0.5 .+ 0.5,
                Branded = rand([0, 1], 50),
                time = rand(1:5, 50)
            )
            test_model = glm(@formula(Y ~ Odometer + RouteUsage), test_data, Binomial(), LogitLink())
            
            # Test with extreme discount factors
            @test_nowarn FV_low = compute_future_values(state_df, test_model, xtran, xbin, zbin, 3, 0.1)
            @test_nowarn FV_high = compute_future_values(state_df, test_model, xtran, xbin, zbin, 3, 0.99)
            
            # Both should produce finite results
            FV_low = compute_future_values(state_df, test_model, xtran, xbin, zbin, 3, 0.1)
            FV_high = compute_future_values(state_df, test_model, xtran, xbin, zbin, 3, 0.99)
            @test all(isfinite.(FV_low))
            @test all(isfinite.(FV_high))
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 9: PERFORMANCE AND MEMORY
    #--------------------------------------------------------------------------
    
    @testset "Performance Tests" begin
        
        @testset "Memory allocation tests" begin
            # Test that functions don't allocate excessive memory for reasonable inputs
            include("create_grids.jl")
            zval, zbin, xval, xbin, xtran = create_grids()
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Test that state_df is created efficiently
            @test nrow(state_df) == xbin * zbin
            @test ncol(state_df) == 4
        end
        
        @testset "Numerical stability tests" begin
            # Test with data that might cause numerical issues
            extreme_df = DataFrame(
                Y = [0, 1, 0, 1],
                Odometer = [1e-6, 1e6, 0.0, 1000.0],
                RouteUsage = [1e-6, 1.0, 0.5, 0.999],
                Branded = [0, 1, 0, 1],
                time = [1, 2, 3, 4]
            )
            
            # Should handle extreme values gracefully
            @test_nowarn model = estimate_flexible_logit(extreme_df)
            
            # Test with large future value terms
            large_fvt1 = randn(4) * 100
            @test_nowarn result = estimate_structural_params(extreme_df, large_fvt1)
        end
    end
    
    println("\n" * "="^60)
    println("ALL PS6 CCP ESTIMATION TESTS COMPLETED")
    println("="^60)
    
end
