using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

# Set up test environment
cd(@__DIR__)
include("PS5_Goin_Source.jl")

@testset "PS5 Bus Engine Replacement Model Tests" begin

    #--------------------------------------------------------------------------
    # TEST SET 1: DATA LOADING AND PREPARATION
    #--------------------------------------------------------------------------
    
    @testset "Data Loading Functions" begin
        
        @testset "load_static_data() tests" begin
            # Test basic functionality
            @test_nowarn df_long = load_static_data()
            
            df_long = load_static_data()
            
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
            
            # Test that bus_id is sequential
            unique_buses = unique(df_long.bus_id)
            @test minimum(unique_buses) == 1
            @test maximum(unique_buses) == length(unique_buses)
            
            # Test that each bus has exactly 20 observations
            bus_counts = combine(groupby(df_long, :bus_id), nrow => :count)
            @test all(count -> count == 20, bus_counts.count)
        end
        
        @testset "load_dynamic_data() tests" begin
            # Test basic functionality
            @test_nowarn d = load_dynamic_data()
            
            d = load_dynamic_data()
            
            # Test that it returns a named tuple
            @test isa(d, NamedTuple)
            
            # Test required fields exist
            required_fields = [:Y, :X, :B, :Xstate, :Zstate, :N, :T, :xval, :xbin, :zbin, :xtran, :β]
            for field in required_fields
                @test haskey(d, field)
            end
            
            # Test dimensions consistency
            @test size(d.Y) == (d.N, d.T)
            @test size(d.X) == (d.N, d.T)
            @test size(d.Xstate) == (d.N, d.T)
            @test length(d.B) == d.N
            @test length(d.Zstate) == d.N
            
            # Test state space dimensions
            @test d.xbin == length(d.xval)
            @test size(d.xtran) == (d.zbin * d.xbin, d.xbin)
            
            # Test discount factor
            @test d.β == 0.9
            
            # Test that Y is binary
            @test all(y -> y in [0, 1], d.Y)
            
            # Test that B is binary
            @test all(b -> b in [0, 1], d.B)
            
            # Test that state indices are valid
            @test all(x -> 1 <= x <= d.xbin, d.Xstate)
            @test all(z -> 1 <= z <= d.zbin, d.Zstate)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 2: STATIC ESTIMATION
    #--------------------------------------------------------------------------
    
    @testset "Static Estimation Tests" begin
        
        @testset "estimate_static_model() tests" begin
            # Create small test dataset
            test_df = DataFrame(
                Y = [0, 1, 0, 1, 0],
                Odometer = [100.0, 200.0, 150.0, 300.0, 250.0],
                Branded = [0, 1, 0, 1, 0]
            )
            
            # Test that function runs without error
            @test_nowarn estimate_static_model(test_df)
            
            # Test with actual data
            df_long = load_static_data()
            @test_nowarn estimate_static_model(df_long)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 3: DYNAMIC ESTIMATION - HELPER FUNCTIONS
    #--------------------------------------------------------------------------
    
    @testset "Dynamic Estimation Helper Tests" begin
        
        # Create minimal test data structure
        function create_test_data(N=2, T=3, xbin=5, zbin=3)
            # Create simple transition matrix
            xtran = zeros(zbin * xbin, xbin)
            for i in 1:size(xtran, 1)
                xtran[i, min(i, xbin)] = 1.0  # Simple diagonal-ish pattern
            end
            
            return (
                Y = rand([0, 1], N, T),
                X = rand(0:100, N, T),
                B = rand([0, 1], N),
                Xstate = rand(1:xbin, N, T),
                Zstate = rand(1:zbin, N),
                N = N,
                T = T,
                xval = collect(0:25:100),  # xbin=5
                xbin = xbin,
                zbin = zbin,
                xtran = xtran,
                β = 0.9
            )
        end
        
        @testset "compute_future_value!() tests" begin
            d = create_test_data(2, 3, 5, 3)
            θ = [1.0, -0.1, 0.5]
            FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
            
            # Test basic functionality
            @test_nowarn compute_future_value!(FV, θ, d)
            
            # Test dimensions
            result_FV = compute_future_value!(FV, θ, d)
            @test size(result_FV) == (d.zbin * d.xbin, 2, d.T + 1)
            
            # Test terminal condition (FV[T+1] should remain zero)
            @test all(result_FV[:, :, d.T + 1] .== 0.0)
            
            # Test that future values are finite
            @test all(isfinite.(result_FV))
            
            # Test that future values are generally decreasing over time
            # (earlier periods should have higher values due to discounting)
            for i in 1:size(result_FV, 1), j in 1:size(result_FV, 2)
                for t in 1:(d.T-1)
                    @test result_FV[i, j, t] >= result_FV[i, j, t+1] - 1e-10  # Allow small numerical error
                end
            end
        end
        
        @testset "log_likelihood_dynamic() tests" begin
            d = create_test_data(2, 3, 5, 3)
            θ = [1.0, -0.1, 0.5]
            
            # Test basic functionality
            @test_nowarn ll = log_likelihood_dynamic(θ, d)
            
            # Test that it returns a finite number
            ll = log_likelihood_dynamic(θ, d)
            @test isfinite(ll)
            
            # Test that likelihood is negative (since we return -loglike)
            @test ll < 0
            
            # Test parameter sensitivity
            θ2 = [1.5, -0.1, 0.5]
            ll2 = log_likelihood_dynamic(θ2, d)
            @test ll != ll2  # Different parameters should give different likelihoods
            
            # Test extreme parameter robustness
            θ_extreme = [100.0, -10.0, 10.0]
            @test_nowarn log_likelihood_dynamic(θ_extreme, d)
            
            # Test that result is finite even with extreme parameters
            ll_extreme = log_likelihood_dynamic(θ_extreme, d)
            @test isfinite(ll_extreme)
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 4: ESTIMATION WRAPPER FUNCTIONS
    #--------------------------------------------------------------------------
    
    @testset "Estimation Wrapper Tests" begin
        
        @testset "estimate_dynamic_model() tests" begin
            d = create_test_data(2, 3, 5, 3)
            
            # Test with provided starting values
            θ_start = [1.0, -0.1, 0.5]
            @test_nowarn estimate_dynamic_model(d, θ_start=θ_start)
            
            # Test with random starting values (nothing provided)
            @test_nowarn estimate_dynamic_model(d)
        end
        
        @testset "main() function tests" begin
            # This is an integration test - just ensure it doesn't crash
            # Note: This will actually try to download data, so might be slow
            @test_nowarn main()
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 5: MATHEMATICAL PROPERTIES
    #--------------------------------------------------------------------------
    
    @testset "Mathematical Properties Tests" begin
        
        @testset "Future Value Mathematical Properties" begin
            d = create_test_data(2, 3, 5, 3)
            θ = [1.0, -0.1, 0.5]
            FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
            compute_future_value!(FV, θ, d)
            
            # Test monotonicity with respect to discount factor
            d_high_beta = (d..., β=0.95)
            FV_high_beta = zeros(d.zbin * d.xbin, 2, d.T + 1)
            compute_future_value!(FV_high_beta, θ, d_high_beta)
            
            # Higher discount factor should generally lead to higher future values
            # (This is a weak test due to the complexity of the model)
            mean_fv_low = mean(FV[FV .> 0])
            mean_fv_high = mean(FV_high_beta[FV_high_beta .> 0])
            @test mean_fv_high >= mean_fv_low - 1e-10
        end
        
        @testset "Likelihood Properties" begin
            d = create_test_data(5, 4, 6, 4)  # Slightly larger for better tests
            
            # Test that likelihood is well-defined over parameter space
            test_params = [
                [0.0, 0.0, 0.0],
                [1.0, -0.1, 0.5],
                [-1.0, 0.05, -0.3],
                [2.0, -0.2, 1.0]
            ]
            
            for θ in test_params
                ll = log_likelihood_dynamic(θ, d)
                @test isfinite(ll)
                @test ll < 0  # Log likelihood should be negative
            end
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 6: EDGE CASES AND ROBUSTNESS
    #--------------------------------------------------------------------------
    
    @testset "Edge Cases and Robustness Tests" begin
        
        @testset "Small Data Tests" begin
            # Test with minimal data size
            d_small = create_test_data(1, 1, 3, 2)
            θ = [0.5, -0.05, 0.2]
            
            # Should handle small datasets gracefully
            @test_nowarn log_likelihood_dynamic(θ, d_small)
            
            FV_small = zeros(d_small.zbin * d_small.xbin, 2, d_small.T + 1)
            @test_nowarn compute_future_value!(FV_small, θ, d_small)
        end
        
        @testset "Parameter Boundary Tests" begin
            d = create_test_data(3, 2, 4, 3)
            
            # Test near-zero parameters
            θ_zero = [1e-10, -1e-10, 1e-10]
            @test_nowarn log_likelihood_dynamic(θ_zero, d)
            
            # Test large parameters (should not crash, but might be unrealistic)
            θ_large = [10.0, -1.0, 5.0]
            ll_large = log_likelihood_dynamic(θ_large, d)
            @test isfinite(ll_large)
        end
        
        @testset "Data Consistency Tests" begin
            # Test that functions handle the actual data structure correctly
            d = load_dynamic_data()
            θ = [1.0, -0.1, 0.5]
            
            # Verify data consistency
            @test d.N > 0
            @test d.T > 0
            @test d.xbin > 0
            @test d.zbin > 0
            
            # Test one likelihood evaluation with real data
            @test_nowarn log_likelihood_dynamic(θ, d)
            
            # The likelihood should be finite
            ll_real = log_likelihood_dynamic(θ, d)
            @test isfinite(ll_real)
            @test ll_real < 0
        end
    end
    
    #--------------------------------------------------------------------------
    # TEST SET 7: PERFORMANCE AND MEMORY
    #--------------------------------------------------------------------------
    
    @testset "Performance and Memory Tests" begin
        
        @testset "Memory Allocation Tests" begin
            d = create_test_data(3, 2, 4, 3)
            θ = [1.0, -0.1, 0.5]
            
            # Test that compute_future_value! modifies array in place
            FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
            FV_original = copy(FV)
            result = compute_future_value!(FV, θ, d)
            
            # Should return the same array (modified in place)
            @test result === FV
            @test FV != FV_original  # Should have been modified
        end
        
        @testset "Numerical Stability Tests" begin
            d = create_test_data(2, 2, 3, 2)
            
            # Test with parameters that might cause numerical issues
            θ_unstable = [50.0, -5.0, 10.0]  # Large parameters
            
            # Should not throw numerical errors
            @test_nowarn log_likelihood_dynamic(θ_unstable, d)
            
            ll_unstable = log_likelihood_dynamic(θ_unstable, d)
            @test isfinite(ll_unstable)
        end
    end
    
    println("\n" * "="^60)
    println("ALL TESTS COMPLETED")
    println("="^60)
    
end