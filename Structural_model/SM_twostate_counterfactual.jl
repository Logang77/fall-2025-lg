#!/usr/bin/env julia

############################################################
# COUNTERFACTUAL ESTIMATION
#
# Estimates the same structural model but on counterfactual
# data without crash shocks (pure random walk).
############################################################

# NOTE: Packages should be loaded in master script

########################################################################
# 0. User settings
########################################################################

# Path to COUNTERFACTUAL panel data:
@isdefined(DATA_PATH) || (const DATA_PATH = joinpath(@__DIR__, "data", "data_panel_counterfactual.csv"))

# Figures output directory
@isdefined(FIGURES_PATH) || (const FIGURES_PATH = joinpath(@__DIR__, "figures", "counterfactual"))

# Bounds for raw parameters [ρ_raw, γ_g_raw]
@isdefined(PARAM_LOWER) || (const PARAM_LOWER = [-5.0, -5.0])
@isdefined(PARAM_UPPER) || (const PARAM_UPPER = [ 3.0,  3.0])

# Initial guess for raw parameters
@isdefined(THETA0) || (const THETA0 = [-1.0, -1.0])   # implies ρ ≈ 0.37, γ_g ≈ 0.37

# Discount factors for scenario runs
@isdefined(BETA_VALUES) || (const BETA_VALUES = [0.2])

########################################################################
# Results storage structure
########################################################################

struct EstimationResult
    β::Float64
    θ_raw::Vector{Float64}
    ρ::Float64
    γ_g::Float64
    loglik::Float64
    V::Vector{Float64}
    v_a::Matrix{Float64}
    P_choice::Matrix{Float64}
    h_star::Float64
    se_ρ::Float64
    se_γ_g::Float64
end

########################################################################
# 1. Load data and preprocess
########################################################################

function load_data(path::String)
    df = CSV.read(path, DataFrame)

    println("Columns found in CSV: ", names(df))
    println("Column types: ", eltype.(eachcol(df)))

    # Map actions to Int: 1 = stay, 2 = deleverage
    if !("action" in names(df))
        error("Data must contain :action column. Found columns: $(names(df))")
    end

    if eltype(df.action) <: AbstractString
        df.action = map(a ->
            a == "stay" ? 1 :
            a == "deleverage" ? 2 :
            error("Unknown action string: $a"), df.action)
    elseif eltype(df.action) <: Integer
        # Assume 0 = stay, 1 = deleverage -> map to 1 and 2
        if maximum(df.action) <= 1
            df.action .= df.action .+ 1
        else
            # assume already 1/2
        end
    else
        error("Unsupported action type for :action column.")
    end

    # Sort by account_id, t
    sort!(df, [:account_id, :t])

    # Restrict to h within shared grid bounds [H_MIN, H_MAX]
    filter!(row -> (H_MIN <= row.h <= H_MAX), df)

    # Only use data after day 10
    filter!(row -> row.t > 10, df)

    return df
end

########################################################################
# 2. Log-likelihood with CCP interpolation
########################################################################

"""
    loglikelihood(θ_raw, β, h_t, action_t)

θ_raw    :: Vector (raw parameters for [ρ, γ_g])
β        :: discount factor
h_t      :: Vector{Float64}, observed health factors
action_t :: Vector{Int},     observed actions (1=stay, 2=deleverage)

Uses structural value function + CCPs on H_GRID and interpolates CCPs
for each observed (h_t, action_t).
"""
function loglikelihood(θ_raw::AbstractVector{T},
                       β::Real,
                       h_t::AbstractVector{<:Real},
                       action_t::AbstractVector{Int}) where T<:Real

    if !all(isfinite.(θ_raw)) || β <= 0.0 || β >= 1.0
        return -1e10
    end

    # Solve structural DDC with shared transitions
    V, v_a = solve_value_function(θ_raw, β)

    if !all(isfinite.(V)) || !all(isfinite.(v_a))
        return -1e10
    end

    P_choice = choice_probabilities(v_a)
    p_stay_grid = collect(@view P_choice[1, :])
    p_del_grid  = collect(@view P_choice[2, :])

    ll = 0.0
    N  = length(h_t)

    @inbounds for n in 1:N
        h = float(h_t[n])
        a = action_t[n]

        p = if a == 1
            interp1(H_GRID, p_stay_grid, h)
        elseif a == 2
            interp1(H_GRID, p_del_grid,  h)
        else
            return -1e10
        end

        p = clamp(p, LOG_EPS, 1.0)
        ll += log(p)
    end

    if !isfinite(ll)
        return -1e10
    end

    return ll
end

########################################################################
# 3. MLE wrapper struct
########################################################################

if !@isdefined(DDCProblem)
    struct DDCProblem
        h_t::Vector{Float64}
        action_t::Vector{Int}
        β::Float64
    end
end

function negative_loglikelihood(θ_raw::AbstractVector{T}, prob::DDCProblem) where T<:Real
    return -loglikelihood(θ_raw, prob.β, prob.h_t, prob.action_t)
end

########################################################################
# 3.5. Plotting Functions
########################################################################

function create_all_figures(results::Vector{EstimationResult}, df::DataFrame, 
                           h_t::Vector{Float64}, action_t::Vector{Int})
    # Create figures directory if it doesn't exist
    mkpath(FIGURES_PATH)
    
    # 1. Policy Function
    plot_policy_function(results)
    
    # 2. Value Function
    plot_value_function(results)
    
    # 3. Observed vs Predicted Actions
    plot_observed_vs_predicted(results, h_t, action_t)
    
    # 4. Time series comparison
    plot_health_timeseries(df)
    
    println("All counterfactual figures saved to: $FIGURES_PATH")
end

function plot_policy_function(results::Vector{EstimationResult})
    p = plot(title = "Counterfactual: Policy Function P(deleverage | h)",
             xlabel = "Health Factor h",
             ylabel = "Probability of Deleveraging",
             legend = :topleft,
             size = (800, 600),
             linewidth = 2.5)
    
    hline!([0.5], label = "50% threshold", linestyle = :dash, color = :gray, linewidth = 1.5)
    
    colors = [:green, :blue, :red]
    for (idx, res) in enumerate(results)
        p_delever = res.P_choice[2, :]
        plot!(H_GRID, p_delever, 
              label = @sprintf("β = %.2f", res.β),
              color = colors[min(idx, length(colors))],
              linewidth = 2.5)
        
        if !isnan(res.h_star)
            vline!([res.h_star], 
                   label = @sprintf("h* = %.3f (β=%.2f)", res.h_star, res.β),
                   linestyle = :dot, 
                   color = colors[min(idx, length(colors))])
        end
    end
    
    savefig(p, joinpath(FIGURES_PATH, "01_policy_function_counterfactual.png"))
    println("  ✓ Saved: 01_policy_function_counterfactual.png")
end

function plot_value_function(results::Vector{EstimationResult})
    p = plot(title = "Counterfactual: Value Function V(h)",
             xlabel = "Health Factor h",
             ylabel = "Value V(h)",
             legend = :bottomright,
             size = (800, 600),
             linewidth = 2.5)
    
    colors = [:green, :blue, :red]
    for (idx, res) in enumerate(results)
        plot!(H_GRID, res.V, 
              label = @sprintf("β = %.2f", res.β),
              color = colors[min(idx, length(colors))],
              linewidth = 2.5)
    end
    
    savefig(p, joinpath(FIGURES_PATH, "02_value_function_counterfactual.png"))
    println("  ✓ Saved: 02_value_function_counterfactual.png")
end

function plot_observed_vs_predicted(results::Vector{EstimationResult}, 
                                   h_t::Vector{Float64}, action_t::Vector{Int})
    best_res = results[argmax([r.loglik for r in results])]
    
    n_bins = 20
    h_bins = range(H_MIN, H_MAX, length=n_bins+1)
    bin_centers = [(h_bins[i] + h_bins[i+1])/2 for i in 1:n_bins]
    
    empirical_delever = zeros(n_bins)
    predicted_delever = zeros(n_bins)
    bin_counts = zeros(Int, n_bins)
    
    p_delever_grid = best_res.P_choice[2, :]
    
    for (h, a) in zip(h_t, action_t)
        bin_idx = searchsortedlast(h_bins, h)
        bin_idx = clamp(bin_idx, 1, n_bins)
        
        bin_counts[bin_idx] += 1
        empirical_delever[bin_idx] += (a == 2 ? 1 : 0)
        predicted_delever[bin_idx] += interp1(H_GRID, p_delever_grid, h)
    end
    
    for i in 1:n_bins
        if bin_counts[i] > 0
            empirical_delever[i] /= bin_counts[i]
            predicted_delever[i] /= bin_counts[i]
        end
    end
    
    p = plot(title = "Counterfactual: Observed vs Predicted",
             xlabel = "Health Factor h",
             ylabel = "Share Deleveraging",
             legend = :topleft,
             size = (800, 600),
             linewidth = 2.5)
    
    scatter!(bin_centers, empirical_delever, 
             label = "Empirical", 
             color = :green, 
             markersize = 6,
             alpha = 0.7)
    
    plot!(bin_centers, predicted_delever, 
          label = @sprintf("Model (β=%.2f)", best_res.β),
          color = :red, 
          linewidth = 2.5)
    
    savefig(p, joinpath(FIGURES_PATH, "03_observed_vs_predicted_counterfactual.png"))
    println("  ✓ Saved: 03_observed_vs_predicted_counterfactual.png")
end

function plot_health_timeseries(df::DataFrame)
    avg_health = combine(groupby(df, :t), :h => mean => :avg_h)
    sort!(avg_health, :t)
    
    p = plot(avg_health.t, avg_health.avg_h,
             xlabel = "Time (t)",
             ylabel = "Average Health Factor",
             title = "Counterfactual: Average Health Over Time",
             legend = :bottomright,
             linewidth = 2,
             color = :green,
             label = "Avg Health",
             size = (800, 600))
    
    hline!([H_BAR], linestyle = :dash, color = :gray, alpha = 0.5, label = "H_BAR")
    
    savefig(p, joinpath(FIGURES_PATH, "04_health_timeseries_counterfactual.png"))
    println("  ✓ Saved: 04_health_timeseries_counterfactual.png")
end

function print_final_summary(results::Vector{EstimationResult})
    println("\n" * "="^80)
    println("  COUNTERFACTUAL ESTIMATION RESULTS SUMMARY")
    println("="^80)
    
    println("\nEstimated Parameters (No Crash Shocks):")
    println("-" * "-"^79)
    @printf("%-10s | %-20s | %-20s | %-15s | %-12s\n", 
            "β", "ρ (risk)", "γ_g (gas)", "Log-Lik", "h* threshold")
    println("-" * "-"^79)
    
    for res in results
        if !isnan(res.se_ρ) && !isnan(res.se_γ_g)
            @printf("%-10.4f | %.4f (SE: %.4f) | %.4f (SE: %.4f) | %-15.2f | %-12.4f\n",
                    res.β, res.ρ, res.se_ρ, res.γ_g, res.se_γ_g, res.loglik, res.h_star)
        else
            @printf("%-10.4f | %-20.4f | %-20.4f | %-15.2f | %-12.4f\n",
                    res.β, res.ρ, res.γ_g, res.loglik, res.h_star)
        end
    end
    println("-" * "-"^79)
    
    best_idx = argmax([r.loglik for r in results])
    best_res = results[best_idx]
    
    println("\n📊 BEST FIT MODEL:")
    @printf("  β   = %.4f\n", best_res.β)
    if !isnan(best_res.se_ρ) && !isnan(best_res.se_γ_g)
        @printf("  ρ   = %.4f  (SE: %.4f)  [risk penalty scale]\n", best_res.ρ, best_res.se_ρ)
        @printf("  γ_g = %.4f  (SE: %.4f)  [deleverage gas cost]\n", best_res.γ_g, best_res.se_γ_g)
    else
        @printf("  ρ   = %.4f  (risk penalty scale)\n", best_res.ρ)
        @printf("  γ_g = %.4f  (deleverage gas cost)\n", best_res.γ_g)
    end
    @printf("  h*  = %.4f  (threshold health factor)\n", best_res.h_star)
    @printf("  LL  = %.2f\n", best_res.loglik)
    
    println("\n📈 COUNTERFACTUAL INTERPRETATION:")
    println("  • Without crash shocks, behavior is driven by:")
    println("    - Baseline drift in health (LogNormal shocks)")
    println("    - Risk-return trade-off without extreme events")
    
    for res in results
        avg_delever_prob = mean(res.P_choice[2, :])
        @printf("    - β=%.2f: Avg deleveraging probability = %.3f\n", 
                res.β, avg_delever_prob)
    end
    
    println("\n📁 All figures saved to: $FIGURES_PATH")
    println("="^80)
end

########################################################################
# 4. Main routine: run MLE for multiple β
########################################################################

function main()
    println("="^70)
    println("  COUNTERFACTUAL DEFI DDC ESTIMATION")
    println("  (No Crash Shocks - Pure Random Walk)")
    println("="^70)

    # 1) Load counterfactual data
    println("\nLoading data from: $DATA_PATH")
    df = load_data(DATA_PATH)
    println("Number of observations after filtering: $(nrow(df))")

    # Extract (h_t, action_t) for likelihood
    h_t      = collect(Float64, df.h)
    action_t = collect(Int,     df.action)

    println("Shared state grid: N = $(length(H_GRID)), h ∈ [$(H_MIN), $(H_MAX)]")

    # Store results for all β values
    results = EstimationResult[]

    for β in BETA_VALUES
        println("\n" * "="^70)
        @printf("Estimating counterfactual model for β = %.4f\n", β)
        println("="^70)

        prob = DDCProblem(h_t, action_t, β)

        obj(θ) = negative_loglikelihood(θ, prob)

        ll0 = -obj(THETA0)
        @printf("Initial log-likelihood (θ0) = %.4f\n", ll0)
        if !isfinite(ll0)
            @warn "Initial log-likelihood is not finite. Skipping β = $β"
            continue
        end

        println("\nStarting optimization with LBFGS...")
        
        iter_count = [0]
        last_ll = [ll0]
        
        function obj_callback(state)
            iter_count[1] += 1
            if iter_count[1] % 5 == 0
                current_ll = -state.value
                @printf("  Iter %4d: LL = %12.4f  |g| = %.2e\n", 
                        iter_count[1], current_ll, state.g_norm)
                last_ll[1] = current_ll
            end
            return false
        end

        res_opt = optimize(
            obj,
            PARAM_LOWER,
            PARAM_UPPER,
            THETA0,
            Fminbox(LBFGS(linesearch = LineSearches.BackTracking())),
            Optim.Options(
                iterations = 1000,
                g_tol = 1e-4,
                show_trace = false,
                callback = obj_callback
            )
        )

        θ_hat = Optim.minimizer(res_opt)
        ll_final = -Optim.minimum(res_opt)

        ρ_hat, γg_hat = transform_params(θ_hat)

        # Calculate standard errors from Hessian
        println("\n✓ Optimization finished")
        println("Calculating standard errors...")
        
        H = ForwardDiff.hessian(x -> -obj(x), θ_hat)
        
        # Standard errors from inverse Hessian
        se_ρ = NaN
        se_γg = NaN
        try
            cov_matrix = inv(H)
            se_raw = sqrt.(diag(cov_matrix))
            
            # Transform standard errors using delta method
            se_ρ = ρ_hat * se_raw[1]
            se_γg = γg_hat * se_raw[2]
            
            @printf("  Estimated θ_raw = [%.4f, %.4f]\n", θ_hat[1], θ_hat[2])
            @printf("  => ρ   = %.4f  (SE: %.4f)\n", ρ_hat, se_ρ)
            @printf("  => γ_g = %.4f  (SE: %.4f)\n", γg_hat, se_γg)
            @printf("  Final log-likelihood = %.4f\n", ll_final)
        catch e
            println("Warning: Could not compute standard errors (Hessian may be singular)")
            @printf("  Estimated θ_raw = [%.4f, %.4f]\n", θ_hat[1], θ_hat[2])
            @printf("  => ρ   = %.4f\n", ρ_hat)
            @printf("  => γ_g = %.4f\n", γg_hat)
            @printf("  Final log-likelihood = %.4f\n", ll_final)
        end

        # Solve value function at estimated parameters
        V_hat, v_a_hat = solve_value_function(θ_hat, β)
        P_choice_hat = choice_probabilities(v_a_hat)
        p_stay_grid_hat = collect(@view P_choice_hat[1, :])
        h_star_hat = approximate_threshold(p_stay_grid_hat; cutoff=0.5)

        @printf("  Threshold h* (50%% cutoff) = %.4f\n", h_star_hat)

        result = EstimationResult(
            β, θ_hat, ρ_hat, γg_hat, ll_final,
            V_hat, v_a_hat, P_choice_hat, h_star_hat, se_ρ, se_γg
        )
        push!(results, result)
    end

    # Create all figures
    if !isempty(results)
        println("\n" * "="^70)
        println("Creating counterfactual visualization figures...")
        println("="^70)
        create_all_figures(results, df, h_t, action_t)
        print_final_summary(results)
    end

    println("\nCounterfactual estimation complete.")
end

main()
