#!/usr/bin/env julia

# NOTE: Packages should be loaded in master script

########################################################################
# 0. User settings
########################################################################

# Path to panel data produced by generator:
@isdefined(DATA_PATH) || (const DATA_PATH = joinpath(@__DIR__, "data", "data_panel.csv"))

# Figures output directory
@isdefined(FIGURES_PATH) || (const FIGURES_PATH = joinpath(@__DIR__, "figures"))

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
            a == "stay"       ? 1 :
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
    # 1. Policy Function (MOST IMPORTANT)
    plot_policy_function(results)
    
    # 2. Value Function
    plot_value_function(results)
    
    # 3. Choice-Specific Value Functions
    plot_choice_specific_values(results)
    
    # 4. Flow Utility Components
    plot_flow_utility_components(results)
    
    # 5. Transition Dynamics
    plot_transition_dynamics(results)
    
    # 6. Observed vs Predicted Actions
    plot_observed_vs_predicted(results, h_t, action_t)
    
    # 7. β-Sweep Comparative Statics
    plot_beta_comparative_statics(results)
    
    # 8. Simulated Paths
    plot_simulated_paths(results)
    
    # 9. Shock Scenario
    plot_shock_scenario(results[end])  # Use highest β (most forward-looking)
    
    println("All figures saved to: $FIGURES_PATH")
end

# 1. Policy Function P(deleverage | h)
function plot_policy_function(results::Vector{EstimationResult})
    p = plot(title = "Policy Function: P(deleverage | h)",
             xlabel = "Health Factor h",
             ylabel = "Probability of Deleveraging",
             legend = :topleft,
             size = (800, 600),
             linewidth = 2.5)
    
    hline!([0.5], label = "50% threshold", linestyle = :dash, color = :gray, linewidth = 1.5)
    
    colors = [:blue, :red, :green]
    for (idx, res) in enumerate(results)
        p_delever = res.P_choice[2, :]
        plot!(H_GRID, p_delever, 
              label = @sprintf("β = %.2f", res.β),
              color = colors[idx],
              linewidth = 2.5)
        
        # Mark threshold
        if !isnan(res.h_star)
            vline!([res.h_star], 
                   label = @sprintf("h* (β=%.2f) = %.3f", res.β, res.h_star),
                   linestyle = :dot, color = colors[idx], linewidth = 1.5)
        end
    end
    
    savefig(p, joinpath(FIGURES_PATH, "01_policy_function.png"))
    println("  ✓ Saved: 01_policy_function.png")
end

# 2. Value Function V(h)
function plot_value_function(results::Vector{EstimationResult})
    p = plot(title = "Value Function V(h)",
             xlabel = "Health Factor h",
             ylabel = "Value V(h)",
             legend = :bottomright,
             size = (800, 600),
             linewidth = 2.5)
    
    colors = [:blue, :red, :green]
    for (idx, res) in enumerate(results)
        plot!(H_GRID, res.V, 
              label = @sprintf("β = %.2f", res.β),
              color = colors[idx],
              linewidth = 2.5)
    end
    
    savefig(p, joinpath(FIGURES_PATH, "02_value_function.png"))
    println("  ✓ Saved: 02_value_function.png")
end

# 3. Choice-Specific Value Functions v(h, a)
function plot_choice_specific_values(results::Vector{EstimationResult})
    n_beta = length(results)
    plots_array = []
    
    for (idx, res) in enumerate(results)
        p = plot(title = @sprintf("β = %.2f", res.β),
                 xlabel = "Health Factor h",
                 ylabel = "Choice-Specific Value",
                 legend = :bottomright,
                 linewidth = 2.5)
        
        plot!(H_GRID, res.v_a[1, :], label = "v(h, stay)", color = :blue, linewidth = 2.5)
        plot!(H_GRID, res.v_a[2, :], label = "v(h, deleverage)", color = :red, linewidth = 2.5)
        
        # Mark threshold where curves cross
        if !isnan(res.h_star)
            vline!([res.h_star], label = "Threshold", linestyle = :dash, color = :black)
        end
        
        push!(plots_array, p)
    end
    
    combined = plot(plots_array..., layout = (1, n_beta), size = (1200, 400))
    savefig(combined, joinpath(FIGURES_PATH, "03_choice_specific_values.png"))
    println("  ✓ Saved: 03_choice_specific_values.png")
end

# 4. Flow Utility Components
function plot_flow_utility_components(results::Vector{EstimationResult})
    # Use parameters from best-fit model (highest log-likelihood)
    best_res = results[argmax([r.loglik for r in results])]
    ρ = best_res.ρ
    γ_g = best_res.γ_g
    
    h_range = range(H_MIN, H_MAX, length=200)
    
    # Risk penalty
    risk_penalty = [-ρ * max(0, H_BAR - h)^2 for h in h_range]
    
    p = plot(title = "Flow Utility Components",
             xlabel = "Health Factor h",
             ylabel = "Utility",
             legend = :bottomright,
             size = (800, 600),
             linewidth = 2.5)
    
    plot!(h_range, risk_penalty, 
          label = @sprintf("Risk penalty: -ρ(h̄-h)² (ρ=%.3f)", ρ),
          color = :blue, linewidth = 2.5)
    
    hline!([-γ_g], 
           label = @sprintf("Gas cost: -γ_g (γ_g=%.3f)", γ_g),
           linestyle = :dash, color = :red, linewidth = 2.5)
    
    vline!([H_BAR], label = "Target h̄", linestyle = :dot, color = :gray, linewidth = 1.5)
    
    savefig(p, joinpath(FIGURES_PATH, "04_flow_utility_components.png"))
    println("  ✓ Saved: 04_flow_utility_components.png")
end

# 5. Transition Dynamics
function plot_transition_dynamics(results::Vector{EstimationResult})
    h_range = collect(range(H_MIN, H_MAX, length=50))
    
    p = plot(title = "Expected Next-Period Health: E[h' | h, a]",
             xlabel = "Current Health h",
             ylabel = "Expected Next Health E[h']",
             legend = :bottomright,
             size = (800, 600),
             linewidth = 2.5)
    
    # 45-degree line (no change)
    plot!(h_range, h_range, label = "45° line (h'=h)", 
          linestyle = :dash, color = :gray, linewidth = 1.5)
    
    # Use one β for illustration (highest)
    res = results[end]
    V = res.V
    
    # E[h' | h, stay] - Monte Carlo approximation
    E_h_stay = Float64[]
    for h in h_range
        expected_h = 0.0
        n_samples = 500
        for _ in 1:n_samples
            ε = rand(BASE_SHOCK_DIST)
            h_next = clamp(h * ε, H_MIN, H_MAX)
            expected_h += h_next
        end
        push!(E_h_stay, expected_h / n_samples)
    end
    
    # E[h' | h, deleverage] - deterministic
    E_h_delever = [max(H_BAR, h) for h in h_range]
    
    plot!(h_range, E_h_stay, label = "E[h' | h, stay]", color = :blue, linewidth = 2.5)
    plot!(h_range, E_h_delever, label = "E[h' | h, deleverage]", color = :red, linewidth = 2.5)
    vline!([H_BAR], label = "Target h̄", linestyle = :dot, color = :orange, linewidth = 1.5)
    
    savefig(p, joinpath(FIGURES_PATH, "05_transition_dynamics.png"))
    println("  ✓ Saved: 05_transition_dynamics.png")
end

# 6. Observed vs Predicted Actions
function plot_observed_vs_predicted(results::Vector{EstimationResult}, 
                                   h_t::Vector{Float64}, action_t::Vector{Int})
    # Use best-fit model
    best_res = results[argmax([r.loglik for r in results])]
    
    # Create bins
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
    
    # Normalize
    for i in 1:n_bins
        if bin_counts[i] > 0
            empirical_delever[i] /= bin_counts[i]
            predicted_delever[i] /= bin_counts[i]
        end
    end
    
    p = plot(title = "Observed vs Predicted Deleveraging Rates",
             xlabel = "Health Factor h",
             ylabel = "Share Deleveraging",
             legend = :topleft,
             size = (800, 600),
             linewidth = 2.5)
    
    scatter!(bin_centers, empirical_delever, 
             label = "Empirical", 
             color = :blue, 
             markersize = 6,
             alpha = 0.7)
    
    plot!(bin_centers, predicted_delever, 
          label = @sprintf("Model (β=%.2f)", best_res.β),
          color = :red, 
          linewidth = 2.5)
    
    savefig(p, joinpath(FIGURES_PATH, "06_observed_vs_predicted.png"))
    println("  ✓ Saved: 06_observed_vs_predicted.png")
end

# 7. β-Sweep Comparative Statics
function plot_beta_comparative_statics(results::Vector{EstimationResult})
    betas = [r.β for r in results]
    thresholds = [r.h_star for r in results]
    
    # Panel 1: Policy functions
    p1 = plot(title = "Policy Functions Across β",
              xlabel = "Health Factor h",
              ylabel = "P(deleverage | h)",
              legend = :topleft,
              linewidth = 2.5)
    
    colors = [:blue, :red, :green]
    for (idx, res) in enumerate(results)
        plot!(H_GRID, res.P_choice[2, :], 
              label = @sprintf("β = %.2f", res.β),
              color = colors[idx],
              linewidth = 2.5)
    end
    
    # Panel 2: Thresholds
    p2 = plot(title = "Threshold h* vs β",
              xlabel = "Discount Factor β",
              ylabel = "Threshold h*",
              legend = false,
              linewidth = 2.5,
              marker = :circle,
              markersize = 8)
    
    plot!(betas, thresholds, color = :blue, linewidth = 2.5)
    
    # Panel 3: Average deleveraging rate
    p3 = plot(title = "Implied Deleveraging Rate",
              xlabel = "Discount Factor β",
              ylabel = "Avg P(deleverage)",
              legend = false,
              linewidth = 2.5,
              marker = :circle,
              markersize = 8)
    
    avg_rates = [mean(r.P_choice[2, :]) for r in results]
    plot!(betas, avg_rates, color = :red, linewidth = 2.5)
    
    combined = plot(p1, p2, p3, layout = (1, 3), size = (1500, 400))
    savefig(combined, joinpath(FIGURES_PATH, "07_beta_comparative_statics.png"))
    println("  ✓ Saved: 07_beta_comparative_statics.png")
end

# 9. Simulated Paths
function plot_simulated_paths(results::Vector{EstimationResult})
    # Use the most forward-looking agent
    res = results[end]
    ρ, γ_g = res.ρ, res.γ_g
    β = res.β
    
    Random.seed!(42)
    n_agents = 10
    T_sim = 100
    
    p = plot(title = @sprintf("Simulated Health Trajectories (β=%.2f)", β),
             xlabel = "Time Period",
             ylabel = "Health Factor h",
             legend = false,
             size = (1000, 600),
             alpha = 0.6)
    
    hline!([H_BAR], linestyle = :dash, color = :gray, linewidth = 2, label = "Target h̄")
    hline!([res.h_star], linestyle = :dot, color = :red, linewidth = 2, label = "Threshold h*")
    
    for agent in 1:n_agents
        h_path = zeros(T_sim)
        h_path[1] = H_MIN + rand() * (H_MAX - H_MIN)  # Random initial h
        
        delever_times = Int[]
        
        for t in 2:T_sim
            h = h_path[t-1]
            
            # Get optimal action probability
            p_delever = interp1(H_GRID, res.P_choice[2, :], h)
            action = rand() < p_delever ? 2 : 1
            
            if action == 2
                push!(delever_times, t)
                h_path[t] = max(H_BAR, h)
            else
                ε = rand(BASE_SHOCK_DIST)
                h_path[t] = clamp(h * ε, H_MIN, H_MAX)
            end
        end
        
        plot!(1:T_sim, h_path, color = :blue, linewidth = 1.5, alpha = 0.5)
        
        # Mark deleveraging events
        if !isempty(delever_times)
            scatter!(delever_times, h_path[delever_times], 
                    color = :red, markersize = 4, alpha = 0.7)
        end
    end
    
    savefig(p, joinpath(FIGURES_PATH, "08_simulated_paths.png"))
    println("  ✓ Saved: 08_simulated_paths.png")
end

# 10. Shock Scenario
function plot_shock_scenario(res::EstimationResult)
    # Simulate a crash: large negative shock to all h values
    h_pre_crash = collect(range(H_MIN, H_MAX, length=50))
    crash_multiplier = 0.85  # 15% drop in health
    h_post_crash = [clamp(h * crash_multiplier, H_MIN, H_MAX) for h in h_pre_crash]
    
    # CCPs before and after crash
    p_delever_pre = [interp1(H_GRID, res.P_choice[2, :], h) for h in h_pre_crash]
    p_delever_post = [interp1(H_GRID, res.P_choice[2, :], h) for h in h_post_crash]
    
    p1 = plot(title = "Policy Response to Crash",
              xlabel = "Health Factor h",
              ylabel = "P(deleverage | h)",
              legend = :topleft,
              linewidth = 2.5,
              size = (800, 500))
    
    plot!(h_pre_crash, p_delever_pre, label = "Pre-crash", color = :blue, linewidth = 2.5)
    plot!(h_post_crash, p_delever_post, label = "Post-crash (shifted)", color = :red, linewidth = 2.5)
    
    # Distribution after crash under different actions
    p2 = plot(title = "Post-Crash Health Distribution",
              xlabel = "Health Factor h",
              ylabel = "Density",
              legend = :topright,
              linewidth = 2.5,
              size = (800, 500))
    
    # If stay: remains at crashed level
    histogram!(h_post_crash, bins=20, alpha=0.5, label="If stay", normalize=:pdf, color=:blue)
    
    # If deleverage: jumps to max(H_BAR, h)
    h_after_delever = [max(H_BAR, h) for h in h_post_crash]
    histogram!(h_after_delever, bins=20, alpha=0.5, label="If deleverage", normalize=:pdf, color=:red)
    
    combined = plot(p1, p2, layout = (2, 1), size = (800, 900))
    savefig(combined, joinpath(FIGURES_PATH, "09_shock_scenario.png"))
    println("  ✓ Saved: 09_shock_scenario.png")
end

function print_final_summary(results::Vector{EstimationResult})
    println("\n" * "="^80)
    println("  FINAL ESTIMATION RESULTS SUMMARY")
    println("="^80)
    
    println("\nEstimated Parameters Across All β Values:")
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
    
    # Best fit by log-likelihood
    best_idx = argmax([r.loglik for r in results])
    best_res = results[best_idx]
    
    println("\n📊 BEST FIT MODEL (Highest Log-Likelihood):")
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
    
    println("\n📈 BEHAVIORAL INTERPRETATION:")
    println("  • Forward-looking behavior calibration:")
    for res in results
        avg_delever_prob = mean(res.P_choice[2, :])
        @printf("    - β=%.2f: Avg deleveraging probability = %.3f\n", 
                res.β, avg_delever_prob)
    end
    
    println("\n  • Risk vs Gas Cost Trade-off:")
    @printf("    - Risk penalty dominates when (h̄-h)² > %.4f\n", 
            best_res.γ_g / best_res.ρ)
    @printf("    - At threshold h*=%.3f, agents are indifferent\n", best_res.h_star)
    
    println("\n  • Policy Implications:")
    if best_res.β > 0.7
        println("    - Users exhibit HIGH patience (forward-looking)")
        println("    - Respond proactively to risk before liquidation threat")
    elseif best_res.β > 0.4
        println("    - Users exhibit MODERATE patience")
        println("    - Balance short-term costs with medium-term risk")
    else
        println("    - Users exhibit LOW patience (myopic)")
        println("    - React mainly to immediate liquidation risk")
    end
    
    println("\n📁 All figures saved to: $FIGURES_PATH")
    println("="^80)
end

########################################################################
# 4. Main routine: run MLE for multiple β
########################################################################

function main()
    println("===========================================================")
    println("  DEFI DDC ESTIMATION (structural, 2 actions, shared utils)")
    println("===========================================================")

    # 1) Load data
    println("Loading data from: $DATA_PATH")
    df = load_data(DATA_PATH)
    println("Number of observations after filtering: $(nrow(df))")

    # Extract (h_t, action_t) for likelihood
    h_t      = collect(Float64, df.h)
    action_t = collect(Int,     df.action)

    println("Shared state grid: N = $(length(H_GRID)), h ∈ [$(H_MIN), $(H_MAX)]")

    # Store results for all β values
    results = EstimationResult[]

    for β in BETA_VALUES
        println("\n===========================================================")
        @printf("Estimating model for β = %.4f\n", β)
        println("===========================================================")

        prob = DDCProblem(h_t, action_t, β)

        obj(θ) = negative_loglikelihood(θ, prob)

        ll0 = -obj(THETA0)
        @printf("Initial log-likelihood (θ0) = %.4f\n", ll0)
        if !isfinite(ll0)
            println("Initial log-likelihood is not finite; skipping β = $β.")
            continue
        end

        println("\nStarting optimization with LBFGS (gradient-based method)...")
        println("Showing every 5th iteration.\n")
        
        iter_count = [0]
        
        function traced_obj(θ)
            val = obj(θ)
            iter_count[1] += 1
            
            if iter_count[1] % 5 == 0 || iter_count[1] == 1
                ρ_curr, γg_curr = transform_params(θ)
                @printf("  Iter %3d: LL = %10.4f | ρ = %.4f, γ_g = %.4f\n", 
                        iter_count[1], -val, ρ_curr, γg_curr)
            end
            
            return val
        end

        # LBFGS with automatic differentiation and tighter tolerances
        opts = Optim.Options(
            iterations = 1000,
            show_trace = false,
            show_every = 5,
            g_tol = 1e-4,      # Gradient tolerance
            f_tol = 1e-6,      # Function value tolerance
            x_tol = 1e-6       # Parameter tolerance
        )

        # Use more robust line search
        res = optimize(traced_obj, PARAM_LOWER, PARAM_UPPER, THETA0, 
                      Fminbox(LBFGS(; linesearch = LineSearches.BackTracking())), 
                      opts; autodiff = :forward)

        println("Optimization completed.")
        println(res)

        θ_hat_raw = Optim.minimizer(res)
        ρ_hat, γg_hat = transform_params(θ_hat_raw)
        ll_hat = -Optim.minimum(res)

        # Calculate standard errors from Hessian
        println("\nCalculating standard errors...")
        H = ForwardDiff.hessian(x -> -loglikelihood(x, β, h_t, action_t), θ_hat_raw)
        
        # Standard errors from inverse Hessian (Fisher information)
        try
            cov_matrix = inv(H)
            se_raw = sqrt.(diag(cov_matrix))
            
            # Transform standard errors using delta method
            # For ρ = exp(θ₁), SE(ρ) ≈ ρ * SE(θ₁)
            # For γ = exp(θ₂), SE(γ) ≈ γ * SE(θ₂)
            se_ρ = ρ_hat * se_raw[1]
            se_γg = γg_hat * se_raw[2]
            
            println("\nEstimated structural parameters for this β:")
            @printf("  ρ   (risk penalty scale)    = %.4f  (SE: %.4f)\n", ρ_hat, se_ρ)
            @printf("  γ_g (deleverage gas cost)   = %.4f  (SE: %.4f)\n", γg_hat, se_γg)
            @printf("  β   (discount factor)       = %.4f\n", β)
            @printf("  Log-likelihood              = %.4f\n", ll_hat)
        catch e
            println("Warning: Could not compute standard errors (Hessian may be singular)")
            println("\nEstimated structural parameters for this β:")
            @printf("  ρ   (risk penalty scale)    = %.4f\n", ρ_hat)
            @printf("  γ_g (deleverage gas cost)   = %.4f\n", γg_hat)
            @printf("  β   (discount factor)       = %.4f\n", β)
            @printf("  Log-likelihood              = %.4f\n", ll_hat)
        end

        # Policy / thresholds for this β (enable convergence warnings for final computation)
        V_hat, v_a_hat     = solve_value_function(θ_hat_raw, β; warn_convergence=true)
        P_choice_hat       = choice_probabilities(v_a_hat)
        p_stay_hat_grid    = @view P_choice_hat[1, :]

        h_star = approximate_threshold(p_stay_hat_grid; cutoff=0.5)

        println("\nExample choice probabilities at a few grid points:")
        for i in round.(Int, range(1, length(H_GRID); length=5))
            h    = H_GRID[i]
            p_st = P_choice_hat[1, i]
            p_dl = P_choice_hat[2, i]
            @printf("  h = %.3f -> P(stay)=%.3f, P(delever)=%.3f\n", h, p_st, p_dl)
        end

        if !isnan(h_star)
            @printf("\nApproximate threshold h* where P(deleverage) ≈ 0.5: h* ≈ %.4f\n", h_star)
        else
            println("\nNo interior threshold with P(deleverage) ≈ 0.5 found on [H_MIN, H_MAX].")
        end

        # Store result with standard errors
        if @isdefined se_ρ
            push!(results, EstimationResult(β, θ_hat_raw, ρ_hat, γg_hat, ll_hat, 
                                            V_hat, v_a_hat, P_choice_hat, h_star, se_ρ, se_γg))
        else
            # If SE calculation failed, use NaN
            push!(results, EstimationResult(β, θ_hat_raw, ρ_hat, γg_hat, ll_hat, 
                                            V_hat, v_a_hat, P_choice_hat, h_star, NaN, NaN))
        end
    end

    # Create all figures
    if !isempty(results)
        println("\n===========================================================")
        println("  CREATING FIGURES")
        println("===========================================================")
        mkpath(FIGURES_PATH)
        create_all_figures(results, df, h_t, action_t)
        
        # Print final summary
        print_final_summary(results)
    end

    println("\nDone.")
end

main()