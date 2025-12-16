############################################################
# COUNTERFACTUAL: Simulated Panel Data Generator - NO CRASH SHOCKS
#
# Identical to data_generation.jl but WITHOUT crash shocks.
# Pure random walk dynamics with BASE_SHOCK_DIST only.
#
# - 2 actions: "stay" and "deleverage"
# - State: health factor h_t ∈ [1.0, 2.0] (shared grid)
# - Dynamics:
#     stay:       h_{t+1} = clamp(h_t * ε_{t+1}, H_MIN, H_MAX)
#                 where ε ~ BASE_SHOCK_DIST (no crash shocks)
#     deleverage: h_{t+1} = max(H_BAR, h_t)  (deterministic reset)
#
# Output:
#   data_panel_counterfactual.csv with columns:
#     account_id :: Int
#     t          :: Int   (0, 1, ..., T-1)
#     h          :: Float64
#     action     :: String  ("stay" or "deleverage")
############################################################

# NOTE: Packages should be loaded in master script

############################################################
# 1. Global Parameters
############################################################

# Structural parameters used in the generator (fixed here)
const ρ_gen   = 2.0
const γ_g_gen = 0.15
const β_gen   = 0.2

# encode as raw parameters for shared solver
const θ_raw_gen = [log(ρ_gen), log(γ_g_gen)]

# Panel / initial conditions (SAME as original)
const T           = 14          # total days (2 weeks)
const N_accounts  = 2000

const init_h_mean = 1.4
const init_h_sd   = 0.25
const INIT_DIST   = truncated(Normal(init_h_mean, init_h_sd), H_MIN, H_MAX)

############################################################
# 2. CCP interpolation helper
############################################################

"""
    choice_probs(h, p_stay_grid)

Wrapper around shared interpolation for CCPs.
"""
function choice_probs(h::Float64, p_stay_grid::Vector{Float64})
    return choice_probs_continuous(h, p_stay_grid)
end

############################################################
# 3. Panel Simulation - COUNTERFACTUAL (No Crash Shocks)
############################################################

"""
    simulate_panel(p_stay_grid)

Simulate N_accounts independent accounts over T days using
P(stay | h) interpolated from p_stay_grid.

COUNTERFACTUAL: All shocks are drawn from BASE_SHOCK_DIST.
No crash shocks applied.

Dynamics:
  - At each t, draw action based on P(a | h_t).
  - If action == "stay":
        h_{t+1} = clamp(h_t * ε_{t+1}, H_MIN, H_MAX)
    with ε ~ BASE_SHOCK_DIST (LogNormal(μ=-0.0005, σ=0.06))
  - If action == "deleverage":
        h_{t+1} = max(H_BAR, h_t).
"""
function simulate_panel(p_stay_grid::Vector{Float64})
    Random.seed!(1234)  # Same seed for comparability

    account_ids = Int[]
    times       = Int[]
    hs          = Float64[]
    actions     = String[]

    for acc in 1:N_accounts
        h = rand(INIT_DIST)

        for t in 0:(T-1)
            p_stay, p_del = choice_probs(h, p_stay_grid)
            u = rand()
            action = u < p_stay ? "stay" : "deleverage"

            push!(account_ids, acc)
            push!(times,       t)
            push!(hs,          h)
            push!(actions,     action)

            if t == T - 1
                break
            end

            if action == "stay"
                # COUNTERFACTUAL: Always use BASE_SHOCK_DIST, no crash shocks
                ε = rand(BASE_SHOCK_DIST)
                h = clamp(h * ε, H_MIN, H_MAX)
            elseif action == "deleverage"
                h = max(H_BAR, h)
            else
                error("Unknown action $action")
            end
        end
    end

    return DataFrame(
        account_id = account_ids,
        t          = times,
        h          = hs,
        action     = actions
    )
end

############################################################
# 4. Main: solve Bellman, simulate panel, write CSV
############################################################

function main()
    println("="^70)
    println("  COUNTERFACTUAL DATA GENERATION (No Crash Shocks)")
    println("="^70)
    
    println("\nSolving value function on shared health grid...")
    V, v_a = solve_value_function(θ_raw_gen, β_gen)
    P_choice = choice_probabilities(v_a)
    p_stay_grid = copy(@view P_choice[1, :])

    println("Value function solved.")
    println("  V(H_MIN) = ", V[1],
            ", V(H_BAR) ≈ ", interp1(H_GRID, V, H_BAR),
            ", V(H_MAX) = ", V[end])

    println("\n📊 Simulating panel data with PURE RANDOM WALK (no crash shocks)...")
    println("  Using BASE_SHOCK_DIST: LogNormal(μ=$MU, σ=$SIGMA)")
    df = simulate_panel(p_stay_grid)

    # Create data directory if it doesn't exist
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    
    outpath = joinpath(data_dir, "data_panel_counterfactual.csv")
    println("\n💾 Writing $outpath ...")
    CSV.write(outpath, df)
    println("✓ Done. data_panel_counterfactual.csv written with $(nrow(df)) rows.")

    # Create time series plot of average health
    println("\n📈 Creating time series plot of average health...")
    avg_health = combine(groupby(df, :t), :h => mean => :avg_h)
    sort!(avg_health, :t)
    
    p = plot(avg_health.t, avg_health.avg_h,
             xlabel = "Time (t)",
             ylabel = "Average Health Factor",
             title = "Counterfactual: Average Health Over Time (No Crash Shocks)",
             legend = false,
             linewidth = 2,
             color = :green,
             size = (800, 500))
    
    # Add reference line at H_BAR
    hline!([H_BAR], linestyle = :dash, color = :gray, alpha = 0.5, label = "H_BAR")
    
    # Create figures directory if it doesn't exist
    fig_dir = joinpath(@__DIR__, "figures")
    mkpath(fig_dir)
    figpath = joinpath(fig_dir, "health_timeseries_counterfactual.png")
    savefig(p, figpath)
    println("✓ Saved time series plot to $figpath")

    # Summary statistics
    println("\n📊 SUMMARY STATISTICS:")
    println("  Total observations: $(nrow(df))")
    println("  Deleverage actions: $(sum(df.action .== "deleverage")) ($(round(100*mean(df.action .== "deleverage"), digits=2))%)")
    println("  Stay actions:       $(sum(df.action .== "stay")) ($(round(100*mean(df.action .== "stay"), digits=2))%)")
    println("  Average h:          $(round(mean(df.h), digits=4))")
    println("  Std dev h:          $(round(std(df.h), digits=4))")
    println("  Min h:              $(round(minimum(df.h), digits=4))")
    println("  Max h:              $(round(maximum(df.h), digits=4))")
    
    println("\n" * "="^70)

    return df
end

# Call main when this file is included
main()
