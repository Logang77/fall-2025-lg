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
const γ_g_gen = 1.5
const β_gen   = 0.2

# encode as raw parameters for shared solver
const θ_raw_gen = [log(ρ_gen), log(γ_g_gen)]

# Panel / initial conditions (SAME as original)
const T           = 14          # total days (2 weeks)
const N_accounts  = 2000

const init_h_mean = 1.4
const init_h_sd   = 0.25
const INIT_DIST   = truncated(Normal(init_h_mean, init_h_sd), H_MIN, H_MAX)

# COUNTERFACTUAL: No transitions to crash regime
const π01_COUNTERFACTUAL = 0.0

############################################################
# 2. CCP interpolation helper
############################################################

"""
    choice_probs(h, s, p_stay_grid)

Wrapper around shared interpolation for CCPs.
"""
function choice_probs(h::Float64, s::Int, p_stay_grid::AbstractMatrix{Float64})
    return choice_probs_continuous(h, s, p_stay_grid)
end

############################################################
# 3. Panel Simulation - COUNTERFACTUAL (No Crash Shocks)
############################################################

"""
    simulate_panel(p_stay_grid)

Simulate N_accounts independent accounts over T days using
P(stay | h, s) interpolated from p_stay_grid.

COUNTERFACTUAL: π01 = 0, so no transitions to crash regime.
Regime-dependent shocks still apply if started in crash.

Dynamics:
  - At each t, draw action based on P(a | h_t, s_t).
  - If action == "stay":
        h_{t+1} = clamp(h_t * ε_{t+1}, H_MIN, H_MAX)
    with ε from regime-dependent distribution.
  - If action == "deleverage":
        h_{t+1} = max(H_BAR, h_t).
  - Regime transitions: π01 = 0 (counterfactual).
"""
function simulate_panel(p_stay_grid::AbstractMatrix{Float64})
    Random.seed!(1234)  # Same seed for comparability

    account_ids = Int[]
    times       = Int[]
    hs          = Float64[]
    actions     = String[]
    regimes     = Int[]

    for acc in 1:N_accounts
        h = rand(INIT_DIST)
        s = 0  # Initialize in normal regime

        for t in 0:(T-1)
            p_stay, p_del = choice_probs(h, s, p_stay_grid)
            u = rand()
            action = u < p_stay ? "stay" : "deleverage"

            push!(account_ids, acc)
            push!(times,       t)
            push!(hs,          h)
            push!(actions,     action)
            push!(regimes,     s)

            if t == T - 1
                break
            end

            if action == "stay"
                # Regime-dependent shock
                if s == 0
                    ε = rand(BASE_SHOCK_DIST)
                else
                    ε = rand(CRASH_SHOCK_DIST)
                end
                h = clamp(h * ε, H_MIN, H_MAX)
            elseif action == "deleverage"
                h = max(H_BAR, h)
            else
                error("Unknown action $action")
            end
            
            # COUNTERFACTUAL: π01 = 0 (no transitions to crash)
            if s == 0
                s = rand() < π01_COUNTERFACTUAL ? 1 : 0  # Will always stay at 0
            else
                s = rand() < π10_DEFAULT ? 0 : 1
            end
        end
    end

    return DataFrame(
        account_id = account_ids,
        t          = times,
        h          = hs,
        action     = actions,
        regime     = regimes
    )
end

############################################################
# 4. Main: solve Bellman, simulate panel, write CSV
############################################################

function main()
    println("="^70)
    println("  COUNTERFACTUAL DATA GENERATION (No Crash Shocks)")
    println("="^70)
    
    println("Solving value function on shared health grid with regimes (COUNTERFACTUAL: π01=0)...")
    V, v_a = solve_value_function(θ_raw_gen, β_gen; π01=π01_COUNTERFACTUAL, use_quadrature=true)
    P_choice = choice_probabilities(v_a)
    p_stay_grid = copy(P_choice[1, :, :])  # Matrix: [h_index, regime_index]

    println("Value function solved.")
    println("  V(H_MIN, s=0) = ", V[1, 1],
            ", V(H_BAR, s=0) ≈ ", interp1(H_GRID, @view(V[:, 1]), H_BAR),
            ", V(H_MAX, s=0) = ", V[end, 1])
    println("  V(H_MIN, s=1) = ", V[1, 2],
            ", V(H_BAR, s=1) ≈ ", interp1(H_GRID, @view(V[:, 2]), H_BAR),
            ", V(H_MAX, s=1) = ", V[end, 2])

    println("\n📊 Simulating panel data with PURE RANDOM WALK (no crash shocks)...")
    println("  Using BASE_SHOCK_DIST: LogNormal(μ=$MU, σ=$SIGMA)")
    df = simulate_panel(p_stay_grid)

    # Create data directory if it doesn't exist
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    
    outpath = joinpath(data_dir, "data_panel_counterfactual.csv")
    println("\n[*] Writing $outpath ...")
    CSV.write(outpath, df)
    println("[*] Done. data_panel_counterfactual.csv written with $(nrow(df)) rows.")

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
