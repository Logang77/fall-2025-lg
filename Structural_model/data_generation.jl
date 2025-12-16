############################################################
# Simulated Panel Data Generator for Rust-Style DDC DeFi Model
#
# - 2 actions: "stay" and "deleverage"
# - State: health factor h_t ∈ [1.0, 2.0] (shared grid)
# - Dynamics:
#     stay:       h_{t+1} = clamp(h_t      * ε_{t+1}, H_MIN, H_MAX)
#     deleverage: h_{t+1} = max(H_BAR, h_t)  (deterministic reset)
# - One large negative shock to ε at t = shock_day (simulation only)
#
# Output:
#   data_panel.csv with columns:
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

# Crash period: days 4-10 with minimum -50% at day 7
const CRASH_START = 4
const CRASH_END = 10
const CRASH_PEAK = 7  # day with worst shock (-50%)

# Crash shock distribution (used during crash period)
const μ_s      = -0.40
const σ_s      = 0.10
const CRASH_SHOCK_DIST = LogNormal(μ_s, σ_s)

# Extreme crash shock for peak day (roughly -50%)
const PEAK_CRASH_SHOCK = 0.5  # multiplier for -50% drop

# Panel / initial conditions
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
# 3. Panel Simulation with crash shock
############################################################

"""
    simulate_panel(p_stay_grid)

Simulate N_accounts independent accounts over T days using
P(stay | h) interpolated from p_stay_grid.

Dynamics:
  - At each t, draw action based on P(a | h_t).
  - If action == "stay":
        h_{t+1} = clamp(h_t * ε_{t+1}, H_MIN, H_MAX)
    with ε drawn from:
        CRASH_SHOCK_DIST if t ∈ shock_days,
        BASE_SHOCK_DIST  otherwise.
  - If action == "deleverage":
        h_{t+1} = max(H_BAR, h_t).
"""
function simulate_panel(p_stay_grid::Vector{Float64})
    Random.seed!(1234)

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
                # Determine shock based on crash period
                if t >= CRASH_START && t <= CRASH_END
                    if t == CRASH_PEAK
                        # Peak crash day: -50% shock
                        ε = PEAK_CRASH_SHOCK
                    else
                        # Other crash period days: random crash shocks
                        ε = rand(CRASH_SHOCK_DIST)
                    end
                else
                    # Normal days: base shock distribution
                    ε = rand(BASE_SHOCK_DIST)
                end
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
    println("Solving value function on shared health grid...")
    V, v_a = solve_value_function(θ_raw_gen, β_gen)
    P_choice = choice_probabilities(v_a)
    p_stay_grid = copy(@view P_choice[1, :])

    println("Value function solved.")
    println("  V(H_MIN) = ", V[1],
            ", V(H_BAR) ≈ ", interp1(H_GRID, V, H_BAR),
            ", V(H_MAX) = ", V[end])

    println("Simulating panel data with crash period from t = $CRASH_START to $CRASH_END (peak at t = $CRASH_PEAK) ...")
    df = simulate_panel(p_stay_grid)

    # Create data directory if it doesn't exist
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    
    outpath = joinpath(data_dir, "data_panel.csv")
    println("Writing $outpath ...")
    CSV.write(outpath, df)
    println("Done. data_panel.csv written with $(nrow(df)) rows.")

    # Create time series plot of average health
    println("Creating time series plot of average health...")
    avg_health = combine(groupby(df, :t), :h => mean => :avg_h)
    sort!(avg_health, :t)
    
    p = plot(avg_health.t, avg_health.avg_h,
             xlabel = "Time (t)",
             ylabel = "Average Health Factor",
             title = "Average Health Over Time",
             legend = false,
             linewidth = 2,
             color = :blue,
             size = (800, 500))
    
    # Add shaded region for crash period
    vspan!([CRASH_START, CRASH_END], color = :red, alpha = 0.2, label = "")
    # Add vertical line at peak crash day
    vline!([CRASH_PEAK], linestyle = :dash, color = :red, linewidth = 2, alpha = 0.7, label = "")
    
    # Create figures directory if it doesn't exist
    fig_dir = joinpath(@__DIR__, "figures")
    mkpath(fig_dir)
    figpath = joinpath(fig_dir, "health_timeseries.png")
    savefig(p, figpath)
    println("Saved time series plot to $figpath")

    return df
end

# Call main when this file is included
main()
