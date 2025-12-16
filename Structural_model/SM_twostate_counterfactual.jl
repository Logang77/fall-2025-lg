#!/usr/bin/env julia

############################################################
# STRUCTURAL COUNTERFACTUAL (NO CRASH ARRIVAL)
#
# Correct workflow:
# 1) Load baseline estimates θ̂ from results/baseline_estimates.csv
# 2) Hold θ̂ fixed (NO re-estimation)
# 3) Re-solve DP with π01 = 0.0 (no crash arrival)
# 4) Optionally evaluate fit / plots on counterfactual data
############################################################

# Packages are assumed loaded in master:
# using CSV, DataFrames, Optim, ForwardDiff, Printf, Statistics, Plots

########################################################################
# 0. User settings (DO NOT allow master to override by accident)
########################################################################

const BASELINE_EST_PATH = joinpath(@__DIR__, "results", "baseline_estimates.csv")

# Data used ONLY for "observed vs predicted" plots (not estimation)
const COUNTERFACTUAL_DATA_PATH = joinpath(@__DIR__, "data", "data_panel_counterfactual.csv")

# Output figures
const FIGURES_PATH = joinpath(@__DIR__, "figures", "counterfactual")

# Which β to use
const BETA_VALUES = [0.2]

########################################################################
# Helper: load counterfactual data for plotting only
########################################################################

function load_data_for_plots(path::String)
    df = CSV.read(path, DataFrame)

    if !("action" in names(df));  error("Data must contain :action"); end
    if !("regime" in names(df));  error("Data must contain :regime"); end

    if eltype(df.action) <: AbstractString
        df.action = map(a ->
            a == "stay" ? 1 :
            a == "deleverage" ? 2 :
            error("Unknown action string: $a"), df.action)
    elseif eltype(df.action) <: Integer
        if maximum(df.action) <= 1
            df.action .= df.action .+ 1
        end
    else
        error("Unsupported action type for :action column.")
    end

    sort!(df, [:account_id, :t])

    filter!(row -> (H_MIN <= row.h <= H_MAX), df)

    # IMPORTANT: If you want to keep your original restriction, keep this.
    # But note: your counterfactual panel is only 14 days; t>10 leaves few days.
    # I keep your restriction to match your pipeline.
    filter!(row -> row.t > 10, df)

    return df
end

########################################################################
# Helper: load baseline θ̂ for a given β
########################################################################

"""
baseline_estimates.csv should contain at least:
β, θ1, θ2  (or β, theta1, theta2)
Optionally ρ, γ_g columns too.

Your master script said it saved:
θ̂ = [0.781..., 0.418...]
so we read those raw parameters back.
"""
function load_baseline_theta(beta::Real; path::String=BASELINE_EST_PATH)
    est = CSV.read(path, DataFrame)

    # tolerate a few common column name variants
    βcol = ("β" in names(est)) ? "β" : (("beta" in names(est)) ? "beta" : error("No β column in $path"))
    θ1col = ("θ1" in names(est)) ? "θ1" :
            (("theta1" in names(est)) ? "theta1" :
             (("theta_1" in names(est)) ? "theta_1" :
              (("θ_raw_1" in names(est)) ? "θ_raw_1" : error("No θ1 column in $path"))))
    θ2col = ("θ2" in names(est)) ? "θ2" :
            (("theta2" in names(est)) ? "theta2" :
             (("theta_2" in names(est)) ? "theta_2" :
              (("θ_raw_2" in names(est)) ? "θ_raw_2" : error("No θ2 column in $path"))))

    # select the closest β row (avoid float equality issues)
    idx = argmin(abs.(est[!, βcol] .- float(beta)))

    θ̂ = [float(est[idx, θ1col]), float(est[idx, θ2col])]
    return θ̂
end

########################################################################
# Helper: compute CCPs under structural counterfactual π01 = 0
########################################################################

function solve_counterfactual_policy(θ_raw::Vector{Float64}, β::Float64)
    # This is the structural counterfactual: kill crash ARRIVAL
    V, v_a = solve_value_function(θ_raw, β; π01=0.0, use_quadrature=true)
    P_choice = choice_probabilities(v_a)  # [action, h_index, regime_index]
    return V, v_a, P_choice
end

########################################################################
# Plotting (reuse your existing plot functions with small fixes)
########################################################################

function plot_policy_function_counterfactual(β::Float64, P_choice::Array{Float64,3})
    mkpath(FIGURES_PATH)

    p = plot(title="Structural Counterfactual (π01=0): P(deleverage | h)",
             xlabel="Health Factor h",
             ylabel="P(deleverage)",
             legend=:topleft,
             size=(800,600),
             linewidth=2.5)

    hline!([0.5], label="0.5", linestyle=:dash, color=:gray, linewidth=1.5)

    p_del_normal = P_choice[2, :, 1]
    p_del_crash  = P_choice[2, :, 2]

    plot!(H_GRID, p_del_normal, label=@sprintf("β=%.2f normal", β), linestyle=:solid)
    plot!(H_GRID, p_del_crash,  label=@sprintf("β=%.2f crash",  β), linestyle=:dash)

    savefig(p, joinpath(FIGURES_PATH, "01_policy_function_counterfactual.png"))
    println("  ✓ Saved: 01_policy_function_counterfactual.png")
end

function plot_value_function_counterfactual(β::Float64, V::Matrix{Float64})
    p = plot(title="Structural Counterfactual (π01=0): Value Function V(h,s)",
             xlabel="Health Factor h",
             ylabel="V(h,s)",
             legend=:bottomright,
             size=(800,600),
             linewidth=2.5)

    plot!(H_GRID, V[:,1], label=@sprintf("β=%.2f normal", β), linestyle=:solid)
    plot!(H_GRID, V[:,2], label=@sprintf("β=%.2f crash",  β), linestyle=:dash)

    savefig(p, joinpath(FIGURES_PATH, "02_value_function_counterfactual.png"))
    println("  ✓ Saved: 02_value_function_counterfactual.png")
end

function plot_observed_vs_predicted_counterfactual(df::DataFrame, P_choice::Array{Float64,3}; β::Float64)
    h_t = collect(Float64, df.h)
    a_t = collect(Int, df.action)

    n_bins = 20
    h_bins = range(H_MIN, H_MAX, length=n_bins+1)
    bin_centers = [(h_bins[i] + h_bins[i+1])/2 for i in 1:n_bins]

    empirical = zeros(n_bins)
    predicted = zeros(n_bins)
    counts = zeros(Int, n_bins)

    # average predicted CCP across regimes using empirical regime weights
    reg = collect(Int, df.regime)
    w0 = mean(reg .== 0)
    w1 = 1.0 - w0
    p_del_grid = w0 .* P_choice[2, :, 1] .+ w1 .* P_choice[2, :, 2]

    for (h, a) in zip(h_t, a_t)
        b = searchsortedlast(h_bins, h)
        b = clamp(b, 1, n_bins)

        counts[b] += 1
        empirical[b] += (a == 2 ? 1.0 : 0.0)
        predicted[b] += interp1(H_GRID, p_del_grid, h)
    end

    for i in 1:n_bins
        if counts[i] > 0
            empirical[i] /= counts[i]
            predicted[i] /= counts[i]
        end
    end

    p = plot(title="Counterfactual Data vs Model (θ̂ fixed, π01=0)",
             xlabel="Health Factor h",
             ylabel="Share Deleveraging",
             legend=:topleft,
             size=(800,600),
             linewidth=2.5)

    scatter!(bin_centers, empirical, label="Empirical (counterfactual data)", markersize=6, alpha=0.7)
    plot!(bin_centers, predicted, label=@sprintf("Model (β=%.2f)", β), linewidth=2.5)

    savefig(p, joinpath(FIGURES_PATH, "03_observed_vs_predicted_counterfactual.png"))
    println("  ✓ Saved: 03_observed_vs_predicted_counterfactual.png")
end

function plot_health_timeseries_counterfactual(df::DataFrame)
    avg_health = combine(groupby(df, :t), :h => mean => :avg_h)
    sort!(avg_health, :t)

    p = plot(avg_health.t, avg_health.avg_h,
             xlabel="Time (t)",
             ylabel="Average Health Factor",
             title="Counterfactual Data: Average Health Over Time",
             legend=:bottomright,
             linewidth=2,
             label="Avg health",
             size=(800,600))

    hline!([H_BAR], linestyle=:dash, color=:gray, alpha=0.5, label="H_BAR")

    savefig(p, joinpath(FIGURES_PATH, "04_health_timeseries_counterfactual.png"))
    println("  ✓ Saved: 04_health_timeseries_counterfactual.png")
end

########################################################################
# Main
########################################################################

function main()
    println("="^70)
    println("  STRUCTURAL COUNTERFACTUAL ANALYSIS")
    println("  (θ̂ fixed from baseline, π01 = 0)")
    println("="^70)

    mkpath(FIGURES_PATH)

    # Load counterfactual data for plots only
    println("\nLoading counterfactual data (plots only): $COUNTERFACTUAL_DATA_PATH")
    df_cf = load_data_for_plots(COUNTERFACTUAL_DATA_PATH)
    println("Observations after filtering: $(nrow(df_cf))")

    for β in BETA_VALUES
        println("\n" * "="^70)
        @printf("β = %.4f\n", β)
        println("="^70)

        println("Loading baseline θ̂ from: $BASELINE_EST_PATH")
        θ̂ = load_baseline_theta(β)
        ρ̂, γ̂g = transform_params(θ̂)
        @printf("Baseline θ̂_raw = [%.4f, %.4f]  => ρ̂=%.4f, γ̂_g=%.4f\n", θ̂[1], θ̂[2], ρ̂, γ̂g)

        println("Solving DP under counterfactual π01=0.0 (no crash arrival)...")
        V_cf, v_a_cf, P_cf = solve_counterfactual_policy(θ̂, float(β))

        # Figures
        println("\nCreating figures...")
        plot_policy_function_counterfactual(float(β), P_cf)
        plot_value_function_counterfactual(float(β), V_cf)
        plot_observed_vs_predicted_counterfactual(df_cf, P_cf; β=float(β))
        plot_health_timeseries_counterfactual(df_cf)

        # Optional: report whether an interior 0.5 threshold exists in normal regime
        p_del_normal = P_cf[2, :, 1]
        has_cross = (minimum(p_del_normal) <= 0.5 <= maximum(p_del_normal))
        if has_cross
            # threshold defined as smallest h where P_del >= 0.5
            idx = findfirst(x -> x >= 0.5, p_del_normal)
            h_star = isnothing(idx) ? NaN : H_GRID[idx]
            @printf("Interior threshold exists (normal): approx h* ≈ %.4f\n", h_star)
        else
            println("No interior threshold in normal regime (P_del never crosses 0.5 on [H_MIN,H_MAX]).")
        end

        println("\nAll counterfactual figures saved to: $FIGURES_PATH")
    end

    println("\nDone.")
end

main()
