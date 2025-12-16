#!/usr/bin/env julia

############################################################
# STRUCTURAL POLICY COMPARISON: Baseline vs Counterfactual
#
# Compares policies holding preferences fixed at baseline θ̂.
# Only changes environment: crash arrival π01.
#
# Outputs:
#   - policy functions P(del|h,s) under π01_base vs π01=0
#   - value functions V(h,s) under π01_base vs π01=0
#   - Δ policy and Δ value plots
############################################################

# NOTE: Packages should be loaded in master script
# Requires from master/ddc_utils.jl:
#   H_GRID, H_MIN, H_MAX, H_BAR
#   solve_value_function(θ_raw, β; π01=..., use_quadrature=true)
#   choice_probabilities(v_a)
#   transform_params(θ_raw)
#   interp1(xgrid, ygrid, x)  (if you want empirical overlays)
#   LOG_EPS (optional)

########################################################################
# Paths / settings
########################################################################

@isdefined(RESULTS_PATH) || (const RESULTS_PATH = joinpath(@__DIR__, "results", "baseline_estimates.csv"))

# Optional: data only for descriptive overlays (not used for structural objects)
@isdefined(DATA_BASELINE) || (const DATA_BASELINE = joinpath(@__DIR__, "data", "data_panel.csv"))
@isdefined(DATA_COUNTERFACTUAL) || (const DATA_COUNTERFACTUAL = joinpath(@__DIR__, "data", "data_panel_counterfactual.csv"))

@isdefined(FIGURES_PATH) || (const FIGURES_PATH = joinpath(@__DIR__, "figures", "comparison"))

# Baseline crash arrival probability used in Step 3 environment
@isdefined(PI01_BASE) || (const PI01_BASE = 0.50)

# If baseline_estimates.csv contains β, we’ll use that; else default:
@isdefined(BETA_DEFAULT) || (const BETA_DEFAULT = 0.20)

# Empirical overlay binning
@isdefined(N_BINS) || (const N_BINS = 20)

########################################################################
# Helpers
########################################################################

function load_baseline_estimates(path::String)
    df = CSV.read(path, DataFrame)
    if nrow(df) < 1
        error("No rows found in baseline estimates file: $path")
    end

    # Prefer first row (you only have one β anyway)
    row = df[1, :]

    # Robust parsing: allow either θ_raw columns or ρ/γg
    θ1 = hasproperty(row, :theta1) ? Float64(row.theta1) :
         (hasproperty(row, Symbol("θ1")) ? Float64(row[Symbol("θ1")]) :
          (hasproperty(row, :theta_raw_1) ? Float64(row.theta_raw_1) : NaN))

    θ2 = hasproperty(row, :theta2) ? Float64(row.theta2) :
         (hasproperty(row, Symbol("θ2")) ? Float64(row[Symbol("θ2")]) :
          (hasproperty(row, :theta_raw_2) ? Float64(row.theta_raw_2) : NaN))

    β  = hasproperty(row, :beta) ? Float64(row.beta) :
         (hasproperty(row, Symbol("β")) ? Float64(row[Symbol("β")]) : BETA_DEFAULT)

    # Fallback: if θ_raw columns not found but ρ/γ_g exist, reconstruct θ_raw = log(params)
    if !isfinite(θ1) || !isfinite(θ2)
        ρ  = hasproperty(row, :rho) ? Float64(row.rho) :
             (hasproperty(row, Symbol("ρ")) ? Float64(row[Symbol("ρ")]) : NaN)
        γg = hasproperty(row, :gamma_g) ? Float64(row.gamma_g) :
             (hasproperty(row, :gamma_g_hat) ? Float64(row.gamma_g_hat) :
              (hasproperty(row, Symbol("γ_g")) ? Float64(row[Symbol("γ_g")]) : NaN))

        if !isfinite(ρ) || !isfinite(γg)
            error("Could not find θ_raw or (ρ, γ_g) in $path. Columns = $(names(df))")
        end
        θ1, θ2 = log(ρ), log(γg)
    end

    θ_raw = [θ1, θ2]
    return θ_raw, β
end

function solve_policy_value(θ_raw::Vector{Float64}, β::Float64; π01::Float64)
    V, v_a = solve_value_function(θ_raw, β; π01=π01, use_quadrature=true)
    P = choice_probabilities(v_a)  # [action, h_index, regime_index]
    return V, P
end

# Descriptive: empirical deleveraging share by h-bin
function empirical_delever_by_h(df::DataFrame)
    # Ensure action is string labels
    if eltype(df.action) <: Integer
        # map 1->stay,2->deleverage if needed
        act = df.action
        df = copy(df)
        df.action = map(a -> a == 2 ? "deleverage" : "stay", act)
    end

    bins = range(H_MIN, H_MAX, length=N_BINS+1)
    centers = [(bins[i] + bins[i+1])/2 for i in 1:N_BINS]
    shares = fill(NaN, N_BINS)
    counts = zeros(Int, N_BINS)

    for r in eachrow(df)
        h = Float64(r.h)
        b = searchsortedlast(bins, h)
        b = clamp(b, 1, N_BINS)
        counts[b] += 1
    end

    delever_counts = zeros(Int, N_BINS)
    for r in eachrow(df)
        h = Float64(r.h)
        b = searchsortedlast(bins, h)
        b = clamp(b, 1, N_BINS)
        delever_counts[b] += (r.action == "deleverage" ? 1 : 0)
    end

    for i in 1:N_BINS
        if counts[i] > 0
            shares[i] = delever_counts[i] / counts[i]
        end
    end

    return centers, shares
end

########################################################################
# Plotting
########################################################################

function plot_structural_policy_comparison(P_base, P_cf, β, π01_base)
    mkpath(FIGURES_PATH)

    # P_del grids
    Pdel_base_s0 = P_base[2, :, 1]
    Pdel_base_s1 = P_base[2, :, 2]
    Pdel_cf_s0   = P_cf[2, :, 1]
    Pdel_cf_s1   = P_cf[2, :, 2]

    p1 = plot(title = @sprintf("Structural Policy: P(delever|h) (β=%.2f)", β),
              xlabel = "Health factor h", ylabel = "P(delever)",
              legend = :topright, size = (1000, 650), linewidth = 2.5)

    plot!(H_GRID, Pdel_base_s0, label = @sprintf("Baseline env (π01=%.2f), s=0", π01_base), linestyle=:solid)
    plot!(H_GRID, Pdel_cf_s0,   label = "Counterfactual env (π01=0), s=0", linestyle=:dash)

    plot!(H_GRID, Pdel_base_s1, label = @sprintf("Baseline env (π01=%.2f), s=1", π01_base), linestyle=:solid, alpha=0.6)
    plot!(H_GRID, Pdel_cf_s1,   label = "Counterfactual env (π01=0), s=1", linestyle=:dash, alpha=0.6)

    hline!([0.5], linestyle=:dot, alpha=0.5, label="0.5")
    vline!([H_BAR], linestyle=:dot, alpha=0.5, label="H_BAR")

    savefig(p1, joinpath(FIGURES_PATH, "policy_structural_comparison.png"))
    println("  ✓ Saved: policy_structural_comparison.png")

    # Δ policy
    ΔP_s0 = Pdel_cf_s0 .- Pdel_base_s0
    ΔP_s1 = Pdel_cf_s1 .- Pdel_base_s1

    p2 = plot(title = @sprintf("Δ Policy: P_cf - P_base (β=%.2f)", β),
              xlabel = "Health factor h", ylabel = "ΔP(delever)",
              legend = :topright, size = (1000, 650), linewidth = 2.5)

    plot!(H_GRID, ΔP_s0, label="ΔP, s=0", linestyle=:solid)
    plot!(H_GRID, ΔP_s1, label="ΔP, s=1", linestyle=:dash)

    hline!([0.0], linestyle=:dot, alpha=0.5, label="0")
    vline!([H_BAR], linestyle=:dot, alpha=0.5, label="H_BAR")

    savefig(p2, joinpath(FIGURES_PATH, "policy_delta_structural.png"))
    println("  ✓ Saved: policy_delta_structural.png")
end

function plot_structural_value_comparison(V_base, V_cf, β, π01_base)
    mkpath(FIGURES_PATH)

    Vb_s0 = V_base[:, 1]; Vb_s1 = V_base[:, 2]
    Vc_s0 = V_cf[:, 1];   Vc_s1 = V_cf[:, 2]

    p1 = plot(title = @sprintf("Structural Value: V(h,s) (β=%.2f)", β),
              xlabel = "Health factor h", ylabel = "V",
              legend = :bottomright, size = (1000, 650), linewidth = 2.5)

    plot!(H_GRID, Vb_s0, label=@sprintf("Baseline env (π01=%.2f), s=0", π01_base), linestyle=:solid)
    plot!(H_GRID, Vc_s0, label="Counterfactual env (π01=0), s=0", linestyle=:dash)

    plot!(H_GRID, Vb_s1, label=@sprintf("Baseline env (π01=%.2f), s=1", π01_base), linestyle=:solid, alpha=0.6)
    plot!(H_GRID, Vc_s1, label="Counterfactual env (π01=0), s=1", linestyle=:dash, alpha=0.6)

    vline!([H_BAR], linestyle=:dot, alpha=0.5, label="H_BAR")

    savefig(p1, joinpath(FIGURES_PATH, "value_structural_comparison.png"))
    println("  ✓ Saved: value_structural_comparison.png")

    # Δ value
    p2 = plot(title = @sprintf("Δ Value: V_cf - V_base (β=%.2f)", β),
              xlabel = "Health factor h", ylabel = "ΔV",
              legend = :topright, size = (1000, 650), linewidth = 2.5)

    plot!(H_GRID, (Vc_s0 .- Vb_s0), label="ΔV, s=0", linestyle=:solid)
    plot!(H_GRID, (Vc_s1 .- Vb_s1), label="ΔV, s=1", linestyle=:dash)

    hline!([0.0], linestyle=:dot, alpha=0.5, label="0")
    vline!([H_BAR], linestyle=:dot, alpha=0.5, label="H_BAR")

    savefig(p2, joinpath(FIGURES_PATH, "value_delta_structural.png"))
    println("  ✓ Saved: value_delta_structural.png")
end

function plot_optional_empirical_overlays(P_base, P_cf)
    # purely descriptive: empirical delever share by h in each dataset
    if !isfile(DATA_BASELINE) || !isfile(DATA_COUNTERFACTUAL)
        println("  (Skipping empirical overlays: data files not found.)")
        return
    end
    df_base = CSV.read(DATA_BASELINE, DataFrame)
    df_cf   = CSV.read(DATA_COUNTERFACTUAL, DataFrame)

    centers_base, share_base = empirical_delever_by_h(df_base)
    centers_cf,   share_cf   = empirical_delever_by_h(df_cf)

    p = plot(title="Descriptive: Empirical delever share by h (datasets)",
             xlabel="Health factor h", ylabel="Empirical share(delever)",
             legend=:topright, size=(1000,650), linewidth=2.5)

    scatter!(centers_base, share_base, label="Baseline data", alpha=0.7, markersize=6)
    scatter!(centers_cf, share_cf, label="Counterfactual data", alpha=0.7, markersize=6)

    vline!([H_BAR], linestyle=:dot, alpha=0.5, label="H_BAR")

    savefig(p, joinpath(FIGURES_PATH, "empirical_delever_by_h.png"))
    println("  ✓ Saved: empirical_delever_by_h.png")
end

########################################################################
# Numerical reporting
########################################################################

function report_policy_value_differences(P_base, P_cf, V_base, V_cf, π01_base)
    println("\n" * "="^80)
    println("  NUMERICAL SUMMARY: Policy and Value Differences")
    println("="^80)
    
    # Extract deleveraging probabilities: P[2, :, :] corresponds to action 2 (deleverage)
    Pdel_base_s0 = P_base[2, :, 1]
    Pdel_base_s1 = P_base[2, :, 2]
    Pdel_cf_s0   = P_cf[2, :, 1]
    Pdel_cf_s1   = P_cf[2, :, 2]
    
    # Compute differences
    # ΔP(h) = P_π01=0.10(delever|h) - P_π01=0(delever|h) = P_base - P_cf
    ΔP_s0 = Pdel_base_s0 .- Pdel_cf_s0
    ΔP_s1 = Pdel_base_s1 .- Pdel_cf_s1
    
    # ΔV(h) = V_π01=0(h) - V_π01=0.10(h) = V_cf - V_base
    ΔV_s0 = V_cf[:, 1] .- V_base[:, 1]
    ΔV_s1 = V_cf[:, 2] .- V_base[:, 2]
    
    # Report policy differences
    println("\n--- Policy Differences: ΔP(h) = P(π01=$(π01_base)) - P(π01=0) ---")
    println("\nRegime s=0 (Normal):")
    @printf("  Mean ΔP(h):    %9.6f\n", mean(ΔP_s0))
    @printf("  Median ΔP(h):  %9.6f\n", median(ΔP_s0))
    @printf("  Min ΔP(h):     %9.6f  (at h=%.4f)\n", minimum(ΔP_s0), H_GRID[argmin(ΔP_s0)])
    @printf("  Max ΔP(h):     %9.6f  (at h=%.4f)\n", maximum(ΔP_s0), H_GRID[argmax(ΔP_s0)])
    @printf("  Std ΔP(h):     %9.6f\n", std(ΔP_s0))
    
    println("\nRegime s=1 (Crisis):")
    @printf("  Mean ΔP(h):    %9.6f\n", mean(ΔP_s1))
    @printf("  Median ΔP(h):  %9.6f\n", median(ΔP_s1))
    @printf("  Min ΔP(h):     %9.6f  (at h=%.4f)\n", minimum(ΔP_s1), H_GRID[argmin(ΔP_s1)])
    @printf("  Max ΔP(h):     %9.6f  (at h=%.4f)\n", maximum(ΔP_s1), H_GRID[argmax(ΔP_s1)])
    @printf("  Std ΔP(h):     %9.6f\n", std(ΔP_s1))
    
    # Report value differences
    println("\n--- Value Differences: ΔV(h) = V(π01=0) - V(π01=$(π01_base)) ---")
    println("\nRegime s=0 (Normal):")
    @printf("  Mean ΔV(h):    %9.6f\n", mean(ΔV_s0))
    @printf("  Median ΔV(h):  %9.6f\n", median(ΔV_s0))
    @printf("  Min ΔV(h):     %9.6f  (at h=%.4f)\n", minimum(ΔV_s0), H_GRID[argmin(ΔV_s0)])
    @printf("  Max ΔV(h):     %9.6f  (at h=%.4f)\n", maximum(ΔV_s0), H_GRID[argmax(ΔV_s0)])
    @printf("  Std ΔV(h):     %9.6f\n", std(ΔV_s0))
    
    println("\nRegime s=1 (Crisis):")
    @printf("  Mean ΔV(h):    %9.6f\n", mean(ΔV_s1))
    @printf("  Median ΔV(h):  %9.6f\n", median(ΔV_s1))
    @printf("  Min ΔV(h):     %9.6f  (at h=%.4f)\n", minimum(ΔV_s1), H_GRID[argmin(ΔV_s1)])
    @printf("  Max ΔV(h):     %9.6f  (at h=%.4f)\n", maximum(ΔV_s1), H_GRID[argmax(ΔV_s1)])
    @printf("  Std ΔV(h):     %9.6f\n", std(ΔV_s1))
    
    # Table at selected h values
    println("\n--- Values at Selected Health Levels ---")
    h_selected = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
    
    println("\nΔP(h) and ΔV(h) at selected h:")
    println("─"^80)
    @printf("%-8s | %-12s | %-12s | %-12s | %-12s\n", 
            "h", "ΔP(h) s=0", "ΔP(h) s=1", "ΔV(h) s=0", "ΔV(h) s=1")
    println("─"^80)
    
    for h_val in h_selected
        # Interpolate values at h_val
        ΔP_s0_h = interp1(H_GRID, ΔP_s0, h_val)
        ΔP_s1_h = interp1(H_GRID, ΔP_s1, h_val)
        ΔV_s0_h = interp1(H_GRID, ΔV_s0, h_val)
        ΔV_s1_h = interp1(H_GRID, ΔV_s1, h_val)
        
        @printf("%-8.2f | %12.6f | %12.6f | %12.6f | %12.6f\n",
                h_val, ΔP_s0_h, ΔP_s1_h, ΔV_s0_h, ΔV_s1_h)
    end
    println("─"^80)
    
    # Concentration near liquidation analysis
    println("\n--- Concentration Near Liquidation ---")
    
    # Low-health band: h ≤ 1.2
    low_health_idx = H_GRID .<= 1.2
    # High-health band: h ≥ 1.8
    high_health_idx = H_GRID .>= 1.8
    
    if sum(low_health_idx) > 0 && sum(high_health_idx) > 0
        println("\nAverage ΔP(h) by health band:")
        println("  Low health (h ≤ 1.2):")
        @printf("    s=0: %9.6f\n", mean(ΔP_s0[low_health_idx]))
        @printf("    s=1: %9.6f\n", mean(ΔP_s1[low_health_idx]))
        
        println("  High health (h ≥ 1.8):")
        @printf("    s=0: %9.6f\n", mean(ΔP_s0[high_health_idx]))
        @printf("    s=1: %9.6f\n", mean(ΔP_s1[high_health_idx]))
        
        println("\nAverage ΔV(h) by health band:")
        println("  Low health (h ≤ 1.2):")
        @printf("    s=0: %9.6f\n", mean(ΔV_s0[low_health_idx]))
        @printf("    s=1: %9.6f\n", mean(ΔV_s1[low_health_idx]))
        
        println("  High health (h ≥ 1.8):")
        @printf("    s=0: %9.6f\n", mean(ΔV_s0[high_health_idx]))
        @printf("    s=1: %9.6f\n", mean(ΔV_s1[high_health_idx]))
    else
        println("  (Health bands not found in grid)")
    end
    
    println("\n" * "="^80)
end

########################################################################
# Main
########################################################################

function main()
    println("="^80)
    println("  STRUCTURAL COMPARISON: Policies under π01_base vs π01=0 (θ̂ fixed)")
    println("="^80)

    println("Loading baseline estimates: $RESULTS_PATH")
    θ_raw_hat, β = load_baseline_estimates(RESULTS_PATH)
    ρ_hat, γg_hat = transform_params(θ_raw_hat)
    @printf("Using θ̂_raw = [%.4f, %.4f]  => ρ̂=%.4f, γ̂_g=%.4f, β=%.2f\n",
            θ_raw_hat[1], θ_raw_hat[2], ρ_hat, γg_hat, β)

    println("\nSolving baseline environment DP (π01 = $(PI01_BASE)) ...")
    V_base, P_base = solve_policy_value(θ_raw_hat, β; π01=PI01_BASE)

    println("Solving counterfactual environment DP (π01 = 0.0) ...")
    V_cf, P_cf = solve_policy_value(θ_raw_hat, β; π01=0.0)

    # Print numerical summary of differences
    report_policy_value_differences(P_base, P_cf, V_base, V_cf, PI01_BASE)

    println("\nCreating structural comparison figures...")
    plot_structural_policy_comparison(P_base, P_cf, β, PI01_BASE)
    plot_structural_value_comparison(V_base, V_cf, β, PI01_BASE)

    println("\n(Optional) Creating descriptive empirical overlays...")
    plot_optional_empirical_overlays(P_base, P_cf)

    println("\n✓ All figures saved to: $FIGURES_PATH")
    println("="^80)
end

main()
