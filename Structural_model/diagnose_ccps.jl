#!/usr/bin/env julia

############################################################
# DIAGNOSTIC SCRIPT: Analyze CCP Flatness at High h
#
# This script diagnoses why P(delever) is flat (~0.47)
# at high health values by computing:
# - Flow utilities
# - Expected continuation values
# - Choice-specific values
# - CCPs with explicit σ_ε
#
# Usage: julia diagnose_ccps.jl
############################################################

println("="^80)
println("  CCP DIAGNOSTIC ANALYSIS")
println("="^80)

# ======================================================================
# PACKAGE LOADING
# ======================================================================
println("\n📦 Loading packages...")

using Random
using Distributions
using DataFrames
using CSV
using Statistics
using Printf

println("✓ All packages loaded")

# ======================================================================
# LOAD UTILITIES
# ======================================================================
println("\n🔧 Loading shared utilities (ddc_utils.jl)...")
include("ddc_utils.jl")
println("✓ Utilities loaded")

# ======================================================================
# SCENARIO 1: Estimated Parameters (if available)
# ======================================================================
println("\n" * "▶"^40)
println("SCENARIO 1: Using typical estimated parameters")
println("▶"^40)

# Typical estimated values from your model
# Adjust these based on your actual estimation output
θ_estimated = [log(2.0), log(1.5)]  # ρ ≈ 2.0, γ_g ≈ 1.5
β_estimated = 0.2

println("\nRunning diagnostic with:")
println("  ρ = $(exp(θ_estimated[1]))")
println("  γ_g = $(exp(θ_estimated[2]))")
println("  β = $β_estimated")

V1, v_a1, P1 = diagnose_ccp_flatness(
    θ_estimated, β_estimated;
    h_values = [1.5, 1.8, 2.0],
    regimes = [0, 1],
    π01 = 0.10,
    π10 = 0.30,
    use_quadrature = true,
    σ_ε = 1.0  # Current implicit value
)

# ======================================================================
# SCENARIO 2: Counterfactual with No Crash Risk
# ======================================================================
println("\n" * "▶"^40)
println("SCENARIO 2: Counterfactual (no crash shocks, π01=0)")
println("▶"^40)

V2, v_a2, P2 = diagnose_ccp_flatness(
    θ_estimated, β_estimated;
    h_values = [1.5, 1.8, 2.0],
    regimes = [0],  # Only normal regime matters
    π01 = 0.0,  # No crashes
    π10 = 0.30,
    use_quadrature = true,
    σ_ε = 1.0
)

# ======================================================================
# SCENARIO 3: What if σ_ε were smaller?
# ======================================================================
println("\n" * "▶"^40)
println("SCENARIO 3: Hypothetical with σ_ε = 0.1 (more deterministic)")
println("▶"^40)

println("\nNOTE: This is a HYPOTHETICAL scenario showing what would happen")
println("if we had a smaller taste shock scale. This would require:")
println("  (a) Re-estimating with σ_ε as a free parameter, OR")
println("  (b) Normalizing β instead of σ_ε")
println()

# For this scenario, we need to manually compute CCPs with different σ_ε
println("Computing value differences at h = 2.0, s = 0:")

h_test = 2.0
s_test = 0
ρ, γ_g = transform_params(θ_estimated)

# Use the value function from scenario 1
u_stay = flow_utility(h_test, 1, ρ, γ_g)
u_del = flow_utility(h_test, 2, ρ, γ_g)
EV_stay = expected_V_action(h_test, s_test, V1, 1; π01=0.10, π10=0.30, use_quadrature=true)
EV_del = expected_V_action(h_test, s_test, V1, 2; π01=0.10, π10=0.30, use_quadrature=true)

v_stay = u_stay + β_estimated * EV_stay
v_del = u_del + β_estimated * EV_del
Δv = v_stay - v_del

println("\nValue gap: Δv = $(@sprintf("%.6f", Δv))")
println("\nImplied P(del) under different σ_ε:")

for σ in [1.0, 0.5, 0.1, 0.01]
    P_del = 1.0 / (1.0 + exp(Δv / σ))
    println("  σ_ε = $σ  →  P(del) = $(@sprintf("%.4f", P_del))")
end

println("\n→ Smaller σ_ε makes choices more deterministic")
if Δv > 0
    println("→ With current Δv ≈ $(@sprintf("%.4f", Δv)), stay is preferred")
    println("→ Larger γ_g = 1.5 should give clearer preference for stay at high h")
else
    println("→ With current Δv ≈ $(@sprintf("%.4f", Δv)), delever may be preferred")
end

# ======================================================================
# SCENARIO 4: What if γ_g were larger?
# ======================================================================
println("\n" * "▶"^40)
println("SCENARIO 4: Effect of γ_g = 1.5 (data generation value)")
println("▶"^40)

θ_large_gamma = [log(2.0), log(1.5)]  # ρ = 2.0, γ_g = 1.5

V4, v_a4, P4 = diagnose_ccp_flatness(
    θ_large_gamma, β_estimated;
    h_values = [2.0],
    regimes = [0],
    π01 = 0.10,
    π10 = 0.30,
    use_quadrature = true,
    σ_ε = 1.0
)

# ======================================================================
# VERIFICATION: Check actual transitions
# ======================================================================
println("\n" * "▶"^40)
println("VERIFICATION: Check transition mechanics")
println("▶"^40)

println("\nTesting transitions at h = 2.0 (capped at H_MAX):")
println("\nStay action (stochastic):")
println("  h' = clamp(h * η, H_MIN, H_MAX)")
println("  For h = 2.0:")

Random.seed!(123)
for i in 1:5
    η = rand(BASE_SHOCK_DIST)
    h_next = clamp(2.0 * η, H_MIN, H_MAX)
    println("    η = $(@sprintf("%.4f", η)) → h' = $(@sprintf("%.4f", h_next))")
end

println("\n  → Stay has downside risk even at h = H_MAX")
println("  → Upside is capped at H_MAX = 2.0")

println("\nDelever action (deterministic):")
println("  h' = max(H_BAR, h)")
println("  For h = 2.0:")
println("    h' = max(1.5, 2.0) = 2.0")
println("\n  → Delever provides insurance (no downside risk)")

# ======================================================================
# FINAL RECOMMENDATIONS
# ======================================================================
println("\n" * "="^80)
println("FINAL RECOMMENDATIONS")
println("="^80)

println("\nBased on the diagnostic:")
println()
println("1. IS THIS A BUG?")
println("   Check the value differences Δv at h ≥ H_BAR:")
println("   • If Δv is small (< 0.5) and correct sign, NO BUG")
println("   • If Δv has wrong sign or is zero when shouldn't be, YES BUG")
println()
println("2. IF NOT A BUG (likely case):")
println("   The flat CCP is an economic implication of:")
println("   • Small flow utility difference (γ_g ≈ 0.134 is tiny)")
println("   • Insurance value of deleveraging (deterministic vs. stochastic)")
println("   • Capped upside at h = 2.0")
println("   • EV1 scale σ_ε = 1.0 (not estimated)")
println()
println("3. TO OBTAIN P(del | h ≥ H_BAR) ≈ 0:")
println("   Choose ONE modification:")
println()
println("   Option A: Estimate σ_ε as a free parameter")
println("     → Add σ_ε to θ: [log(ρ), log(γ_g), log(σ_ε)]")
println("     → Normalize β or another parameter instead")
println("     → Re-estimate the model")
println()
println("   Option B: Increase γ_g scale")
println("     → Use different units (e.g., γ_g in basis points)")
println("     → Or force γ_g ≥ 0.5 in bounds")
println()
println("   Option C: Remove taste shocks (deterministic model)")
println("     → Change V = γ_E + log Σ exp(v_a) to V = max_a v_a")
println("     → CCPs become 0/1 (no smooth probabilities)")
println()
println("   Option D: Change stay transition at high h")
println("     → Make stay deterministic when h ≥ H_BAR")
println("     → This changes the economics (removes insurance value)")
println()
println("4. FOR COUNTERFACTUAL ANALYSIS:")
println("   • DO NOT re-estimate with counterfactual data")
println("   • Use baseline θ̂ (estimated from data_panel.csv)")
println("   • Only change π01 = 0 in value function iteration")
println("   • This is a STRUCTURAL counterfactual (policy change)")
println()
println("="^80)
println("Diagnostic complete! Check output above for details.")
println("="^80)
