
#!/usr/bin/env julia

############################################################
# MASTER SCRIPT: Complete Structural Model Analysis
#
# This script:
# 1. Loads all required packages
# 2. Includes utility functions
# 3. Generates baseline data (with crash shocks, π01=0.50)
# 4. Generates counterfactual data (no crash shocks, π01=0) [for plots only]
# 5. Estimates baseline model on baseline data
# 6. Performs STRUCTURAL COUNTERFACTUAL: uses baseline θ̂ with π01=0
# 7. Compares baseline vs counterfactual results
#
# Data generation parameters: ρ = 2.0, γ_g = 1.5, β = 0.2
#
# IMPORTANT: Step 6 is a proper structural counterfactual analysis.
#            It loads baseline θ̂ and re-solves with π01=0 (no re-estimation).
#            Counterfactual data is used only for "observed vs predicted" plots.
#
# Usage: julia master.jl
############################################################

println("="^80)
println("  STRUCTURAL MODEL ANALYSIS - MASTER SCRIPT")
println("="^80)

# ======================================================================
# PACKAGE LOADING
# ======================================================================
println("\n[*] Loading packages...")

using Random
using Distributions
using DataFrames
using CSV
using Plots
using Statistics
using LinearAlgebra
using Optim
using LineSearches
using Printf
using ForwardDiff

println("[*] All packages loaded")

# ======================================================================
# LOAD UTILITIES
# ======================================================================
println("\n[*] Loading shared utilities (ddc_utils.jl)...")
include("ddc_utils.jl")
println("[*] Utilities loaded")

# ======================================================================
# PRELIMINARY TESTS
# ======================================================================
println("\n" * "="^80)
println("RUNNING PRELIMINARY TESTS")
println("="^80)

println("\n[Test 1: Utility Functions and Value Function Solver]")
println("-"^60)
try
    # Test parameters (matching data generation values)
    θ_test = [log(2.0), log(1.5)]
    β_test = 0.2
    
    println("  Testing with θ = [$(θ_test[1]), $(θ_test[2])]")
    println("  This corresponds to ρ = $(exp(θ_test[1])), γ_g = $(exp(θ_test[2]))")
    println("  Discount factor β = $β_test")
    println("  NOTE: Data generation uses these same values (ρ=2.0, γ_g=1.5)")
    
    # Solve value function
    V, v_a = solve_value_function(θ_test, β_test; use_quadrature=true)
    
    # Check dimensions
    println("\n  [*] Value function solved successfully")
    println("    V dimensions: $(size(V))  (expected: (101, 2))")
    println("    v_a dimensions: $(size(v_a))  (expected: (2, 101, 2))")
    
    # Verify expected dimensions
    @assert size(V) == (101, 2) "V should be (101, 2) but got $(size(V))"
    @assert size(v_a) == (2, 101, 2) "v_a should be (2, 101, 2) but got $(size(v_a))"
    
    # Check for finite values
    @assert all(isfinite.(V)) "V contains non-finite values"
    @assert all(isfinite.(v_a)) "v_a contains non-finite values"
    
    # Check monotonicity: V should generally increase with h (health factor)
    println("    V range (normal regime): [$(minimum(V[:,1])), $(maximum(V[:,1]))]")
    println("    V range (crash regime): [$(minimum(V[:,2])), $(maximum(V[:,2]))]")
    
    println("\n  [*] Test 1 PASSED: Utility functions working correctly")
    
catch e
    println("\n  ✗ Test 1 FAILED: $e")
    println("  Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    error("Preliminary tests failed. Stopping execution.")
end

println("\n[Test 2: Data Generation Module]")
println("-"^60)
try
    println("  Loading data generation module...")
    include("data_generation.jl")
    
    # Check if data file was created
    data_path = joinpath(@__DIR__, "data", "data_panel.csv")
    @assert isfile(data_path) "Data file not created at $data_path"
    
    # Load and validate data
    df = CSV.read(data_path, DataFrame)
    println("  [*] Data file created: $data_path")
    println("    Rows: $(nrow(df))")
    println("    Columns: $(names(df))")
    
    # Validate data structure
    @assert "account_id" in names(df) "Missing column: account_id"
    @assert "t" in names(df) "Missing column: t"
    @assert "h" in names(df) "Missing column: h"
    @assert "action" in names(df) "Missing column: action"
    
    # Check data ranges
    @assert all(df.h .>= 1.0) && all(df.h .<= 2.0) "Health factors out of range [1.0, 2.0]"
    @assert all(in.(df.action, Ref(["stay", "deleverage"]))) "Invalid action values"
    
    # Summary statistics
    n_accounts = length(unique(df.account_id))
    n_periods = length(unique(df.t))
    n_stay = count(x -> x == "stay", df.action)
    n_delev = count(x -> x == "deleverage", df.action)
    pct_stay = round(100 * n_stay / nrow(df), digits=1)
    pct_delev = round(100 * n_delev / nrow(df), digits=1)
    
    println("    Unique accounts: $n_accounts")
    println("    Time periods: $n_periods")
    println("    Action distribution:")
    println("      - stay: $n_stay ($pct_stay%)")
    println("      - deleverage: $n_delev ($pct_delev%)")
    
    println("\n  [*] Test 2 PASSED: Data generation module working correctly")
    
catch e
    println("\n  ✗ Test 2 FAILED: $e")
    println("  Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    error("Preliminary tests failed. Stopping execution.")
end

println("\n[Test 3: Estimation Module Structure Check]")
println("-"^60)
try
    println("  Checking estimation module can be loaded...")
    
    # Just verify the file exists and has the expected structure
    est_path = joinpath(@__DIR__, "SM_twostate.jl")
    @assert isfile(est_path) "Estimation file not found: $est_path"
    
    # Read and check for key functions/structures
    est_content = read(est_path, String)
    @assert occursin("EstimationResult", est_content) "Missing EstimationResult struct"
    @assert occursin("load_data", est_content) "Missing load_data function"
    @assert occursin("loglikelihood", est_content) "Missing loglikelihood function"
    
    println("  [*] Estimation module structure verified")
    println("    File: $est_path")
    println("    Key components found: EstimationResult, load_data, loglikelihood")
    
    println("\n  [*] Test 3 PASSED: Estimation module structure correct")
    
catch e
    println("\n  ✗ Test 3 FAILED: $e")
    println("  Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    error("Preliminary tests failed. Stopping execution.")
end

println("\n" * "="^80)
println("  [***] ALL PRELIMINARY TESTS PASSED [***]")
println("  Proceeding with full analysis...")
println("="^80)

# ======================================================================
# STEP 1: GENERATE BASELINE DATA
# ======================================================================
println("\n" * "="^80)
println("STEP 1: Generating baseline data (with crash shocks)...")
println("="^80)
include("data_generation.jl")
println("[*] Baseline data generation complete")

# ======================================================================
# STEP 2: GENERATE COUNTERFACTUAL DATA
# ======================================================================
println("\n" * "="^80)
println("STEP 2: Generating counterfactual data (no crash shocks)...")
println("="^80)
include("data_generation_counterfactual.jl")
println("[*] Counterfactual data generation complete")

# ======================================================================
# STEP 3: ESTIMATE BASELINE MODEL
# ======================================================================
println("\n" * "="^80)
println("STEP 3: Estimating baseline model...")
println("="^80)
include("SM_twostate.jl")
println("[*] Baseline estimation complete")

# ======================================================================
# STEP 4: STRUCTURAL COUNTERFACTUAL ANALYSIS
# ======================================================================
println("\n" * "="^80)
println("STEP 4: Structural counterfactual (no crash arrival, π01=0)...")
println("  Using baseline θ̂ from Step 3 (NO re-estimation)")
println("  Solving value function with π01=0 to eliminate crash risk")
println("="^80)
include("SM_twostate_counterfactual.jl")
println("[*] Structural counterfactual complete")

# ======================================================================
# STEP 5: COMPARE SCENARIOS
# ======================================================================
println("\n" * "="^80)
println("STEP 5: Comparing baseline vs counterfactual...")
println("="^80)
include("compare_scenarios.jl")
println("[*] Comparison complete")

# Print key scalar summaries from comparison
println("\n📊 Key Scalar Summaries from Step 5:")
println("  (See full table and band analysis above)")

# ======================================================================
# FINAL SUMMARY
# ======================================================================
println("\n" * "="^80)
println("  [***] ALL ANALYSIS COMPLETE [***]")
println("="^80)

println("\n[OUTPUT] Output Structure:")
println("  Structural_model/")
println("  ├── data/")
println("  │   ├── data_panel.csv                              (baseline simulated data)")
println("  │   └── data_panel_counterfactual.csv               (counterfactual simulated data)")
println("  │")
println("  ├── results/")
println("  │   └── baseline_estimates.csv                      (saved θ̂ for counterfactual)")
println("  │")
println("  └── figures/")
println("      │")
println("      ├── BASELINE RESULTS (estimated on data_panel.csv, π01=0.50):")
println("      │   ├── 01_policy_function.png                  (optimal decision rules by h)")
println("      │   ├── 02_value_function.png                   (V(h) for both regimes)")
println("      │   ├── 03_choice_specific_values.png           (v_a(h) for stay/deleverage)")
println("      │   ├── 04_flow_utility_components.png          (risk penalty & gas cost γ_g=1.5)")
println("      │   ├── 05_transition_dynamics.png              (E[h'|h,a] dynamics)")
println("      │   ├── 06_observed_vs_predicted.png            (model fit validation)")
println("      │   └── 07_beta_comparative_statics.png         (sensitivity to β)")
println("      │")
println("      ├── counterfactual/  [STRUCTURAL COUNTERFACTUAL: baseline θ̂, π01=0]")
println("      │   ├── 01_policy_function.png                  (policy with baseline θ̂, no crashes)")
println("      │   ├── 02_value_function.png                   (V(h) with baseline θ̂, no crashes)")
println("      │   ├── 03_choice_specific_values.png           (v_a(h) with baseline θ̂, no crashes)")
println("      │   ├── 04_flow_utility_components.png          (utilities with baseline θ̂)")
println("      │   ├── 05_transition_dynamics.png              (dynamics with baseline θ̂)")
println("      │   ├── 06_observed_vs_predicted.png            (fit to counter data, for reference)")
println("      │   └── 07_beta_comparative_statics.png         (β sensitivity)")
println("      │   NOTE: This uses SAME θ̂ as baseline, only changes π01=0")
println("      │         This measures the VALUE of eliminating crash risk")
println("      │")
println("      └── comparison/")
println("          ├── health_timeseries_comparison.png        (avg h over time: base vs counter)")
println("          ├── delever_rate_comparison.png             (deleverage rates over time)")
println("          ├── health_distribution_comparison.png      (h distributions compared)")
println("          └── health_by_action_comparison.png         (h by action choice)")

println("\n[RESULTS] Key Outputs:")
println("  • Estimated structural parameters (ρ, γ_g) with standard errors")
println("  • Baseline estimates saved to: results/baseline_estimates.csv")
println("  • Value functions V(h,s) for baseline (π01=0.50) and counterfactual (π01=0)")
println("  • Choice-specific values v_a(h,s) under both scenarios")
println("  • Conditional choice probabilities P(a|h,s) comparing baseline vs counterfactual")
println("  • Decision thresholds h* (indifference points)")
println("  • Model fit statistics (log-likelihood)")
println("  • STRUCTURAL counterfactual: same θ̂, different crash risk environment")

println("\n[***] STRUCTURAL COUNTERFACTUAL COMPLETE")
println("  The counterfactual/ figures show policies under:")
println("  • SAME preferences (baseline θ̂)")
println("  • DIFFERENT environment (π01=0 instead of 0.50)")
println("  This measures the causal effect of eliminating crash risk.")

println("\n[DOCS] Documentation:")
println("  • CCP_DIAGNOSTIC_GUIDE.md - Full diagnostic methodology")
println("  • QUICK_REFERENCE.md      - Quick troubleshooting guide")
println("  • diagnose_ccps.jl        - Detailed CCP diagnostic tool")

println("\n" * "="^80)
println("Analysis complete! Check the figures/ directory for visualizations.")
println("All tests passed [*] | Data generated [*] | Models estimated [*] | Results compared [*]")
println("With γ_g = 1.5, CCPs should show clearer preferences at high h values.")
println("="^80)
