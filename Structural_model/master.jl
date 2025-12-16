#!/usr/bin/env julia

############################################################
# MASTER SCRIPT: Complete Structural Model Analysis
#
# This script:
# 1. Loads all required packages
# 2. Includes utility functions
# 3. Generates baseline data (with crash shocks)
# 4. Generates counterfactual data (no crash shocks)
# 5. Estimates baseline model
# 6. Estimates counterfactual model
# 7. Compares results
#
# Usage: julia master.jl
############################################################

println("="^80)
println("  STRUCTURAL MODEL ANALYSIS - MASTER SCRIPT")
println("="^80)

# ======================================================================
# PACKAGE LOADING
# ======================================================================
println("\n📦 Loading packages...")

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

println("✓ All packages loaded")

# ======================================================================
# LOAD UTILITIES
# ======================================================================
println("\n🔧 Loading shared utilities (ddc_utils.jl)...")
include("ddc_utils.jl")
println("✓ Utilities loaded")

# ======================================================================
# STEP 1: GENERATE BASELINE DATA
# ======================================================================
println("\n" * "▶"^40)
println("STEP 1: Generating baseline data (with crash shocks)...")
println("▶"^40)
include("data_generation.jl")
println("✓ Baseline data generation complete")

# ======================================================================
# STEP 2: GENERATE COUNTERFACTUAL DATA
# ======================================================================
println("\n" * "▶"^40)
println("STEP 2: Generating counterfactual data (no crash shocks)...")
println("▶"^40)
include("data_generation_counterfactual.jl")
println("✓ Counterfactual data generation complete")

# ======================================================================
# STEP 3: ESTIMATE BASELINE MODEL
# ======================================================================
println("\n" * "▶"^40)
println("STEP 3: Estimating baseline model...")
println("▶"^40)
include("SM_twostate.jl")
println("✓ Baseline estimation complete")

# ======================================================================
# STEP 4: ESTIMATE COUNTERFACTUAL MODEL
# ======================================================================
println("\n" * "▶"^40)
println("STEP 4: Estimating counterfactual model...")
println("▶"^40)
include("SM_twostate_counterfactual.jl")
println("✓ Counterfactual estimation complete")

# ======================================================================
# STEP 5: COMPARE SCENARIOS
# ======================================================================
println("\n" * "▶"^40)
println("STEP 5: Comparing baseline vs counterfactual...")
println("▶"^40)
include("compare_scenarios.jl")
println("✓ Comparison complete")

# ======================================================================
# FINAL SUMMARY
# ======================================================================
println("\n" * "="^80)
println("  ✓✓✓ ALL ANALYSIS COMPLETE ✓✓✓")
println("="^80)

println("\n📁 Output Structure:")
println("  Structural_model/")
println("  ├── data/")
println("  │   ├── data_panel.csv                    (baseline)")
println("  │   └── data_panel_counterfactual.csv     (counterfactual)")
println("  │")
println("  └── figures/")
println("      ├── *.png                             (baseline results)")
println("      ├── counterfactual/")
println("      │   └── *.png                         (counterfactual results)")
println("      └── comparison/")
println("          └── *.png                         (side-by-side comparisons)")

println("\n" * "="^80)
println("Analysis complete! Check the figures/ directory for visualizations.")
println("="^80)
