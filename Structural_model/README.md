# Structural Model - Master Script Structure

## Credits
Created by Logan Goin with the help of Claude and ChatGPT.
##
As much as I dislike how AI comments code, I will admit that it did a pretty good job at taking my sloppy comments/code and making it intuitive and easy to follow. 
All of this code was either written or maticulously checked by myself. Any time the code was updated or written by AI I took the code and ran it through the other AI to have it looked over as well as checked it myself (if ChatGPT wrote the code, I checked it as well as Claude and vice-versa). 

## Quick Start

Run the complete analysis with:

```julia
cd Structural_model
julia master.jl
```

This single command will:
1. Load all packages
2. Generate baseline data (with crash shocks)
3. Generate counterfactual data (no crash shocks)
4. Estimate both models
5. Create comparison figures

## File Structure

```
Structural_model/
├── master.jl                           # 🎯 RUN THIS - Main entry point
│
├── ddc_utils.jl                        # Shared utilities (grid, VFI, CCPs)
│
├── data_generation.jl                  # Baseline data generator
├── data_generation_counterfactual.jl   # Counterfactual data generator
│
├── SM_twostate.jl                      # Baseline model estimation
├── SM_twostate_counterfactual.jl       # Counterfactual estimation
│
├── compare_scenarios.jl                # Comparison analysis
│
├── data/                               # Generated datasets (auto-created)
│   ├── data_panel.csv
│   └── data_panel_counterfactual.csv
│
└── figures/                            # All output figures (auto-created)
    ├── *.png                           # Baseline results
    ├── counterfactual/*.png            # Counterfactual results
    └── comparison/*.png                # Side-by-side comparisons
```

## Key Features

✅ **Single entry point**: `master.jl` loads all packages once
✅ **Relative paths**: All paths are relative to `Structural_model/`
✅ **Auto-creation**: `data/` and `figures/` directories created automatically
✅ **No duplication**: Packages loaded only in `master.jl`, not in individual files
✅ **Modular**: Each file focused on one task

## Package Requirements

All packages are loaded in `master.jl`:
- Random, Distributions
- DataFrames, CSV
- Plots, Statistics
- LinearAlgebra
- Optim, LineSearches
- Printf, ForwardDiff

## Individual File Usage

⚠️ **Important**: Do NOT run individual files directly (e.g., `julia data_generation.jl`)

Individual files do not load packages and will error if run standalone.

If you need to run specific steps, use `master.jl` and comment out unwanted sections.

## Output

After running `master.jl`, you'll have:

**Data** (in `data/`):
- `data_panel.csv` - Baseline with crash shocks at t=25, 50, 75
- `data_panel_counterfactual.csv` - Pure random walk

**Figures** (in `figures/`):
- Baseline: Policy functions, value functions, fit diagnostics
- Counterfactual: Same set of figures for no-shock scenario
- Comparison: Side-by-side health trajectories, action rates, distributions

## Troubleshooting

**Problem**: "Package X not found"
- **Solution**: Install missing packages: `using Pkg; Pkg.add("X")`

**Problem**: "Cannot find data file"
- **Solution**: Make sure you run `julia master.jl` from the `Structural_model/` directory

**Problem**: Individual file gives errors
- **Solution**: Don't run individual files. Use `master.jl` instead.

## Technical Notes

- All files use relative paths via `@__DIR__`
- Directories (`data/`, `figures/`) created with `mkpath()` if missing
- Same estimation method (LBFGS) for both scenarios
- Same shared utilities (`ddc_utils.jl`) for consistency
- Parameters: β=0.2, grid h∈[1.0, 2.0] with 101 points

## Citation

Dynamic Discrete Choice Model for DeFi Deleveraging
Structural Estimation with Counterfactual Analysis
Fall 2025
