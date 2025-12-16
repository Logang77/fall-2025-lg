#!/usr/bin/env julia

############################################################
# COMPARISON SCRIPT: Baseline vs Counterfactual
#
# Loads results from both scenarios and creates
# side-by-side comparison figures.
############################################################

# NOTE: Packages should be loaded in master script

########################################################################
# Paths
########################################################################

const DATA_BASELINE = joinpath(@__DIR__, "data", "data_panel.csv")
const DATA_COUNTERFACTUAL = joinpath(@__DIR__, "data", "data_panel_counterfactual.csv")
const FIGURES_PATH = joinpath(@__DIR__, "figures", "comparison")

########################################################################
# Load and prepare data
########################################################################

function load_comparison_data()
    println("Loading baseline data...")
    df_baseline = CSV.read(DATA_BASELINE, DataFrame)
    
    println("Loading counterfactual data...")
    df_counterfactual = CSV.read(DATA_COUNTERFACTUAL, DataFrame)
    
    return df_baseline, df_counterfactual
end

########################################################################
# Comparison plots
########################################################################

function plot_health_comparison(df_base::DataFrame, df_counter::DataFrame)
    # Average health over time
    avg_base = combine(groupby(df_base, :t), :h => mean => :avg_h)
    avg_counter = combine(groupby(df_counter, :t), :h => mean => :avg_h)
    sort!(avg_base, :t)
    sort!(avg_counter, :t)
    
    p = plot(title = "Average Health: Baseline vs Counterfactual",
             xlabel = "Time (t)",
             ylabel = "Average Health Factor",
             legend = :bottomright,
             size = (1000, 600),
             linewidth = 2.5)
    
    plot!(avg_base.t, avg_base.avg_h, 
          label = "Baseline (with crash shocks)",
          color = :blue,
          linewidth = 2.5)
    
    plot!(avg_counter.t, avg_counter.avg_h,
          label = "Counterfactual (pure random walk)",
          color = :green,
          linewidth = 2.5)
    
    # Mark crash shock days in baseline
    crash_days = [25, 50, 75]
    for day in crash_days
        vline!([day], linestyle = :dash, color = :red, alpha = 0.3, label = "")
    end
    
    hline!([H_BAR], linestyle = :dot, color = :gray, alpha = 0.5, label = "H_BAR")
    
    mkpath(FIGURES_PATH)
    savefig(p, joinpath(FIGURES_PATH, "health_timeseries_comparison.png"))
    println("  ✓ Saved: health_timeseries_comparison.png")
end

function plot_action_comparison(df_base::DataFrame, df_counter::DataFrame)
    # Deleveraging rate over time
    delever_base = combine(groupby(df_base, :t), 
                          :action => (x -> mean(x .== "deleverage")) => :delever_rate)
    delever_counter = combine(groupby(df_counter, :t),
                             :action => (x -> mean(x .== "deleverage")) => :delever_rate)
    sort!(delever_base, :t)
    sort!(delever_counter, :t)
    
    # Filter to only t = 0 to t = 14
    delever_base = filter(row -> row.t >= 0 && row.t <= 14, delever_base)
    delever_counter = filter(row -> row.t >= 0 && row.t <= 14, delever_counter)
    
    p = plot(title = "Deleveraging Rate: Baseline vs Counterfactual",
             xlabel = "Time (t)",
             ylabel = "Share Deleveraging",
             legend = :topright,
             size = (1000, 600),
             linewidth = 2.5,
             xlims = (0, 14))
    
    plot!(delever_base.t, delever_base.delever_rate,
          label = "Baseline (with crash shocks)",
          color = :blue,
          linewidth = 2.5,
          alpha = 0.7)
    
    plot!(delever_counter.t, delever_counter.delever_rate,
          label = "Counterfactual (pure random walk)",
          color = :green,
          linewidth = 2.5,
          alpha = 0.7)
    
    savefig(p, joinpath(FIGURES_PATH, "delever_rate_comparison.png"))
    println("  ✓ Saved: delever_rate_comparison.png")
end

function plot_distribution_comparison(df_base::DataFrame, df_counter::DataFrame)
    # Health distribution comparison
    p = plot(title = "Health Distribution: Baseline vs Counterfactual",
             xlabel = "Health Factor h",
             ylabel = "Density",
             legend = :topright,
             size = (1000, 600),
             linewidth = 2.5)
    
    histogram!(df_base.h, bins=30, alpha=0.5, normalize=:pdf,
               label = "Baseline (with crash shocks)",
               color = :blue)
    
    histogram!(df_counter.h, bins=30, alpha=0.5, normalize=:pdf,
               label = "Counterfactual (pure random walk)",
               color = :green)
    
    vline!([H_BAR], linestyle = :dash, color = :gray, linewidth = 2, label = "H_BAR")
    
    savefig(p, joinpath(FIGURES_PATH, "health_distribution_comparison.png"))
    println("  ✓ Saved: health_distribution_comparison.png")
end

function plot_health_by_action(df_base::DataFrame, df_counter::DataFrame)
    # Health levels when actions are taken
    p1 = plot(title = "Baseline: Health at Action",
              xlabel = "Health Factor h",
              ylabel = "Density",
              legend = :topright,
              size = (500, 500))
    
    histogram!(df_base[df_base.action .== "stay", :h], 
               bins=30, alpha=0.5, normalize=:pdf,
               label = "Stay", color = :blue)
    histogram!(df_base[df_base.action .== "deleverage", :h],
               bins=30, alpha=0.5, normalize=:pdf,
               label = "Deleverage", color = :red)
    
    p2 = plot(title = "Counterfactual: Health at Action",
              xlabel = "Health Factor h",
              ylabel = "Density",
              legend = :topright,
              size = (500, 500))
    
    histogram!(df_counter[df_counter.action .== "stay", :h],
               bins=30, alpha=0.5, normalize=:pdf,
               label = "Stay", color = :blue)
    histogram!(df_counter[df_counter.action .== "deleverage", :h],
               bins=30, alpha=0.5, normalize=:pdf,
               label = "Deleverage", color = :red)
    
    combined = plot(p1, p2, layout = (1, 2), size = (1200, 500))
    savefig(combined, joinpath(FIGURES_PATH, "health_by_action_comparison.png"))
    println("  ✓ Saved: health_by_action_comparison.png")
end

function print_summary_statistics(df_base::DataFrame, df_counter::DataFrame)
    println("\n" * "="^80)
    println("  SUMMARY STATISTICS COMPARISON")
    println("="^80)
    
    println("\n📊 BASELINE (with crash shocks):")
    println("  Total observations:    $(nrow(df_base))")
    println("  Deleverage rate:       $(round(100*mean(df_base.action .== "deleverage"), digits=2))%")
    println("  Average health:        $(round(mean(df_base.h), digits=4))")
    println("  Std dev health:        $(round(std(df_base.h), digits=4))")
    println("  Min health:            $(round(minimum(df_base.h), digits=4))")
    println("  Max health:            $(round(maximum(df_base.h), digits=4))")
    
    println("\n📊 COUNTERFACTUAL (pure random walk):")
    println("  Total observations:    $(nrow(df_counter))")
    println("  Deleverage rate:       $(round(100*mean(df_counter.action .== "deleverage"), digits=2))%")
    println("  Average health:        $(round(mean(df_counter.h), digits=4))")
    println("  Std dev health:        $(round(std(df_counter.h), digits=4))")
    println("  Min health:            $(round(minimum(df_counter.h), digits=4))")
    println("  Max health:            $(round(maximum(df_counter.h), digits=4))")
    
    println("\n📈 DIFFERENCES:")
    delever_diff = mean(df_base.action .== "deleverage") - mean(df_counter.action .== "deleverage")
    health_diff = mean(df_base.h) - mean(df_counter.h)
    
    @printf("  Δ Deleverage rate:     %+.2f%% (baseline - counterfactual)\n", 
            100*delever_diff)
    @printf("  Δ Average health:      %+.4f (baseline - counterfactual)\n", 
            health_diff)
    
    if delever_diff > 0
        println("\n  → Crash shocks INCREASE deleveraging activity")
    else
        println("\n  → Crash shocks DECREASE deleveraging activity")
    end
    
    if health_diff < 0
        println("  → Crash shocks LOWER average health levels")
    else
        println("  → Crash shocks RAISE average health levels")
    end
    
    println("\n" * "="^80)
end

########################################################################
# Main
########################################################################

function main()
    println("="^80)
    println("  SCENARIO COMPARISON: Baseline vs Counterfactual")
    println("="^80)
    
    # Load data
    df_base, df_counter = load_comparison_data()
    
    # Create comparison figures
    println("\n📊 Creating comparison figures...")
    plot_health_comparison(df_base, df_counter)
    plot_action_comparison(df_base, df_counter)
    plot_distribution_comparison(df_base, df_counter)
    plot_health_by_action(df_base, df_counter)
    
    # Print statistics
    print_summary_statistics(df_base, df_counter)
    
    println("\n✓ All comparison figures saved to: $FIGURES_PATH")
    println("\n" * "="^80)
end

main()
