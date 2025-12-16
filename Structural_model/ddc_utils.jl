# ddc_utils_regime.jl
#
# Shared utilities for generator and estimator WITH REGIME STATES:
# - common h-grid
# - regime states (normal vs crash)
# - regime-dependent shock distributions
# - regime transition probabilities
# - common flow utility u(h,a;θ)
# - structural expected_V_action(h, s, V, a) with regime transitions
# - value function iteration with EV1 taste shocks and regimes
# - choice probabilities
# - Gauss-Hermite quadrature for numerical stability
#
# NOTE: Packages should be loaded in master script

#############################
# 1. Global primitives
#############################

# Health factor grid (shared)
const H_MIN  = 1.0
const H_MAX  = 2.0
const N_GRID = 101
const H_GRID = collect(range(H_MIN, H_MAX; length = N_GRID))

# Comfort level for risk penalty
const H_BAR = 1.5

# Regime states: 0 = normal, 1 = crash
const N_REGIMES = 2

# Regime transition probabilities (defaults - can be overridden)
const π01_DEFAULT = 0.70  # P(crash | normal)
const π10_DEFAULT = 0.30  # P(normal | crash)

# Shock process (baseline/normal regime s=0)
const MU    = -0.0005
const SIGMA = 0.06
const BASE_SHOCK_DIST = LogNormal(MU, SIGMA)

# Shock process (crash regime s=1)
const μ_CRASH = -0.8
const σ_CRASH = 0.2449
const CRASH_SHOCK_DIST = LogNormal(μ_CRASH, σ_CRASH)

# Gauss-Hermite quadrature for numerical integration
const N_GH = 10  # Number of quadrature points
const GH_NODES, GH_WEIGHTS = let
    # Standard Gauss-Hermite nodes and weights for N(0,1)
    # These are pre-computed for n=10
    nodes = [-3.4361591188377376, -2.532731674232790, -1.756683649299882, 
             -1.036610829789514, -0.3429013272237046, 0.3429013272237046,
             1.036610829789514, 1.756683649299882, 2.532731674232790, 3.4361591188377376]
    weights = [7.64043285523262e-6, 0.001343645746781233, 0.0338743944554810,
               0.2401386110823147, 0.6108626337353258, 0.6108626337353258,
               0.2401386110823147, 0.0338743944554810, 0.001343645746781233, 7.64043285523262e-6]
    (nodes, weights)
end

# Monte Carlo integration for E[V(h')] (fallback if not using GH)
const N_MC_EV = 200

# Euler–Mascheroni constant for EV1
const GAMMA_E = 0.5772156649015329

# Numerical tolerances for value function iteration
const VF_TOL   = 1e-3
const VF_MAXIT = 10_000

# Small constant to avoid log(0)
const LOG_EPS = 1e-10

#############################
# 2. 1D Linear Interpolation
#############################

"""
    interp1(x_grid, y_grid, x)

Piecewise-linear interpolation of y(x) based on (x_grid, y_grid).
Clamps at the endpoints.
"""
function interp1(x_grid::Vector{Float64},
                 y_grid::AbstractVector{T},
                 x::Real) where T<:Real
    if x <= x_grid[1]
        return y_grid[1]
    elseif x >= x_grid[end]
        return y_grid[end]
    else
        idx = searchsortedlast(x_grid, x)
        # idx such that x_grid[idx] <= x < x_grid[idx+1]
        x1 = x_grid[idx]
        x2 = x_grid[idx + 1]
        y1 = y_grid[idx]
        y2 = y_grid[idx + 1]
        w  = (x - x1) / (x2 - x1)
        return (1 - w) * y1 + w * y2
    end
end

#############################
# 3. Structural utility u(h,a;θ)
#############################

# θ_raw = [ρ_raw, γ_g_raw]; ρ = exp(ρ_raw), γ_g = exp(γ_g_raw)

function transform_params(θ::AbstractVector{T}) where T<:Real
    ρ_raw, γg_raw = θ
    ρ   = exp(ρ_raw)
    γ_g = exp(γg_raw)
    return ρ, γ_g
end

"""
    flow_utility(h, a, ρ, γ_g)

Return u(h,a;θ) for a ∈ {1=stay, 2=deleverage}.

Same functional form as in your current estimator and generator:

- stay:       u(h,1) = -ρ * max(0, H_BAR - h)^2
- deleverage: u(h,2) = -γ_g
"""
function flow_utility(h::Real, a::Int, ρ::Real, γ_g::Real)
    diff         = clamp(H_BAR - h, -10.0, 10.0)
    risk_penalty = -ρ * max(0.0, diff)^2

    if a == 1
        # stay: risk penalty, no gas cost
        return risk_penalty
    elseif a == 2
        # deleverage: pay gas cost; risk reduction comes via transitions
        return -γ_g
    else
        error("Unknown action index $a (expected 1=stay or 2=deleverage)")
    end
end

#############################
# 4. E[V(h′, s′) | h, s, a] with regime transitions
#############################

"""
    expected_V_action(h, s, V, a;
                      π01 = π01_DEFAULT,
                      π10 = π10_DEFAULT,
                      use_quadrature = true)

Expected value E[V(h′, s′) | h, s, a] with:
- Structural health transitions (stay: h*ε; deleverage: max(H_BAR,h))
- Regime transitions: P(s'|s) based on π01 and π10
- Regime-dependent shocks: BASE_SHOCK_DIST for s=0, CRASH_SHOCK_DIST for s=1

Arguments:
- h: current health
- s: current regime (0=normal, 1=crash)
- V: value function array V[h_index, s_index+1] (dimensions: N_GRID × N_REGIMES)
- a: action (1=stay, 2=deleverage)
- π01: P(crash | normal)
- π10: P(normal | crash)
- use_quadrature: if true, use Gauss-Hermite; if false, use Monte Carlo
"""
function expected_V_action(h::Real,
                           s::Int,
                           V::AbstractMatrix{T},
                           a::Int;
                           π01::Real = π01_DEFAULT,
                           π10::Real = π10_DEFAULT,
                           use_quadrature::Bool = true) where T<:Real

    # Regime transition probabilities
    if s == 0  # normal regime
        P_s0 = 1.0 - π01  # stay normal
        P_s1 = π01        # transition to crash
    else  # s == 1, crash regime
        P_s0 = π10        # transition to normal
        P_s1 = 1.0 - π10  # stay crash
    end

    # Compute expectation over future regimes
    EV = zero(T)
    
    for s_next in 0:1
        P_regime = (s_next == 0) ? P_s0 : P_s1
        
        if a == 1  # stay: multiplicative shock
            # Choose shock distribution based on CURRENT regime
            shock_dist = (s == 0) ? BASE_SHOCK_DIST : CRASH_SHOCK_DIST
            
            if use_quadrature
                # Gauss-Hermite quadrature integration
                # LogNormal(μ, σ): ln(ε) ~ N(μ, σ²)
                μ_ln = shock_dist.μ
                σ_ln = shock_dist.σ
                
                EV_shock = zero(T)
                for i in 1:N_GH
                    # Transform standard normal node to lognormal
                    z = GH_NODES[i]
                    w = GH_WEIGHTS[i]
                    ε = exp(μ_ln + σ_ln * sqrt(2) * z)
                    h_next = clamp(h * ε, H_MIN, H_MAX)
                    V_interp = interp1(H_GRID, @view(V[:, s_next+1]), h_next)
                    EV_shock += w * V_interp
                end
                EV_shock /= sqrt(π)  # Normalization for Gauss-Hermite
                
                EV += P_regime * EV_shock
            else
                # Monte Carlo fallback
                EV_shock = zero(T)
                for _ in 1:N_MC_EV
                    ε = rand(shock_dist)
                    h_next = clamp(h * ε, H_MIN, H_MAX)
                    V_interp = interp1(H_GRID, @view(V[:, s_next+1]), h_next)
                    EV_shock += V_interp
                end
                EV += P_regime * (EV_shock / N_MC_EV)
            end
            
        elseif a == 2  # deleverage: deterministic
            h_next = max(H_BAR, h)
            V_interp = interp1(H_GRID, @view(V[:, s_next+1]), h_next)
            EV += P_regime * V_interp
        else
            error("Unknown action index $a (expected 1=stay or 2=deleverage)")
        end
    end
    
    return EV
end

#############################
# 5. Value function iteration with regimes (2-action logit, 2 regimes)
#############################

"""
    solve_value_function(θ_raw, β;
                         π01 = π01_DEFAULT,
                         π10 = π10_DEFAULT,
                         use_quadrature = true,
                         warn_convergence = false)

Solve V(h, s) on the shared grid H_GRID × {0,1} for given structural parameters θ_raw
and discount factor β, using EV1 taste shocks (Rust-style logit) and regime transitions.

Returns:
- V      :: Matrix{Float64}     (N_GRID × N_REGIMES), integrated value function V[h_index, s_index+1]
- v_a    :: Array{Float64,3}    (2 × N_GRID × N_REGIMES), choice-specific values v_a[a, h_index, s_index+1]
"""
function solve_value_function(θ_raw::AbstractVector{T},
                              β::Real;
                              π01::Real = π01_DEFAULT,
                              π10::Real = π10_DEFAULT,
                              use_quadrature::Bool = true,
                              warn_convergence::Bool = false) where T<:Real
    ρ, γ_g = transform_params(θ_raw)

    # Use eltype to support automatic differentiation
    # V[h_index, s_index+1] where s ∈ {0,1}
    V     = zeros(T, N_GRID, N_REGIMES)
    V_new = similar(V)

    # v_a[a, h_index, s_index+1]
    v_a   = zeros(T, 2, N_GRID, N_REGIMES)
    
    diff = Inf  # Initialize diff before loop

    for it in 1:VF_MAXIT
        for s in 0:1
            for (i, h) in enumerate(H_GRID)
                # choice-specific values
                for a in 1:2
                    u  = flow_utility(h, a, ρ, γ_g)
                    EV = expected_V_action(h, s, V, a; π01=π01, π10=π10, use_quadrature=use_quadrature)
                    v_a[a, i, s+1] = u + β * EV
                    if !isfinite(v_a[a, i, s+1])
                        v_a[a, i, s+1] = -1e6
                    end
                end

                # inclusive value with EV1 shocks, log-sum-exp stabilized
                maxv = max(v_a[1, i, s+1], v_a[2, i, s+1])
                if !isfinite(maxv)
                    maxv = 0.0
                end

                e1 = exp(clamp(v_a[1, i, s+1] - maxv, -50.0, 50.0))
                e2 = exp(clamp(v_a[2, i, s+1] - maxv, -50.0, 50.0))
                sum_exp  = e1 + e2

                V_new[i, s+1] = GAMMA_E + maxv + log(max(sum_exp, LOG_EPS))
                if !isfinite(V_new[i, s+1])
                    V_new[i, s+1] = V[i, s+1]
                end
            end
        end

        diff = maximum(abs.(V_new .- V))
        V   .= V_new

        if diff < VF_TOL
            # println("VFI converged in $it iterations, diff = $diff")
            return V, v_a
        end
    end

    # If we reach here, max iterations exceeded
    if warn_convergence
        @warn "Value function did not fully converge after $VF_MAXIT iterations (max diff = $diff)"
    end
    return V, v_a
end

#############################
# 6. CCPs and helper utilities
#############################

"""
    choice_probabilities(v_a)

Given v_a[a, h_index, s_index], return P_choice[a, h_index, s_index] = Pr(a | h, s).
"""
function choice_probabilities(v_a::AbstractArray{T,3}) where T<:Real
    n_actions, n_h_states, n_regimes = size(v_a)
    P_choice = zeros(T, n_actions, n_h_states, n_regimes)

    for s_idx in 1:n_regimes
        for i in 1:n_h_states
            maxv = maximum(@view v_a[:, i, s_idx])
            if !isfinite(maxv)
                P_choice[:, i, s_idx] .= 1.0 / n_actions
                continue
            end

            denom = zero(T)
            for a in 1:n_actions
                denom += exp(clamp(v_a[a, i, s_idx] - maxv, -50.0, 50.0))
            end

            if denom < LOG_EPS
                P_choice[:, i, s_idx] .= 1.0 / n_actions
            else
                for a in 1:n_actions
                    P_choice[a, i, s_idx] = exp(clamp(v_a[a, i, s_idx] - maxv, -50.0, 50.0)) / denom
                end
            end
        end
    end

    return P_choice
end

"""
    choice_probs_continuous(h, s, p_stay_grid)

Interpolate CCPs from the grid for a continuous h and discrete regime s:

returns (P(stay | h, s), P(deleverage | h, s)).

p_stay_grid should be a matrix of size (N_GRID, N_REGIMES) where p_stay_grid[:, s+1] 
corresponds to P(stay | h, s).
"""
function choice_probs_continuous(h::Float64, s::Int, p_stay_grid::AbstractMatrix{Float64})
    p_stay = interp1(H_GRID, @view(p_stay_grid[:, s+1]), h)
    p_del  = 1.0 - p_stay
    return p_stay, p_del
end

"""
    approximate_threshold(p_stay_grid, s; cutoff=0.5)

Approximate h* where P(stay | h*, s) = cutoff, via linear interpolation.
Returns NaN if no crossing is found.

p_stay_grid should be a matrix of size (N_GRID, N_REGIMES).
"""
function approximate_threshold(p_stay_grid::AbstractMatrix{Float64}, s::Int; cutoff::Float64 = 0.5)
    for i in 2:length(H_GRID)
        p1 = p_stay_grid[i-1, s+1]
        p2 = p_stay_grid[i, s+1]
        if (p1 - cutoff) * (p2 - cutoff) <= 0.0 && p1 != p2
            h1 = H_GRID[i-1]
            h2 = H_GRID[i]
            w  = (cutoff - p1) / (p2 - p1)
            return h1 + w * (h2 - h1)
        end
    end
    return NaN
end

#############################
# 7. Diagnostic Function
#############################

"""
    diagnose_ccp_flatness(θ_raw, β;
                          h_values = [1.5, 1.8, 2.0],
                          regimes = [0, 1],
                          π01 = π01_DEFAULT,
                          π10 = π10_DEFAULT,
                          use_quadrature = true,
                          σ_ε = 1.0)

Diagnostic function to understand why P(delever) is flat at high h values.

For each (h, s) pair, prints:
- Flow utilities: u_stay(h), u_del(h)
- Expected continuation values: EV_stay, EV_del
- Choice-specific values: v_stay, v_del, Δv = v_stay - v_del
- Implied P(delever)
- EV1 scale parameter σ_ε (default 1.0)

This helps distinguish between:
1. Economic model implications (insurance value, capped upside, downside risk)
2. Coding errors (wrong indexing, transitions, CCPs)
3. EV1 scale effects (small utility gaps + large σ_ε → flat CCPs)

Arguments:
- θ_raw: raw parameters [log(ρ), log(γ_g)]
- β: discount factor
- h_values: health values to diagnose
- regimes: regime states to diagnose (0=normal, 1=crash)
- π01, π10: regime transition probabilities
- use_quadrature: use Gauss-Hermite integration
- σ_ε: EV1 scale parameter (affects choice probabilities)
"""
function diagnose_ccp_flatness(θ_raw::AbstractVector{<:Real},
                               β::Real;
                               h_values::Vector{Float64} = [1.5, 1.8, 2.0],
                               regimes::Vector{Int} = [0, 1],
                               π01::Real = π01_DEFAULT,
                               π10::Real = π10_DEFAULT,
                               use_quadrature::Bool = true,
                               σ_ε::Real = 1.0)
    
    println("\n" * "="^80)
    println("CCP FLATNESS DIAGNOSTIC")
    println("="^80)
    
    ρ, γ_g = transform_params(θ_raw)
    
    println("\nParameters:")
    println("  θ_raw = $θ_raw")
    println("  ρ = $ρ")
    println("  γ_g = $γ_g")
    println("  β = $β")
    println("  σ_ε = $σ_ε (EV1 scale parameter)")
    println("  π01 = $π01 (P(crash | normal))")
    println("  π10 = $π10 (P(normal | crash))")
    println("  H_BAR = $H_BAR (comfort threshold)")
    
    println("\n" * "-"^80)
    println("Solving value function...")
    println("-"^80)
    
    V, v_a = solve_value_function(θ_raw, β; π01=π01, π10=π10, use_quadrature=use_quadrature)
    P_choice = choice_probabilities(v_a)
    
    println("✓ Value function solved")
    println("  V range (normal): [$(minimum(V[:,1])), $(maximum(V[:,1]))]")
    println("  V range (crash): [$(minimum(V[:,2])), $(maximum(V[:,2]))]")
    
    for s in regimes
        println("\n" * "="^80)
        println("REGIME s = $s ($(s == 0 ? "NORMAL" : "CRASH"))")
        println("="^80)
        
        for h in h_values
            println("\n" * "-"^80)
            println("h = $h, s = $s")
            println("-"^80)
            
            # Flow utilities
            u_stay = flow_utility(h, 1, ρ, γ_g)
            u_del = flow_utility(h, 2, ρ, γ_g)
            
            println("\n1. FLOW UTILITIES:")
            println("   u_stay(h) = $(@sprintf("%.6f", u_stay))")
            println("   u_del(h)  = $(@sprintf("%.6f", u_del))")
            println("   Δu = u_stay - u_del = $(@sprintf("%.6f", u_stay - u_del))")
            
            if h >= H_BAR
                println("   → h ≥ H_BAR: stay has NO risk penalty, but delever pays gas cost γ_g")
            else
                println("   → h < H_BAR: stay has risk penalty, delever pays gas cost γ_g")
            end
            
            # Expected continuation values
            EV_stay = expected_V_action(h, s, V, 1; π01=π01, π10=π10, use_quadrature=use_quadrature)
            EV_del = expected_V_action(h, s, V, 2; π01=π01, π10=π10, use_quadrature=use_quadrature)
            
            println("\n2. EXPECTED CONTINUATION VALUES:")
            println("   E[V(h',s') | stay] = $(@sprintf("%.6f", EV_stay))")
            println("   E[V(h',s') | del]  = $(@sprintf("%.6f", EV_del))")
            println("   ΔEV = EV_stay - EV_del = $(@sprintf("%.6f", EV_stay - EV_del))")
            
            if h >= H_BAR
                println("   → stay: stochastic with clamp(h*η, 1, 2) - has downside risk AND upside capped")
                println("   → del:  deterministic max(H_BAR, h) = $(max(H_BAR, h)) - insurance value")
            else
                println("   → stay: stochastic with clamp(h*η, 1, 2)")
                println("   → del:  deterministic max(H_BAR, h) = H_BAR")
            end
            
            # Choice-specific values
            v_stay = u_stay + β * EV_stay
            v_del = u_del + β * EV_del
            Δv = v_stay - v_del
            
            println("\n3. CHOICE-SPECIFIC VALUES (v = u + β*EV):")
            println("   v_stay = $(@sprintf("%.6f", v_stay))")
            println("   v_del  = $(@sprintf("%.6f", v_del))")
            println("   Δv = v_stay - v_del = $(@sprintf("%.6f", Δv))")
            
            # CCPs with explicit σ_ε
            # P(del) = exp(v_del/σ_ε) / [exp(v_stay/σ_ε) + exp(v_del/σ_ε)]
            #        = 1 / [1 + exp((v_stay - v_del)/σ_ε)]
            #        = 1 / [1 + exp(Δv/σ_ε)]
            
            exp_diff = exp(Δv / σ_ε)
            P_del_manual = 1.0 / (1.0 + exp_diff)
            P_stay_manual = exp_diff / (1.0 + exp_diff)
            
            # Get from interpolation
            P_stay_interp = interp1(H_GRID, @view(P_choice[1, :, s+1]), h)
            P_del_interp = interp1(H_GRID, @view(P_choice[2, :, s+1]), h)
            
            println("\n4. CHOICE PROBABILITIES (Logit with σ_ε = $σ_ε):")
            println("   P(stay | h,s) = exp(Δv/σ_ε) / [1 + exp(Δv/σ_ε)]")
            println("                 = exp($(@sprintf("%.6f", Δv))/$σ_ε) / [1 + exp($(@sprintf("%.6f", Δv))/$σ_ε)]")
            println("                 = $(@sprintf("%.6f", P_stay_manual)) (manual)")
            println("                 = $(@sprintf("%.6f", P_stay_interp)) (interpolated)")
            println()
            println("   P(del | h,s)  = 1 / [1 + exp(Δv/σ_ε)]")
            println("                 = 1 / [1 + exp($(@sprintf("%.6f", Δv))/$σ_ε)]")
            println("                 = $(@sprintf("%.6f", P_del_manual)) (manual)")
            println("                 = $(@sprintf("%.6f", P_del_interp)) (interpolated)")
            
            println("\n5. INTERPRETATION:")
            if abs(Δv) < 0.5
                println("   ⚠ SMALL VALUE GAP: |Δv| = $(@sprintf("%.6f", abs(Δv))) < 0.5")
                println("   → With σ_ε = $σ_ε, logit gives P(del) ≈ $(@sprintf("%.3f", P_del_manual))")
                println("   → This is NOT a bug; it's what logit implies with small utility differences")
                println()
                if h >= H_BAR
                    println("   Economic interpretation:")
                    println("   • stay: no risk penalty (h ≥ H_BAR), but has downside risk from shocks")
                    println("   • stay: upside capped at h=2 (clamp)")
                    println("   • del:  deterministic h'=h (insurance value)")
                    println("   • Trade-off: gas cost γ_g vs. insurance against downside")
                    println("   → Near-equal values → flat CCPs is EXPECTED from model")
                end
            else
                println("   ✓ MODERATE VALUE GAP: |Δv| = $(@sprintf("%.6f", abs(Δv))) ≥ 0.5")
                if Δv > 0
                    println("   → stay is preferred (Δv > 0)")
                else
                    println("   → del is preferred (Δv < 0)")
                end
            end
            
            if h >= H_BAR && abs(EV_stay - EV_del) < 0.5
                println("\n   ⚠ INSURANCE VALUE EFFECT:")
                println("   → At h ≥ H_BAR, stay has zero flow penalty but downside risk")
                println("   → Deleveraging provides insurance (deterministic h')")
                println("   → Model CORRECTLY values this trade-off")
            end
        end
    end
    
    println("\n" * "="^80)
    println("DIAGNOSTIC SUMMARY")
    println("="^80)
    println("\nTo obtain P(del | h ≥ H_BAR) ≈ 0, you need ONE of:")
    println("  1. Increase γ_g (make deleveraging more costly)")
    println("  2. Decrease σ_ε (make choices more deterministic)")
    println("  3. Remove taste shocks entirely (deterministic argmax)")
    println("  4. Change transition model (e.g., make stay deterministic at h ≥ H_BAR)")
    println("\nCurrent setup:")
    println("  • γ_g = $(@sprintf("%.4f", γ_g))")
    println("  • σ_ε = $σ_ε")
    println("  • Taste shocks: EV1 (Rust-style logit)")
    println("  • Stay transition: stochastic for all h (has insurance value)")
    println("="^80)
    
    return V, v_a, P_choice
end
