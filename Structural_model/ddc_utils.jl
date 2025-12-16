# ddc_utils.jl
#
# Shared utilities for generator and estimator:
# - common h-grid
# - common flow utility u(h,a;θ)
# - structural expected_V_action(h, V, a)
# - value function iteration with EV1 taste shocks
# - choice probabilities
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

# Shock process (baseline)
const MU    = -0.0005
const SIGMA = 0.06
const BASE_SHOCK_DIST = LogNormal(MU, SIGMA)

# Monte Carlo integration for E[V(h')]
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
# 4. E[V(h′) | h, a] under structural transitions
#############################

"""
    expected_V_action(h, V, a;
                      shock_dist = BASE_SHOCK_DIST,
                      N_mc       = N_MC_EV)

Monte Carlo approximation to E[V(h′) | h, a] with structural transitions:

- a == 1 (stay):
      h′ = clamp(h * ε, H_MIN, H_MAX),   ε ~ LogNormal
- a == 2 (deleverage):
      h′ = max(H_BAR, h)   (deterministic reset to safe region)

This matches your generator’s transition structure.
"""
function expected_V_action(h::Real,
                           V::AbstractVector{T},
                           a::Int;
                           shock_dist::LogNormal = BASE_SHOCK_DIST,
                           N_mc::Int = N_MC_EV) where T<:Real

    if a == 1
        # stay: multiplicative shock
        acc = zero(T)
        for _ in 1:N_mc
            ε      = rand(shock_dist)
            h_next = clamp(h * ε, H_MIN, H_MAX)
            acc   += interp1(H_GRID, V, h_next)
        end
        return acc / N_mc

    elseif a == 2
        # deleverage: deterministic jump to safe level
        h_next = max(H_BAR, h)
        return interp1(H_GRID, V, h_next)

    else
        error("Unknown action index $a (expected 1=stay or 2=deleverage)")
    end
end

#############################
# 5. Value function iteration (2-action logit)
#############################

"""
    solve_value_function(θ_raw, β;
                         shock_dist = BASE_SHOCK_DIST,
                         N_mc       = N_MC_EV)

Solve V(h) on the shared grid H_GRID for given structural parameters θ_raw
and discount factor β, using EV1 taste shocks (Rust-style logit).

Returns:
- V      :: Vector{Float64}     (integrated value function on H_GRID)
- v_a    :: Array{Float64,2}    (2, N_GRID), choice-specific values
"""
function solve_value_function(θ_raw::AbstractVector{T},
                              β::Real;
                              shock_dist::LogNormal = BASE_SHOCK_DIST,
                              N_mc::Int = N_MC_EV,
                              warn_convergence::Bool = false) where T<:Real
    ρ, γ_g = transform_params(θ_raw)

    # Use eltype to support automatic differentiation
    V     = zeros(T, N_GRID)
    V_new = similar(V)

    v_a   = zeros(T, 2, N_GRID)  # [a,i]
    
    diff = Inf  # Initialize diff before loop

    for it in 1:VF_MAXIT
        for (i, h) in enumerate(H_GRID)
            # choice-specific values
            for a in 1:2
                u  = flow_utility(h, a, ρ, γ_g)
                EV = expected_V_action(h, V, a; shock_dist=shock_dist, N_mc=N_mc)
                v_a[a, i] = u + β * EV
                if !isfinite(v_a[a, i])
                    v_a[a, i] = -1e6
                end
            end

            # inclusive value with EV1 shocks, log-sum-exp stabilized
            maxv = max(v_a[1, i], v_a[2, i])
            if !isfinite(maxv)
                maxv = 0.0
            end

            e1 = exp(clamp(v_a[1, i] - maxv, -50.0, 50.0))
            e2 = exp(clamp(v_a[2, i] - maxv, -50.0, 50.0))
            s  = e1 + e2

            V_new[i] = GAMMA_E + maxv + log(max(s, LOG_EPS))
            if !isfinite(V_new[i])
                V_new[i] = V[i]
            end
        end

        diff = maximum(abs.(V_new .- V))
        V   .= V_new

        if diff < VF_TOL
            # println("VFI converged in $it iterations, diff = $diff")
            return V, v_a
        end
    end

    # If we reach here, max iterations exceeded - use diff from last iteration
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

Given v_a[a,i], return P_choice[a,i] = Pr(a | H_GRID[i]).
"""
function choice_probabilities(v_a::AbstractMatrix{T}) where T<:Real
    n_actions, n_states = size(v_a)
    P_choice = zeros(T, n_actions, n_states)

    for i in 1:n_states
        maxv = maximum(@view v_a[:, i])
        if !isfinite(maxv)
            P_choice[:, i] .= 1.0 / n_actions
            continue
        end

        denom = zero(T)
        for a in 1:n_actions
            denom += exp(clamp(v_a[a, i] - maxv, -50.0, 50.0))
        end

        if denom < LOG_EPS
            P_choice[:, i] .= 1.0 / n_actions
        else
            for a in 1:n_actions
                P_choice[a, i] = exp(clamp(v_a[a, i] - maxv, -50.0, 50.0)) / denom
            end
        end
    end

    return P_choice
end

"""
    choice_probs_continuous(h, p_stay_grid)

Interpolate CCPs from the grid for a continuous h:

returns (P(stay | h), P(deleverage | h)).
"""
function choice_probs_continuous(h::Float64, p_stay_grid::Vector{Float64})
    p_stay = interp1(H_GRID, p_stay_grid, h)
    p_del  = 1.0 - p_stay
    return p_stay, p_del
end

"""
    approximate_threshold(p_stay_grid; cutoff=0.5)

Approximate h* where P(stay | h*) = cutoff, via linear interpolation.
Returns NaN if no crossing is found.
"""
function approximate_threshold(p_stay_grid::AbstractVector{Float64}; cutoff::Float64 = 0.5)
    for i in 2:length(H_GRID)
        p1 = p_stay_grid[i-1]
        p2 = p_stay_grid[i]
        if (p1 - cutoff) * (p2 - cutoff) <= 0.0 && p1 != p2
            h1 = H_GRID[i-1]
            h2 = H_GRID[i]
            w  = (cutoff - p1) / (p2 - p1)
            return h1 + w * (h2 - h1)
        end
    end
    return NaN
end
