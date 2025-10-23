################################################################################
# Problem Set 7 - Starter Code
# ECON 6343: Econometrics III
# GMM and SMM Estimation
################################################################################

using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

################################################################################
# Data Loading and Preparation Functions
################################################################################

"""
    load_data(url)

Load wage data and create design matrix for OLS regression.
Returns: DataFrame, X matrix, log wage vector
"""
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    # Create X matrix with intercept, age, race indicator, college grad indicator
    X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]
    # Create y as log wage
    y = log.(df.wage)
    return df, X, y
end

"""
    prepare_occupation_data(df)

Prepare occupation data for multinomial logit.
Collapse occupation categories and create covariates.
"""
function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    df.white = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    return df, X, y
end

################################################################################
# Question 1: OLS via GMM
################################################################################

"""
    ols_gmm(β, X, y)

GMM objective function for OLS regression.
Uses identity weighting matrix.

Mathematical form:
    J(β) = g'Wg where W = I and g = (y - Xβ)

Arguments:
- β: coefficient vector
- X: N×K design matrix
- y: N×1 outcome vector

Returns: scalar objective function value
"""
function ols_gmm(β, X, y)
    # Compute predicted values ŷ = Xβ
    ŷ = X * β
    
    # Compute residuals g = y - ŷ
    g = y .- ŷ
    
    # Compute objective function J = g'Ig
    J = dot(g, g) 
    
    return J
end

################################################################################
# Question 2: Multinomial Logit via MLE and GMM
################################################################################

"""
    mlogit_mle(α, X, y)

Maximum likelihood objective function for multinomial logit.

Model:
    P(y_i = j) = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    where β_J = 0 (normalization)

Arguments:
- α: vectorized coefficients of dimension K*(J-1)
- X: N×K covariate matrix
- y: N×1 choice vector (integer values 1,2,...,J)

Returns: negative log-likelihood value
"""
function mlogit_mle(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # Create choice indicator matrix: bigY[i,j] = 1 if y[i]==j, 0 otherwise
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end
    
    # Reshape α into K×(J-1) matrix and append zeros for normalization
    bigα = [reshape(α, K, J-1) zeros(K)]
    
    # Compute choice probabilities P[i,j] = exp(X[i,:]'β_j) / sum_k(exp(X[i,:]'β_k))
    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

    # Compute negative log-likelihood: -Σ_i Σ_j d_ij * log(P_ij)
    loglike = -sum(bigY .* log.(P))
    
    return loglike
end

"""
    mlogit_gmm(α, X, y)

GMM objective function for multinomial logit using just-identified moments.

Moment conditions:
    E[X_i(d_ij - P_ij(α))] = 0 for j = 1,...,J-1

This gives K*(J-1) moments for K*(J-1) parameters.
"""
function mlogit_gmm(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end
    
    # Reshape coefficients
    bigα = [reshape(α, K, J-1) zeros(K)]
    
    # Compute choice probabilities
    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))
    
    # Compute moment vector g of dimension K*(J-1)
    # Each element is: mean((d_ij - P_ij) * X_ik)
    g = zeros((J-1)*K)
    for j = 1:(J-1)
        for k = 1:K
            g[(j-1)*K + k] = mean((bigY[:, j] .- P[:, j]) .* X[:, k])
        end
    end
    
    # Compute objective function J = N * g'g
    J = N * dot(g, g)
    
    return J
end

"""
    mlogit_gmm_overid(α, X, y)

Over-identified GMM for multinomial logit.

Uses N*J moments: d_ij - P_ij(α) for all i,j
This is over-identified since we only have K*(J-1) parameters.
"""
function mlogit_gmm_overid(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end
    
    # Reshape coefficients
    bigα = [reshape(α, K, J-1) zeros(K)]
    
    # Compute choice probabilities
    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))
    
    # Stack moments as vector: g = vec(d - P)
    # This creates an N*J dimensional vector
    g = bigY[:] .- P[:]
    
    # Compute objective J = g'Wg with W = I
    J = dot(g, g)
    
    return J
end

################################################################################
# Question 3: Simulate Data from Multinomial Logit
################################################################################

"""
    sim_logit(N, J)

Simulate multinomial logit data using inverse CDF method.

Steps:
1. Generate X ~ N(μ, Σ)
2. Set coefficient matrix β (K×J)
3. Compute choice probabilities P_ij
4. Draw uniform random variable ε_i
5. Assign y_i based on which cumulative probability bracket ε_i falls in

Arguments:
- N: number of observations (default: 100,000)
- J: number of choice alternatives (default: 4)

Returns: (Y, X) where Y is N×1 choice vector, X is N×K covariate matrix
"""
function sim_logit(N=100_000, J=4)
    # Generate X matrix with intercept and covariates
    X = hcat(ones(N), randn(N), rand(N) .> 0.5, 10 .* rand(N))
    
    # Create coefficient matrix β (dimension K×J)
    # Last column is zeros for normalization
    if J == 4
        β = hcat([1, -1, 0.5, -0.5], [0, 0.5, 0.3, 2], [0, -0.5, 2, 0.5], zeros(4))
    else
        # Generate random coefficients
        β = -2 .+ 4 .* rand(size(X,2), J)
        β[:, end] = 0
    end
    
    # Compute choice probabilities P_ij = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    P = exp.(X*β) ./ sum.(eachrow(exp.(X*β))) 
    @assert size(P) == (N, J)
    
    # Draw uniform random variables
    draw = rand(N)
    
    # Generate choices based on cumulative probabilities
    # For each person i, find j such that Σ_{k=j}^J P_ik > ε_i
    Y = zeros(N)
    for j = 1:J
        Y += sum(P[:, j:J]; dims = 2) .> draw
    end
    
    return Y, X
end

"""
    sim_logit_w_gumbel(N, J)

Simulate multinomial logit data using Gumbel shocks (alternative method).

This method directly uses the fact that if ε ~ Gumbel(0,1), then
    y_i = argmax_j (X_i'β_j + ε_ij)
generates choices from a multinomial logit model.

This is often simpler and more numerically stable than the inverse CDF method.
"""
function sim_logit_w_gumbel(N=100_000, J=4)
    # Generate X and β (same as sim_logit)
    X = hcat(ones(N), randn(N), rand(N) .> 0.5, 10 .* rand(N))
    if J == 4
        β = hcat([1, -1, 0.5, -0.5], [0, 0.5, 0.3, 2], [0, -0.5, 2, 0.5], zeros(4))
    else
        # Generate random coefficients
        β = -2 .+ 4 .* rand(size(X,2), J)
        β[:, end] = 0
    end
    
    # Draw Gumbel errors
    ε = rand(Gumbel(0,1), N, J)
    
    # Choose alternative that maximizes utility: Y_i = argmax_j (X_i'β_j + ε_ij)
    Y = argmax.(eachrow(X*β .+ ε))
    
    return Y, X
end

################################################################################
# Question 5: Multinomial Logit via SMM
################################################################################

"""
    mlogit_smm_overid(α, X, y, D)

Simulated Method of Moments for multinomial logit.

Algorithm:
1. For given parameters α, compute model-implied probabilities
2. Simulate D datasets using the Gumbel method
3. Compute average simulated choice frequencies
4. Match simulated frequencies to actual frequencies

Arguments:
- α: parameter vector
- X: covariate matrix
- y: actual choices
- D: number of simulation draws

Returns: SMM objective function value
"""
function mlogit_smm_overid(α, X, y, D)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # Create actual choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end
    
    # Initialize simulated choice frequencies
    bigỸ = zeros(N, J)
    
    # Reshape parameters
    bigα = [reshape(α, K, J-1) zeros(K)]
    
    # Simulate D datasets and accumulate frequencies
    Random.seed!(1234)  # For reproducibility
    for d = 1:D
        # Draw Gumbel shocks
        ε = rand(Gumbel(0,1), N, J)
        
        # Generate simulated choices: ỹ = argmax_j(X*bigα + ε)
        ỹ = argmax.(eachrow(X*bigα .+ ε))
        
        # Update frequency counts
        for j = 1:J
            bigỸ[:,j] .+= (ỹ .== j) * (1/D)
        end
    end
    
    # Compute moment vector (actual - simulated frequencies)
    g = bigY[:] .- bigỸ[:]
    
    # Compute objective function
    J_obj = dot(g, g)
    
    return J_obj
end

################################################################################
# Main Function - Question 6: Wrap everything in a function
################################################################################

"""
    main()

Main function that runs all estimation procedures.
Keeps everything out of global scope for better performance.
"""
function main()
    println("="^80)
    println("Problem Set 7: GMM and SMM Estimation")
    println("="^80)
    
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df, X_wage, y_wage = load_data(url)
    df, X, y = prepare_occupation_data(df)
    
    #--------------------------------------------------------------------------
    # Question 1: OLS via GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 1: OLS Estimation via GMM")
    println("="^80)
    
    # Estimate β using GMM
    β_hat_gmm = optimize(b -> ols_gmm(b, X_wage, y_wage), 
                         rand(size(X_wage,2)), 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    
    # Compare with closed-form OLS: (X'X)^(-1)X'y
    β_ols = X_wage \ y_wage
    
    println("GMM estimates: ", β_hat_gmm.minimizer)
    println("OLS estimates: ", β_ols)
    
    
    #--------------------------------------------------------------------------
    # Question 2: Multinomial Logit via MLE and GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 2: Multinomial Logit via MLE and GMM")
    println("="^80)
    
    # Get starting values from series of binary logits
    # Initialize svals matrix to store coefficients (4 coefficients x 7 occupations)
    svals = zeros(4, 7)
    
    # Create dummy variables for each occupation
    for j = 1:7
        tempname = Symbol(string("occ", j))
        df[:, tempname] = df.occupation .== j
        f = term(tempname) ~ term(:age) + term(:white) + term(:collgrad) 
        svals[:,j] = coef(lm(f, df)) 
    end
    # Take differences relative to base category
    svals = svals[:,1:6] .- svals[:,7]
    svals = svals[:]
    
    # Estimate via MLE
    α_hat_mle = optimize(a -> mlogit_mle(a, X, y), 
                         svals, 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-5, iterations=100, show_trace=true))

    println("MLE estimates: ", α_hat_mle.minimizer)
    
    # Estimate via GMM using MLE estimates as starting values
    α_hat_gmm_mle_start = optimize(a -> mlogit_gmm_overid(a, X, y), 
                                    α_hat_mle.minimizer, 
                                    LBFGS(), 
                                    Optim.Options(g_tol=1e-5, iterations=30, show_trace=true))

    println("GMM estimates: ", α_hat_gmm_mle_start.minimizer)
        
    # Estimate via GMM using random starting values
    α_hat_gmm_random_start = optimize(a -> mlogit_gmm_overid(a, X, y), 
                                       rand(length(svals)), 
                                       LBFGS(), 
                                       Optim.Options(g_tol=1e-5, iterations=100, show_trace=true))
    
    println("Random start GMM estimates: ", α_hat_gmm_random_start.minimizer)
    
    #--------------------------------------------------------------------------
    # Question 3: Simulate and Recover Parameters
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 3: Simulate Data and Recover Parameters")
    println("="^80)
    
    # Simulate data
    ySim, XSim = sim_logit(100, 4)
    
    println("Simulated choice range: ", minimum(ySim), " to ", maximum(ySim))
    println("X matrix means: ", mean(XSim; dims = 1))

    # Estimate parameters from simulated data (commented out for now)
    # α_hat_sim = optimize(a -> mlogit_mle(a, XSim, ySim), 
    #                      rand(9),  # 3 covariates × 3 non-base alternatives
    #                      LBFGS(), 
    #                      Optim.Options(g_tol = 1e-6, iterations = 100_000, show_trace = true))
    
    # Compare estimated parameters to true parameters (would be implemented here)
    
    # Try alternative simulation method with Gumbel (commented out for now)
    # ySim_gumbel, XSim_gumbel = sim_logit_w_gumbel(100_000, 4)
    # Estimate and compare
    
    #--------------------------------------------------------------------------
    # Question 5: Multinomial Logit via SMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 5: Multinomial Logit via SMM")
    println("="^80)
    
    # Estimate via SMM (start with small D for testing, increase for final estimates)
    α_hat_smm = optimize(th -> mlogit_smm_overid(th, X, y, 100),  # Small D for testing
                         α_hat_mle.minimizer,  # Use MLE as starting values
                         LBFGS(), 
                         Optim.Options(g_tol=1e-6, iterations=1000, show_trace=true))
    
    # Compare SMM estimates to MLE and GMM estimates
    println("MLE estimates: ", α_hat_mle.minimizer)
    println("GMM estimates: ", α_hat_gmm_mle_start.minimizer)
    println("SMM estimates: ", α_hat_smm.minimizer)
    
    println("\n" * "="^80)
    println("Estimation Complete!")
    println("="^80)
    
    # Return estimates for testing
    α_hat_sim = nothing  # Placeholder since simulation estimation is commented out
    return β_hat_gmm, α_hat_mle, α_hat_gmm_mle_start, α_hat_gmm_random_start, α_hat_sim, α_hat_smm
end

################################################################################
# Run main function
################################################################################

# Uncomment to run:
# @time main()