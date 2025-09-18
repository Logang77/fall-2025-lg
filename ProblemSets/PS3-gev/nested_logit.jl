using Optim, DataFrames, LinearAlgebra, Statistics, Random

function nested_logit(theta, X, y, nests)
    """
    Nested Logit implementation
    theta: parameters [β; λ] where β are alternative-specific parameters and λ are nest parameters
    X: matrix of covariates
    y: vector of choices
    nests: dictionary mapping each alternative to its nest
    """
    N = size(X, 1)      # number of observations
    K = size(X, 2)      # number of covariates
    J = length(unique(y))  # number of alternatives
    M = length(unique(values(nests)))  # number of nests
    
    # Split parameters into betas and lambdas
    n_beta = K * (J-1)  # parameters for alternatives (minus base)
    betas = reshape(theta[1:n_beta], K, J-1)
    lambdas = theta[n_beta+1:end]  # nest parameters
    
    # Initialize log-likelihood
    loglike = 0.0
    
    # Loop over observations
    for i in 1:N
        # Get chosen alternative for this observation
        chosen = Int(y[i])
        
        # Calculate inclusive values for each nest
        IV = zeros(M)
        for m in 1:M
            # Get alternatives in this nest
            nest_alts = findall(x -> x == m, values(nests))
            
            # Sum over alternatives in nest
            sum_exp = 0.0
            for j in nest_alts
                if j == 1
                    util = 0.0  # base alternative
                else
                    util = dot(X[i,:], betas[:,j-1])
                end
                sum_exp += exp(util/lambdas[m])
            end
            IV[m] = sum_exp^lambdas[m]
        end
        
        # Calculate probability of chosen alternative
        chosen_nest = nests[chosen]
        if chosen == 1
            util_chosen = 0.0
        else
            util_chosen = dot(X[i,:], betas[:,chosen-1])
        end
        
        # Probability = P(nest) * P(alt|nest)
        p_nest = IV[chosen_nest] / sum(IV)
        p_alt_given_nest = exp(util_chosen/lambdas[chosen_nest]) / 
                          (IV[chosen_nest]^(1/lambdas[chosen_nest]))
        
        loglike -= log(p_nest * p_alt_given_nest)
    end
    
    return loglike
end

# Example usage:
function run_nested_logit_example()
    # Generate some example data
    N = 1000  # observations
    K = 3     # covariates
    J = 4     # alternatives
    
    # Generate random covariates
    X = [ones(N) randn(N, K-1)]
    
    # True parameters
    true_betas = [1.0 -0.5 0.8;   # for alternative 2
                  0.5 0.3 -0.4;    # for alternative 3
                  -0.3 0.7 0.2]    # for alternative 4
    
    true_lambdas = [0.5, 0.7]  # nest parameters
    
    # Define nests (alternatives 1,2 in nest 1; alternatives 3,4 in nest 2)
    nests = Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2)
    
    # Generate choices based on nested logit probabilities
    y = zeros(Int, N)
    for i in 1:N
        # Calculate utilities
        V = zeros(J)
        for j in 2:J
            V[j] = dot(X[i,:], true_betas[:,j-1])
        end
        
        # Add random component and make choice
        e = rand(Gumbel(0,1), J)
        utilities = V + e
        y[i] = argmax(utilities)
    end
    
    # Initialize parameters
    n_beta = K * (J-1)
    n_lambda = length(unique(values(nests)))
    init_theta = zeros(n_beta + n_lambda)
    
    # Optimize
    result = optimize(theta -> nested_logit(theta, X, y, nests),
                     init_theta,
                     LBFGS(),
                     Optim.Options(iterations=1000, show_trace=true))
    
    # Extract and reshape results
    theta_hat = Optim.minimizer(result)
    beta_hat = reshape(theta_hat[1:n_beta], K, J-1)
    lambda_hat = theta_hat[n_beta+1:end]
    
    return Dict(
        "beta_hat" => beta_hat,
        "lambda_hat" => lambda_hat,
        "true_betas" => true_betas,
        "true_lambdas" => true_lambdas,
        "convergence" => Optim.converged(result),
        "final_ll" => -Optim.minimum(result)
    )
end