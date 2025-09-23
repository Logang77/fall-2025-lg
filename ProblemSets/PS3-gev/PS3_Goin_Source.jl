
#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation
    return X, Z, y
end

#---------------------------------------------------
# Question 1: Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------

function mlogit_with_Z(theta, X, Z, y)
    # Extract parameters
    # theta = [alpha1, alpha2, ..., alpha21, gamma]
    # alpha has K*(J-1) = 3*7 = 21 elements
    # gamma is the coefficient on Z
    alpha = theta[1:end-1]  # first 21 elements
    gamma = theta[end]      # last element
    
    K = size(X, 2)  # number of covariates in X
    J = length(unique(y))  # number of choices (8)
    N = length(y)   # number of observations
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha into K x (J-1) matrix, add zeros for normalized choice
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # Initialize probability matrix
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)
    dem = zeros(T, N)
    
    # Fill in: compute numerator for each choice j
    for j = 1:J
        # num[:,j] = exp.(...)
     num[:, j] = exp.(X*bigAlpha[:, j] .+ gamma * (Z[:, j] - Z[:, J]))

    end
    
    # Fill in: compute denominator (sum of numerators)

    dem = sum(num, dims=2) # takes a sum within a row across columns?
    
    # Fill in: compute probabilities
    #P = num ./ dem #won't run at the moment. fix it. 
    P = num ./ repeat(dem, 1, J)
    
    # Fill in: compute negative log-likelihood
     loglike = -sum(bigY .* log.(P))
    
    return loglike
end

#---------------------------------------------------
# Question 2: Nested Logit
#---------------------------------------------------

function nested_logit_with_Z(theta, X, Z, y, nesting_structure)
    # Extract parameters
    # theta = [beta_WC (3 elements), beta_BC (3 elements), lambda_WC, lambda_BC, gamma]
    alpha = theta[1:end-3]      # first 6 elements (3 for WC, 3 for BC)
    lambda = theta[end-2:end-1] # lambda_WC, lambda_BC
    gamma = theta[end]          # coefficient on Z
    
    # Constrain lambda parameters to be positive to ensure well-behaved nested logit
    lambda = abs.(lambda) .+ 0.01  # ensure lambda > 0
    
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Create coefficient matrix for nested structure
    # First K columns for WC nest, next K for BC nest, zeros for Other
    bigAlpha = zeros(K, J)
    bigAlpha[:, nesting_structure[1]] = repeat(reshape(alpha[1:K], K, 1), 1, length(nesting_structure[1]))
    bigAlpha[:, nesting_structure[2]] = repeat(reshape(alpha[K+1:2K], K, 1), 1, length(nesting_structure[2]))
    
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)
    lidx = zeros(T, N, J)  # linear index for each choice, what's in the exp() term. our u_j
    
    # Fill in: compute linear indices for each choice
    for j = 1:J
        utility = X*bigAlpha[:,j] .+ gamma * (Z[:,j] .- Z[:,J])
        
        if j in nesting_structure[1]  # White collar
            lidx[:,j] = exp.(clamp.(utility ./ lambda[1], -500, 500))  # clamp to prevent overflow
        elseif j in nesting_structure[2]  # Blue collar
            lidx[:,j] = exp.(clamp.(utility ./ lambda[2], -500, 500))  # clamp to prevent overflow
        else  # Other
            lidx[:,j] = ones(N)  # exp(0) = 1
        end
    end
    
    # Fill in: compute numerators using nested logit formula
    for j = 1:J
        if j in nesting_structure[1]
             wc_sum = sum(lidx[:,nesting_structure[1]], dims = 2)
             num[:,j] = lidx[:,j] .* (max.(wc_sum, 1e-10) .^ (lambda[1]-1))
        elseif j in nesting_structure[2]
             bc_sum = sum(lidx[:,nesting_structure[2]], dims = 2)
             num[:,j] = lidx[:,j] .* (max.(bc_sum, 1e-10) .^ (lambda[2]-1))
        else
             num[:,j] = lidx[:,j]
        end
    end
    dem = sum(num, dims=2)
    
    # Fill in: compute probabilities and log-likelihood
    P = num ./ max.(dem, 1e-10)  # prevent division by zero
    P = max.(P, 1e-10)  # ensure probabilities are not zero for log
    
    loglike = -sum(bigY .* log.(P))
    
    # Return a large positive number if likelihood is not finite
    if !isfinite(loglike)
        return 1e10
    end
    
    return loglike
end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------

function optimize_mlogit(X, Z, y)
    # Starting values: 21 alphas + 1 gamma
    startvals = [2*rand(7*size(X,2)).-1; 0.1]
    
    # TODO: Use optimize() function
    # Hint: Use LBFGS() algorithm with g_tol = 1e-5
    
    result = optimize(theta -> mlogit_with_Z(theta, X, Z, y), 
                     startvals, LBFGS(), 
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    
    return result.minimizer
end

function optimize_nested_logit(X, Z, y, nesting_structure)
    # Starting values: 6 alphas + 2 lambdas + 1 gamma
    # Use smaller starting values for better numerical stability
    startvals = [0.1*randn(2*size(X,2)); 0.8; 0.8; 0.01]
    
    # TODO: Use optimize() function for nested logit
    
    result = optimize(theta -> nested_logit_with_Z(theta, X, Z, y, nesting_structure), 
                     startvals, LBFGS(), 
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    
    return result.minimizer
end

#---------------------------------------------------
# Main Function (Question 4)
#---------------------------------------------------

function allwrap()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
    X, Z, y = load_data(url)
    
    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))
    
    # Estimate multinomial logit
    println("\n=== MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)
    
    # TODO: Estimate nested logit
    println("\n=== NESTED LOGIT RESULTS ===")
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]  # WC and BC occupations
    nlogit_theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
    println("Estimates: ", nlogit_theta_hat)
end

# Uncomment to run
 #allwrap()