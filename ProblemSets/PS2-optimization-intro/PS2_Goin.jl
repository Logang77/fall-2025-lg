

function PS2()
    cd(@__DIR__)

    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS()) # what you want to optimize, the starting value, and the algorithm you want to use
    println(Optim.minimizer(result)[1])
    println(Optim.minimum(result))


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y

#standard errors
N = size(X, 1)
K = size(X, 2)
MSE = sum((y - X*bols).^2)/(N-K)
VCOV = MSE*inv(X'*X)
SE = sqrt.(diag(VCOV))
println("Standard Errors: ", SE)

println("OLS closed form: ", bols)
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit(a, X, y)
    xalpha = X * a 
    logp = xalpha .- log1p.(exp.(xalpha))
    log1mp = -log1p.(exp.(xalpha))
    loglike = -sum(y .* logp + (1 .- y) .* log1mp)
    return loglike
end

log_test_1 = optimize(a -> logit(a, X, y), zeros(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000, show_trace=true))

    println(log_test_1.minimizer)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
df.white = df.race.== 1
    alpha_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    
    println(coef(alpha_glm))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

#clean df
df = dropmissing(df, :occupation)

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, y)
    K = size(X, 2)
    N = length(y)
    J = 7 
    
    bigY = zeros(N, J)
    for i in 1:N
        bigY[i, Int(y[i])] = 1
    end
    
    alpha_mat = reshape(alpha, K, J-1)
    bigAlpha = [alpha_mat zeros(K)] 
    
    num = zeros(N, J)
    dem = zeros(N)
    
    for j in 1:J
        num[:, j] = exp.(X * bigAlpha[:, j])
        dem .+= num[:, j]
    end
    
    P = num ./ dem

    loglike = -sum(bigY .* log.(P))
    
    return loglike
end

alpha_init = zeros(size(X,2) * 6) 

log_test_2 = optimize(alpha -> mlogit(alpha, X, y), alpha_init, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true))

    alpha_hat_mle = log_test_2.minimizer
    alpha_mat = reshape(alpha_hat_mle, size(X,2), 6)
    println(alpha_mat)
    
    
    return Dict(
        "q1_minimizer" => Optim.minimizer(result),
        "q1_minimum" => Optim.minimum(result),
        "q2_ols" => bols,
        "q2_ols_se" => SE,
        "q3_logit" => log_test_1.minimizer,
        "q4_glm" => coef(alpha_glm),
        "q5_mlogit" => alpha_hat_mle
    )
end


results = PS2()
