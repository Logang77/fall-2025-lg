using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions, Test

cd(@__DIR__) # change directory to the location of this script

# Set random seed
Random.seed!(1234);

function q1()

    # ---------------------------------------[Problem 1]---------------------------------------
    #i

    #Draw uniform random numbers between -5 and 10 into a 10x7 matrix 
    A = rand(Uniform(-5,10), 10, 7)

    #ii. B10×7 - random numbers distributed N (−2,15) [st dev is 15] 
    B = rand(Normal(-2,15), 10, 7)

    #iii. C5×7 - the first 5 rows and first 5 columns of A and the last two columns and first 5 rows of B
    C = hcat(A[1:5, 1:5], B[1:5, 6:7])

    #iv. D10×7 - where Di,j = Ai,j if Ai,j ≤0, or 0 otherwise
    D = min.(A, 0)

    #---------------------------------------[Part B]---------------------------------------
    #use a built in julia function to list the number of elements in A 
    num_elements_A = length(A)

    #---------------------------------------[Part C]---------------------------------------
    #use a series of built in julia functions to list the number of unique elements in D 
    num_unique_D = length(unique(D))

    #---------------------------------------[Part D]---------------------------------------
    #Using the reshape() function, create a new matrix called E which is the ‘vec’ operator2 applied to B
    E = reshape(B, length(B), 1)
    E = B[:] #both work!

    #---------------------------------------[Part E]---------------------------------------
    #Create a new array called F which is 3-dimensional and contains A in the first columnn of the third dimension and B in the second column of the third dimension
    F = cat(A, B, dims = 3)

    #---------------------------------------[Part F]---------------------------------------
    #Use the permutedims() function to twist F so that it is now F2x10x7. save new matrix as F
    F = permutedims(F, (3, 1, 2))

    #---------------------------------------[Part G]---------------------------------------
    # create a matrix G which is equal to the kronecker product of B and C.

    G = kron(B, C) # the kronecker makes the new matrix as the product of the colunms and the product of the rows

    #what about if we try C and F?
    # G2 = kron(C, F)  you will recieve an error becaue the dimensions do not align

    #---------------------------------------[Part H]---------------------------------------
    #Save the matrices A, B, C, D, E, F and G as a .jld file named matrixpractice
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    #---------------------------------------[Part I]---------------------------------------
    # save the matrices A, B, C, D as a jld file named first matrix
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)

    #---------------------------------------[Part J]---------------------------------------
    # export C as a csv file names Cmatrix.csv
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    #---------------------------------------[Part K]---------------------------------------
    #Export D as a tab-delimited .dat file called Dmatrix. You will first need to transform D into a DataFrame
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim = '\t')

    #---------------------------------------[Part L]---------------------------------------
    #Wrap a function definition around all of the code for question 1. Call the function q1().
    #The function should have 0 inputs and should output the arrays A, B, C and D. At the
    #very bottom of your script you should add the code A,B,C,D = q1() 

return A, B, C, D
end

# Call the function



#---------------------------------------[Problem 2, a]---------------------------------------
function q2(A, B, C)
    AB = zeros(size(A))
    for row in 1:size(A, 1)
        for col in 1:size(A, 2)
            AB[row, col] = A[row, col] + B[row, col]
        end
    end
    return AB

    #not working for some reason this also works
    #AB = A .* B

    #---------------------------------------[Problem 2, b]---------------------------------------
    #Write a loop that creates a column vector called Cprime which contains only the ele-
    #ments of C that are between -5 and 5 (inclusive). Create a vector called Cprime2 which
    #does this calculation without a loop.
    Cprime = []
    for c in 1:size(C, 2)
        for r in 1:size(C, 1)
            if C[r, c] >= -5 && C[r, c] <= 5
                push!(Cprime, C[r, c])
            end
        end
    end

    #---------------------------------------[Problem 2, c]---------------------------------------
    X = zeros(15_169, 6, 5) #15_169 is 15,169 where _ acts as the comma
    N = size(X, 1)
    K = size(X, 2)
    T = size(X, 3)

    #ordering of the second eimension:
    #dummy variable
    #continuous variable(normal)
    #normal
    #binomial ("discrete" normal)
    #another binaomial
    for i in axes(X, 1)
        X[i, 1, :] .= 1.0
        X[i, 5, :] .= rand(Binomial(20, 0.6))
        X[i, 6, :] .= rand(Binomial(20, 0.5))
        for t in axes(X,3)
            X[i, 2, t] = rand() <= .75 * (6-t)/5
            X[i, 3, t] = rand(Normal(15 + t - 1, 5*t-1))
            X[i, 4, t] = rand(Normal(pi * (6 - t), 1/exp(1)))
        end

    end
    #---------------------------------------[Problem 2, d]---------------------------------------
    #comprehensions practice
    β = zeros(K, T)
    β[1, :] = [1 + 0.25 * (t-1) for t in 1:T]
    β[2, :] = [log(t) for t in 1:T]
    β[3, :] = [-sqrt(t) for t in 1:T]
    β[4, :] = [exp(t) - exp(t+1) for t in 1:T]
    β[5, :] = [t for t in 1:T]
    β[6, :] = [t/3 for t in 1:T]
    Y = zeros(N, T)
    Y = [X[:, :, t] * β[:, t] .+ rand(Normal(0, 0.36), N) for t in 1:T]
    Y = [x[:, :, t] * β[:, t] .+ rand(Normal(0, 0.36), N) for t in 1:T]
    



    return nothing
end  

function q3()
    #----------------------------------------[problem 3, a]---------------------------------------
    #load the dataset from the file nlsw88.csv into julia as a DataFrame
    df = DataFrame(CSV.File("nlsw88.csv"))
    @show df[1:5, :] #show the first 5 rows of the dataframe
    @show typeof(df[:, :grade])
    #save as cleaned csv file
    CSV.write("nlsw88_cleaned.csv", df)

    #----------------------------------------[problem 3, b]---------------------------------------
    #percentage never married
    @show mean(df[:, :never_married])

    #----------------------------------------[problem 3, c]---------------------------------------
    @show freqtable(df[: , :race])

    #----------------------------------------[problem 3, d]---------------------------------------
    #create a matrix called summary stats that lists the stats
    vars = names(df)
    summary_stats = describe(df)
    @show summary_stats

    #----------------------------------------[problem 3, e]---------------------------------------
    # cross tabulation of industry and occupation
    @show freqtable(df[:, :industry], df[:, :occupation])

    #----------------------------------------[problem 3, f]---------------------------------------
    # get the mean within the groups of industry and occupation categories
    df_sub = df[:, [:industry, :occupation, :wage]]
    grouped = groupby(df_sub, [:industry, :occupation])
    mean_wage = combine(grouped, :wage => mean => :mean_wage)
    @show mean_wage

    
    return nothing
end



#---------------------------------------[problem 4, B]---------------------------------------
    #---------------------------------------[problem 4, C]---------------------------------------
"""
function matrixops(A, B)
    performs the following operations matrices A and B:
    1. computes the element wise product of A and B
    2. computes the matrix product of A transpose and b
    3. computes the sum of all elements of the sum of A and B
"""
    
function matrixops(A::Array{Float64}, B::Array{Float64})
    #part e check dimensionalityC
    if size(A) != size(B)
        error("matrices A and B must have the same dimensions")
    end

    #(i) eleemen wise product of A and b
    out1 = A .* B
    #(ii) matrix product of A' and b
    out2 = A' * B
    #(iii) sum of all elements of sum of A and B 
    out3 = sum(A+B)
    return nothing
end
# load firstmatrix.jld
    #---------------------------------------[problem 4, A]---------------------------------------
function q4()
    #three ways to load the .jld file
    @load "matrixpractice.jld"
    #load("matrixpractice.jld", "A", "B", "C", "D") #specify which matrices to load
    #@load "matrixpractice.jld" A B C D #specify which matrices to load
    #part d of question 4
    matrixops(A, B)

    #part f of question 4
    try
        matrixops(A, C) #this should throw an error because the dimensions do not align
    catch e 
        @show e
    end

    #part g of question 4
    #read in csv
    nlsw88    = DataFrame(CSV.File("nlsw88_cleaned.csv"))
    ttl_exp = convert(Array, nlsw88.ttl_exp)
    wage    = convert(Array, nlsw88.wage)
    matrixops(ttl_exp, wage)


    return nothing
end


#call the function from q1
A, B, C, D = q1()

# call the function from q2
q2(A, B, C)

#call the function from q3
q3()

#call the function from q4
q4()



