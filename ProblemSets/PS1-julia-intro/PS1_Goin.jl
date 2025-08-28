using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

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