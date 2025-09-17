using Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff

cd(@__DIR__)

#tests
println("Running unit tests")
include("PS2_Goin_tests.jl")

# main file
println("\nRunning main file")
include("PS2_Goin.jl")