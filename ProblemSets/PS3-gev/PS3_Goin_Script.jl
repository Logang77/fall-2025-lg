using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS3_Goin_Source.jl")

allwrap()

#question: interpret the estimated coefficient gamma
# the estimated coefficient is -0.094193. 
# Gamma represents the change in latent utility
#with a one unit change in the relative expected log wage
#in occupation j (relative to Other)
# It is interesting that gamma is negative