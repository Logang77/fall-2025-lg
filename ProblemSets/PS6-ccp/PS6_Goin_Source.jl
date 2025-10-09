#=
Problem Set 6 - Starter Code
ECON 6343: Econometrics III
Rust (1987) Bus Engine Replacement Model - CCP Estimation

This starter code provides the structure for implementing the dynamic discrete
choice estimation using Conditional Choice Probabilities (CCPs).

Instructions:
- Fill in the sections marked with TODO
- Test your functions as you build them
- Use the structure provided but feel free to refactor if needed
=#



#========================================
QUESTION 1: Data Loading and Reshaping
========================================#

"""
    load_and_reshape_data(url::String)

Load the bus data from PS5 and reshape to long panel format.

# Arguments
- `url::String`: URL to the CSV file (second CSV from PS5)

# Returns
- `df_long::DataFrame`: Long panel format with columns for bus ID, time, and decision variables


"""
function load_and_reshape_data(url::String)
    # TODO: Implement data loading
    # Hint: Use CSV.read() and HTTP.get()
   df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))
    
    #---------------------------------------------------
    # Reshape from wide to long (done twice because 
    # stack() requires doing one variable at a time)
    #---------------------------------------------------
    
    # First reshape the decision variable (Y1-Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                      :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfy_long, Not(:variable))
    
    # Next reshape the odometer variable (Odo1-Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, 
                      :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15,
                      :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfx_long, Not(:variable))
    
    # Join reshaped dataframes back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    
    return df_long
end


#========================================
QUESTION 2: Flexible Logit Estimation
========================================#

"""
    estimate_flexible_logit(df::DataFrame)

Estimate a flexible logit model with fully interacted terms up to 7th order.

# Arguments
- `df::DataFrame`: Long panel data frame

# Returns
- Fitted GLM model object

# TODO:
- Create squared terms for continuous variables
- Use GLM with formula syntax to specify fully interacted model
- Remember: Odometer * RouteUsage * Branded * time means all interactions up to the product of all four
"""
function estimate_flexible_logit(df::DataFrame)


    flex_logit = glm(@formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time),
     df, Binomial(), LogitLink())

    
    return flex_logit
end


#========================================
QUESTION 3: CCP-Based Estimation
========================================#

"""
    construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)

Construct the state space grid for all possible states.

# Arguments
- `xbin::Int`: Number of mileage bins
- `zbin::Int`: Number of route usage bins  
- `xval::Vector`: Grid points for mileage
- `zval::Vector`: Grid points for route usage

# Returns
- `state_df::DataFrame`: Data frame with all possible state combinations

# TODO:
- Create vectors using kron(ones(zbin), xval) for Odometer
- Create vectors using kron(ones(xbin), zval) for RouteUsage
- Initialize Branded and time to zeros
"""
function construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)
    # TODO: Implement state space construction
    # Hint: kron() is the Kronecker product function
    
    state_df = DataFrame(
        # Odometer = ...,
        # RouteUsage = ...,
        # Branded = ...,
        # time = ...
    )
    
    return state_df
end


"""
    compute_future_values(state_df::DataFrame, 
                          flex_logit::GeneralizedLinearModel,
                          xtran::Matrix, 
                          xbin::Int, 
                          T::Int, 
                          β::Float64)

Compute future value terms using CCPs from the flexible logit.

# Arguments
- `state_df::DataFrame`: State space data frame
- `flex_logit`: Fitted flexible logit model
- `xtran::Matrix`: State transition matrix
- `xbin::Int`: Number of mileage bins
- `T::Int`: Number of time periods
- `β::Float64`: Discount factor (0.9)

# Returns
- `FV::Array{Float64,3}`: Future value array (states × brand × time)

# TODO:
- Initialize FV as zeros(size(xtran,1), 2, T+1)
- Loop over t = 2 to T
- Loop over brand states b ∈ {0,1}
- Update state_df with current t and b values
- Use predict() to get p0 (probability of replacement)
- Store -β * log(p0) in FV[:, b+1, t+1]
"""
function compute_future_values(state_df::DataFrame, 
                                flex_logit::GeneralizedLinearModel,
                                xtran::Matrix, 
                                xbin::Int, 
                                T::Int, 
                                β::Float64)
    
    # TODO: Initialize the future value array
    # FV = zeros(???, 2, T+1)
    
    # TODO: Nested loops over time and brand
    # for t in 2:T
    #     for b in 0:1
    #         # Update state_df
    #         # Compute p0 using predict()
    #         # Store -β * log.(p0) in FV
    #     end
    # end
    
    return FV
end


"""
    compute_fvt1(df_long::DataFrame, 
                 FV::Array{Float64,3},
                 xtran::Matrix,
                 Xstate::Vector,
                 Zstate::Vector,
                 xbin::Int)

Map future values from state space to actual data.

# Arguments
- `df_long::DataFrame`: Original long data frame
- `FV::Array{Float64,3}`: Future value array
- `xtran::Matrix`: State transition matrix
- `Xstate::Vector`: Mileage state for each observation
- `Zstate::Vector`: Route usage state for each observation  
- `xbin::Int`: Number of mileage bins

# Returns
- `FVT1::Vector`: Future value term for each observation in long format

# TODO:
- Initialize FVT1 matrix to store results
- Loop over observations i and time periods t
- Compute row indices in xtran based on Xstate[i] and Zstate[i]
- Calculate FVT1[i,t] = (xtran[row1,:] - xtran[row0,:])'* FV[row0:row0+xbin-1, B[i]+1, t+1]
- Convert to long format vector
"""
function compute_fvt1(df_long::DataFrame, 
                      FV::Array{Float64,3},
                      xtran::Matrix,
                      Xstate::Vector,
                      Zstate::Vector,
                      xbin::Int)
    
    # Get dimensions
    N = length(unique(df_long.bus_id))  # Adjust column name as needed
    T = # TODO: determine T from data
    
    # TODO: Initialize FVT1
    # FVT1 = zeros(N, T)
    
    # TODO: Loop over observations and time
    # for i in 1:N
    #     for t in 1:T
    #         # Compute row0 and row1 indices
    #         # row0 = (Xstate[i]-1)*zbin + Zstate[i]
    #         # row1 = (Xstate[i])*zbin + Zstate[i]
    #         
    #         # Compute future value contribution
    #         # FVT1[i,t] = ...
    #     end
    # end
    
    # TODO: Convert to long format
    # fvt1_long = FVT1'[:]
    
    return fvt1_long
end


"""
    estimate_structural_params(df_long::DataFrame, fvt1::Vector)

Estimate structural parameters θ using GLM with offset.

# Arguments
- `df_long::DataFrame`: Long panel data with decision variable
- `fvt1::Vector`: Future value term to use as offset

# Returns
- Fitted GLM model with structural parameters

# TODO:
- Add fvt1 as a column to df_long
- Estimate logit with Odometer and Branded as regressors
- Use offset argument to include future value with coefficient = 1
"""
function estimate_structural_params(df_long::DataFrame, fvt1::Vector)
    # TODO: Add future value to data frame
    # df_long = @transform(df_long, fv = fvt1)
    
    # TODO: Estimate structural model
    # theta_hat = glm(@formula(Y ~ Odometer + Branded), 
    #                 df_long, 
    #                 Binomial(), 
    #                 LogitLink(), 
    #                 offset=df_long.fv)
    
    return theta_hat
end


#========================================
MAIN WRAPPER FUNCTION
========================================#

"""
    main()

Main wrapper function to estimate the Rust model using CCPs.
This function calls all the helper functions in sequence.

# TODO:
- Call each function in order
- Pass results between functions appropriately
- Print results at the end
"""
function main()
    println("="^60)
    println("Rust (1987) Bus Engine Replacement Model - CCP Estimation")
    println("="^60)
    
    # Set parameters
    β = 0.9  # Discount factor
    
    # TODO: Define grid parameters (xbin, zbin, xval, zval)
    # These should match what you used in PS5

    include("create_grids.jl")
    
    # Step 1: Load and reshape data
    println("\nStep 1: Loading and reshaping data...")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv" 
    df_long = load_and_reshape_data(url)
    
    
    # Step 2: Estimate flexible logit
    println("\nStep 2: Estimating flexible logit...")
    flexlogitresults = estimate_flexible_logit(df_long)
    println(flexlogitresults)
    return
    
    # Step 3a: Construct state transition matrices
    println("\nStep 3a: Constructing state transition matrices...")
    # TODO: Reuse code from PS5 to construct xtran
    
    # Step 3b: Construct state space
    println("\nStep 3b: Constructing state space...")
    # TODO: Call construct_state_space()
    
    # Step 3c: Compute future values
    println("\nStep 3c: Computing future values...")
    # TODO: Call compute_future_values()
    
    # Step 3d: Map to actual data
    println("\nStep 3d: Mapping future values to data...")
    # TODO: Call compute_fvt1()
    
    # Step 3e: Estimate structural parameters
    println("\nStep 3e: Estimating structural parameters...")
    # TODO: Call estimate_structural_params()
    
    # Print results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    # TODO: Print coefficient estimates and standard errors
    
    return nothing
end

# Run the estimation (uncomment when ready to test)
# @time main()