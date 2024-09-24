#-------------------------------------------------------------
# Author : Kushal 
# Date : September 
# Description : Solves the Hopenhayn Rogerson Model 
#-------------------------------------------------------------
using Optim, Parameters, Plots, LinearAlgebra


# Create data structures for model parameters, value functions, etc ------------------
@with_kw struct primitives
    β::Float64 = 0.8
    θ::Float64 = 0.64
    A::Float64 = 1/200.0
    c_f::Float64 = 10.0
    c_e::Float64 = 5.0

    s_grid::Vector{Float64} = [3.98*10^-4, 3.58, 6.82, 12.18, 18.79]
    v_dist::Vector{Float64} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]
    Π::Array{Float64, 2} = [0.6598 0.2600 0.0416 0.0331 0.0055;
                           0.1997 0.7201 0.0420 0.0326 0.0056;
                           0.2000 0.2000 0.5555 0.0344 0.0101;
                           0.2000 0.2000 0.2502 0.3397 0.0101;
                           0.2000 0.2000 0.2500 0.3400 0.0100]

    ns::Int64 = length(s_grid)
end

@with_kw mutable struct results 
    p::Float64 # price 
    μ::Vector{Float64} # stationary distribution of firms who choose to produce 
    M::Float64 # mass of entrants 
    α::Float64 # variance parameter for EV shock
    W_i::Vector{Float64} # value function of incumbents
    x_pr::Vector{Float64} # enter/exit choice 
end 

function Initialize()
    prim = primitives()

    p = 0.5
    μ = ones(prim.ns)
    M = 1.5
    W_i = zeros(prim.ns)  
    x_pr = Vector{Int64}(undef, prim.ns)
    α = 1.0

    res = results(p, μ, M, α, W_i, x_pr)
    prim, res
end 

# Helper functions

function π_incumbents(θ, c_f, p, s)
    """ Find profits for incumbents"""
    profits = ((θ*p.*s).^(1 / (1 - θ))).*(1/θ - 1) .- p*c_f
    return profits 
end 


function bellman(prim::primitives, res::results)
    """ Do one iteration of bellman T operator """

    @unpack β, ns, s_grid, v_dist, Π, θ, c_f = prim 
    @unpack p, W_i, x_pr, α = res
    γ =  MathConstants.eulergamma
    W_i_pr = zeros(ns) 
    for s_index = 1:ns
        s = s_grid[s_index]
        trans_probs = Π[s_index, :]

        W_continue = π_incumbents(θ, c_f, p, s) + β* trans_probs'*W_i
        W_exit = π_incumbents(θ, c_f, p, s)

        W_i_pr[s_index] = γ/α + (1/α)*log(ℯ, exp(α*W_continue) + exp(α*W_exit)  ) 
        x_pr[s_index] = exp(α*W_exit)/ (exp(α*W_exit ) + exp(α*W_continue))
    end 
    return W_i_pr
end 


function w_iterate(res::results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    while err > tol
        w_next = bellman(prim, res)
        err = maximum(abs.(w_next - res.W_i))
        res.W_i = w_next
    end 
end 

function compute_transition_matrix(prim::primitives, res::results)
    @unpack ns, s_grid, Π, v_dist = prim 
    @unpack x_pr, M, μ = res

    trans_mat = zeros(Float64, ns, ns)

    for s_index = 1:ns 
        exit_choice = x_pr[s_index]
        for s_pr = 1:ns 
            #mass_incumbents = (1 - exit_choice)*Π[s_index, s_pr]*μ[s_index]
            #mass_entrants = (1 - exit_choice)*Π[s_index, s_pr]*M*v_dist[s_index]
            trans_mat[s_index, s_pr] = (1 - exit_choice)*Π[s_index, s_pr]
        end 
    end 
    return trans_mat
end 

function stationary_distribution(P::Matrix{Float64}; tol::Float64 = 1e-10, max_iter::Int = 10000)
    
     ns = size(P,1)
     μ = ones(ns)
     μ_next = copy(μ)
     
     for iter in 1:max_iter
        μ_next = P'* (μ + res.M*prim.v_dist) 
        if norm(μ_next - μ) < tol
            return μ_next
        end 
        μ = μ_next
    end 
end

function solve_stationary_distribution(prim::primitives, res::results)
    """ Solves for the stationary distribution given the policy function. """
    P = compute_transition_matrix(prim, res)  
    μ = stationary_distribution(P)  

    res.μ = vec(μ)
end

function invisible_hand(res::results, prim::primitives; tol = 0.001)

    EC = sum(res.W_i .* prim.v_dist)/res.p - prim.c_e

    if EC > tol 
        res.p = res.p - 0.1*(1- res.p)/2
        return false 
    elseif EC < -1*tol 
        res.p = res.p + 0.1*(1-res.p)/2
        return false 
    elseif abs(EC) < tol
        res.p = res.p 
        return true 
    end
end


function excess_labor(prim::primitives, res::results)
    """ Computes excess labor demand """

    @unpack  s_grid, v_dist, θ, c_f, A = prim 
    @unpack p,  M , μ = res

    # compute industry profits 
    a = π_incumbents(θ, c_f, p, s_grid)
    industry_profits = sum(π_incumbents(θ, c_f, p, s_grid) .* μ) + M*sum(π_incumbents(θ, c_f, p, s_grid) .* v_dist)

    # compute HH labor supply (calculated offline)
    N_s = 1/A - industry_profits

    # Compute firm labor demand 
    N_d = ( θ*p .* s_grid ).^(1/(1 - θ))
    L_d = sum(N_d .* μ) + M*sum(N_d .* v_dist)

    LMC = L_d - N_s
    return LMC
end

function labor_invisible_hand(prim::primitives, res::results; tol = 0.1)
    """ Iterate on (M, mu )""" 
    LMC = excess_labor(prim, res)
    println(res.M)
    
    if LMC > tol
        res.M = res.M - 0.1*(5 - res.M)/2
        return false 
    elseif  LMC < -1*tol
        res.M = res.M + 0.1*(5 - res.M)/2
        return false 
    elseif abs(LMC) < tol 
        res.M = res.M
        return true 
    end  
end

function solve_model(prim::primitives, res::results)
   
    #=
    conv = false 
    while conv == false 
        w_iterate(res)
        conv = invisible_hand(res,prim)
    end 
    =#
    conv = false 
    while conv == false 
        w_iterate(res)
        solve_stationary_distribution(prim ,res)
        conv = invisible_hand(res, prim)  & labor_invisible_hand(prim, res)
    end 

end 


# Solve the EV Shock model for alpha = 1  -----------------------------------
prim, res = Initialize()
solve_model(prim, res)
res.α = 1.0

# main results for table 

mass_incumbents = sum((1 .- res.x_pr) .* res.μ)
mass_exits = sum(res.x_pr .* res.μ)
N_d = ( prim.θ*res.p .* prim.s_grid ).^(1/(1 - prim.θ))
L_d = sum(N_d .* res.μ) + res.M*sum(N_d .* prim.v_dist)
L_d_incumbents = sum((1 .- res.x_pr) .* N_d .* res.μ)
L_d_entrants = res.M*sum(N_d .* prim.v_dist)

println("price = ", res.p)
println("mass incumbents = ", mass_incumbents)
println("mass entrants = ", res.M)
println("mass exits = ",mass_exits )
println("Aggregate Labor Demand = ", L_d)
println("Incumbent Labor Demand = ", L_d_incumbents)
println("Entrant Labor Demand = ", L_d_entrants)
println("Fraction Labor by Entrants = ", L_d_entrants/L_d)


# Solve the EV Shock model for alpha = 2  -----------------------------------
println("")
res.α = 2.0
solve_model(prim, res)

mass_incumbents = sum((1 .- res.x_pr) .* res.μ)
mass_exits = sum(res.x_pr .* res.μ)
N_d = ( prim.θ*res.p .* prim.s_grid ).^(1/(1 - prim.θ))
L_d = sum(N_d .* res.μ) + res.M*sum(N_d .* prim.v_dist)
L_d_incumbents = sum((1 .- res.x_pr) .* N_d .* res.μ)
L_d_entrants = res.M*sum(N_d .* prim.v_dist)

println("price = ", res.p)
println("mass incumbents = ", mass_incumbents)
println("mass entrants = ", res.M)
println("mass exits = ",mass_exits )
println("Aggregate Labor Demand = ", L_d)
println("Incumbent Labor Demand = ", L_d_incumbents)
println("Entrant Labor Demand = ", L_d_entrants)
println("Fraction Labor by Entrants = ", L_d_entrants/L_d)




