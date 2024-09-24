#-------------------------------------------------------------
# Author : Kushal 
# Date : September 
# Description : Solves the Hopenhayn Rogerson Model 
#-------------------------------------------------------------
using Optim, Parameters, Plots, LinearAlgebra



# Helper functions


function bellman_ev(prim::primitives, res::results)
    """ Do one iteration of bellman T operator """

    @unpack β, ns, s_grid, v_dist, Π, θ = prim 
    @unpack p, W_i, x_pr, α, c_f = res
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


function w_iterate(prim::primitives, res::results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    while err > tol
        w_next = bellman_ev(prim, res)
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

    @unpack  s_grid, v_dist, θ, A = prim 
    @unpack p,  M , μ , c_f = res

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

function solve_model_ev(prim::primitives, res::results)
   
    #=
    conv = false 
    while conv == false 
        w_iterate(res)
        conv = invisible_hand(res,prim)
    end 
    =#
    conv = false 
    while conv == false 
        w_iterate(prim, res)
        solve_stationary_distribution(prim ,res)
        conv = invisible_hand(res, prim)  & labor_invisible_hand(prim, res)
    end 

end 







