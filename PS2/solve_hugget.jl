#-------------------------------------------------------------
# Author : Kushal 
# Date : September 
# Description : Solves the hugget model 
#-------------------------------------------------------------
using Optim, Parameters, Plots, Interpolations , LinearAlgebra , LaTeXStrings, StatsBase
using Trapz

@with_kw struct params
    # choose parameters
    β::Float64 = 0.9932 # discount factor
    α::Float64 = 1.5 # coeffiecient of relative risk aversion
end

@with_kw struct model_grids
    ns::Int64 = 2
    s_grid::Vector{Float64} = [1.0, 0.5] # exogenous employment state
    Π::Array{Float64, 2} = [0.97 0.03; 0.5 0.5] # markov transition probabilities

    a_min::Float64 = -2.0
    a_max::Float64 = 5.0
    na::Int64 = 501
    a_grid::Array{Float64, 1} = collect(range(start = a_min, length = na, stop = a_max)) # grid of asset choices
end

mutable struct results 
    val_func::Array{Float64,2}
    pol_func::Array{Float64,2}
    μ_distr::Array{Float64,2}
    q::Float64 
end

function Initialize()
    parameters = params() #initialize parameters
    grids = model_grids()
    val_func = zeros(grids.na, grids.ns) 
    pol_func = zeros(grids.na, grids.ns) 
    μ_distr = ones(grids.na, grids.ns) ./ (grids.ns*grids.na)
    q = 0.995 # guess 
    res = results(val_func, pol_func, μ_distr, q) 
    grids, res, parameters 
end

function cont_val(ap, s, a, q, v_e, v_u, p, parameters)
    """ calculates V(a, s ; q) = u(c) + β*E[V(a',s';q)] """

    @unpack β, α = parameters
    c = s + a - q*ap
    if c <= 0
        u = -Inf
    else 
        u = ( c^(1 - α) - 1 )/ (1 - α)
    end 
    Ev = p[1]*v_e(ap) + p[2]*v_u(ap)
    v = -1*(u + β*Ev)
    return v
end 

function bellman(parameters::params, grids::model_grids, res::results)
    """ Apply the T[] operator , T[v] = max u + β*E[V] """ 
    @unpack α, β = parameters
    @unpack s_grid, Π, a_grid, na, ns, a_min, a_max = grids
    @unpack q = res

    v_next = zeros(na, ns)

    # Interpolate the value function 
    v_e = linear_interpolation(a_grid, res.val_func[:,1])
    v_u = linear_interpolation(a_grid, res.val_func[:,2])

    for s_index = 1:ns
        s = s_grid[s_index]
        p = Π[s_index, :]
        for a_index = 1:na
            a = a_grid[a_index]
            a_hat = min((s + a)/q, a_max) # c ≥ 0 constraint

            optim_results = optimize(ap -> cont_val(ap, s, a, q, v_e, v_u, p, parameters), a_min, a_hat)
            a_star = optim_results.minimizer
            v_star = -1*optim_results.minimum

            v_next[a_index, s_index] = v_star
            res.pol_func[a_index, s_index] = a_star
        end
    end
    return v_next
end 

function v_iterate(parameters::params, grids::model_grids, res::results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    """ Apply the T[] operator (bellman function) until |Tv - v| < ε """
    n = 0 
    while err>tol  
        v_next = bellman(parameters, grids, res) 
        err = maximum(abs.(v_next .- res.val_func)) 
        res.val_func = v_next 
        n+=1
    end
end

function compute_transition_matrix(grids::model_grids, res::results)
    """ Computes the transition matrix based on policy function and Π . """
    @unpack ns, na, a_grid, Π = grids

    # Initialize the transition matrix of size (ns*na) x (ns*na)
    P = zeros(Float64, ns * na, ns * na)

    for s_i in 1:ns
        for a_i in 1:na
            ap = res.pol_func[a_i, s_i]  # get next period's asset from policy function
            ap_id = argmin(abs.(ap .- a_grid))

            for sp in 1:ns
                P[a_i + na*(s_i -1), ap_id + na*(sp - 1) ] =  Π[s_i, sp]
                #P[(a_i - 1) * ns + s_i, (a_prime_id - 1) * ns + s_prime] = Π[s_i, s_prime]
            end
        end
    end
    return P
end

function stationary_distribution(P::Matrix{Float64}; tol::Float64 = 1e-10, max_iter::Int = 10000)
    """ Computes the stationary distribution from the transition matrix. """
    na_ns = size(P, 1)
    μ = ones(na_ns) / na_ns  # Start with a uniform distribution
    μ_next = copy(μ)

    for iter in 1:max_iter
        μ_next = P' * μ
        if norm(μ_next - μ) < tol
            return μ_next / sum(μ_next)  # normalize distribution
        end
        μ = μ_next
    end
end

function solve_stationary_distribution(grids::model_grids, res::results)
    """ Solves for the stationary distribution given the policy function. """
    P = compute_transition_matrix(grids, res)  
    μ = stationary_distribution(P)  

    # Reshape μ into (na, ns) format
    res.μ_distr = reshape(μ, (grids.na, grids.ns))
end


function invisible_hand(grids::model_grids, res::results; tol::Float64 = 1e-3)
    @unpack μ_distr, pol_func, q = res

    ED_q = sum(μ_distr .* pol_func)
    println("Excess Demand is ", ED_q)
    step_size = 0.01

    if ED_q > tol
        res.q = res.q + step_size*(1 - res.q)/2
        return false 
    elseif ED_q < -1*tol
        res.q = res.q - step_size*(1 - res.q)/2
        return false 
    elseif abs(ED_q ) < tol
        return true
    end 
end 


function solve_model(parameters::params, grids::model_grids, res::results)

   conv = false 
   n = 0

   while conv == false
        v_iterate(parameters, grids, res)
        solve_stationary_distribution(grids, res)
        conv = invisible_hand(grids, res)
        n+=1
   end
   println("Invisible hand sets q = ", res.q)

end

function make_lorenz_data(grids::model_grids, res::results)
    @unpack na, ns, a_grid, s_grid = grids

    earnings = vec(s_grid' .+ a_grid)
    
   # Sort earnings and distribution by earnings
   indices = sortperm(earnings)
   earnings_sorted = earnings[indices]
   μ_sorted =  vec(res.μ_distr)[indices]

   # Compute cumulative population share and cumulative earnings share
   μ_cdf = cumsum(μ_sorted) / sum(μ_sorted)
   cumulative_earnings = cumsum(earnings_sorted .* μ_sorted) / sum(earnings_sorted .* μ_sorted)
    return μ_cdf, cumulative_earnings
end

function compute_gini(grids::model_grids, res::results)
    μ_cdf, cumulative_earnings = make_lorenz_data(grids, res)

    area = trapz(μ_cdf, cumulative_earnings)

    gini = 1 - 2 * area
    return gini
end

function compute_consumption_equiv(res::results, parameters::params, grids::model_grids)
    @unpack na, ns = grids
    @unpack val_func = res
    @unpack α , β = parameters

    wf_fb = (0.9715^(1 - α) - 1 ) / ((1 - α)*(1 - β))
    λ = zeros(na, ns)

    α_β_inv = 1 / ((1 - α)*(1 - β))

    num = wf_fb +  α_β_inv
    den = val_func .+ α_β_inv

    frac_inside = num ./ den 

    λ = frac_inside.^(1 / (1 - α)) .- 1 

    return λ
end 


# Solve the model ---------------------------------------
grids, res, parameters =  Initialize()
solve_model(parameters, grids, res)

# Make Plots for Q4 -------------------------------------

# plot policy function
policy_plot = plot(grids.a_grid, 
                   res.pol_func,
                   xlabel = "a",
                   ylabel = "g(a,s)",
                   title = "Policy Function",
                   label = [L"g(a, e)" L"g(a, u)"])
plot!(grids.a_grid,grids.a_grid,label = L"45^∘ line ",color="red",linestyle=:dash)


# plot cross sectional distribution
μ_distribution_plot = plot(grids.a_grid, 
                           res.μ_distr,
                           xlabel = "a",
                           ylabel = "μ(a,s)",
                           title = "Wealth Distribution",
                           label = [L"μ(a, e)" L"μ(a, u)"])

# make a lorenz curve 
lorenz_data = make_lorenz_data(grids, res)

lorenz_plot = plot(lorenz_data[1], 
                   lorenz_data[2],
                   xlabel = "Cumulative Population",
                   ylabel = "Cumulative Share of Wealth",
                   title = "Lorenz Curve",
                   label = "Lorenz Curve")
plot!(lorenz_data[1],lorenz_data[1],label = L"45^∘ line ",color="red")

gini = compute_gini(grids, res)


# plot consumption equivalent 

λ =  compute_consumption_equiv(res, parameters, grids)
ce_plot = plot( λ , 
                xlabel = "a",
                ylabel = L"λ(a,s)",
                title = "Consumption Equivalent",
                label = [L"λ(a,e)" L"λ(a,u)" ])


# Print Welfare Statistics 

W_FB = (0.9715^(1 - parameters.α) - 1 ) / ((1 - parameters.α)*(1 - parameters.β))
W_INC = sum(res.val_func .* res.μ_distr)
W_G = sum(res.μ_distr .* λ)

pos_wg_ind = λ .> 0 
frac_change = sum(pos_wg_ind .* res.μ_distr)


println("Aggregate welfare with complete markets is ", W_FB)
println("Aggregate welfare with incomplete markets is ", W_INC)
println("The aggregate welfare gain is ", W_G)
println("The fractions of the population who would pay for complete markets is ", frac_change)

# Save plots 
savefig(policy_plot ,"./PS2/policy_function.png")
savefig(μ_distribution_plot, "./PS2/cross_sectional_distribution.png")
savefig(lorenz_plot, "./PS2/lorenz_curve.png")
savefig(ce_plot, "./PS2/consumption_equiv.png")

