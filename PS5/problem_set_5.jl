#---------------------------------------
# Kushal 
# ECON 880 Problem Set 5 
# October 2024
#---------------------------------------

# Load Packages 
using Distributed
addprocs(7)
@everywhere using Optim, LaTeXStrings, Plots, SharedArrays, Parameters, DelimitedFiles, LinearAlgebra

# read in efficiency 
ef = float(open(readdlm, "./PS5/ef.txt"))

# Create primitives , results 

@everywhere @with_kw struct Primitives
    # Input
    ef::Vector{Float64}

    σ::Float64 = 2.0
    β::Float64 = 0.97
    θ::Float64
    γ::Float64

    δ::Float64 = 0.06

    α::Float64 = 0.36
    J_r::Int = 46
    n::Float64 = 0.011
    N::Int = 66

    na::Int = 1001
    a_min::Float64 = 0.0
    a_max::Float64 = 75

    nz::Int = 2
    π_hh::Float64 = 0.9261
    π_ll::Float64 = 0.9811

    a_grid::SharedVector{Float64} = SharedVector(collect(range(a_min, a_max, length=na)))
    z_grid::SharedVector{Float64}
    Π = SharedArray{Float64,2}([π_hh 1-π_hh; 1-π_ll π_ll])
    e_dist = SharedVector([0.2037, 0.7963])
    e_grid = SharedVector(ef)
end


@everywhere @with_kw mutable struct results
    v_f::SharedArray{Float64,3}
    g_a::SharedArray{Float64,3}
    g_l::SharedArray{Float64,3}
    ψ_distr::SharedArray{Float64,3}
    w::Float64
    r::Float64
    b::Float64
    K::Float64
    L::Float64
end

@everywhere function initialize(zh, zl, EIS, theta, eff_data)
    prim = Primitives(z_grid=vec([zh, zl]), γ=EIS, θ=theta, ef=vec(eff_data))

    v_f = SharedArray{Float64}(prim.na, prim.nz, prim.N, pids=procs())
    g_a = SharedArray{Float64}(prim.na, prim.nz, prim.N, pids=procs())
    g_l = SharedArray{Float64}(prim.na, prim.nz, prim.N, pids=procs())

    ψ_dist = SharedArray{Float64}(prim.na, prim.nz, prim.N, pids=procs())

    w = 1.05
    r = 0.05
    b = 0.20

    K = 3.5104318056194574
    L = 0.3427556050377294

    res = results(v_f, g_a, g_l, ψ_dist, w, r, b, K, L)
    return prim, res
end

@everywhere function labor_supply(prim::Primitives, res::results, age, a, a_pr, z)
    @unpack θ, γ, ef = prim
    @unpack r, w = res

    e = z * ef[age]
    l = (γ * (1 - θ) * e * w .- (1 - γ) * ((1 + r) * a .- a_pr)) ./ ((1 - θ) * w * e)
    l = clamp.(l,0,1)

    return l
end

@everywhere function crra_y(prim::Primitives, res::results, age, a, a_pr, z)
    @unpack γ, σ, ef, J_r, θ = prim
    @unpack r, w, b = res

    l = labor_supply(prim, res, age, a, a_pr, z)
    e = z * ef[age]
    c = w * (1 - θ) * e * l .+ (1 + r) * a .- a_pr

    # Initialize u with -Inf where c < 0
    u = fill(-Inf, length(c))

    # Calculate utility only where c >= 0
    pos_idx = c .>= 0
    if any(pos_idx)
        u[pos_idx] = (c[pos_idx] .^ γ .* (1 .- l[pos_idx]) .^ (1 - γ)).^(1 - σ)
        u[pos_idx] = u[pos_idx] / (1 - σ)
    end

    return u
end

@everywhere function crra_o(prim::Primitives, res::results, a, a_pr)
    @unpack γ, σ, ef, J_r = prim
    @unpack r, w, b = res

    c = (1 + r) * a + b .- a_pr

    # Initialize u with -Inf where c < 0
    u = fill(-Inf, length(c))

    # Calculate utility only where c >= 0
    pos_idx = c .>= 0
    if any(pos_idx)
        u[pos_idx] = c[pos_idx] .^ ((1 - σ) * γ) / (1 - σ)
    end

    return u
end


function gen_cohort_size(prim::Primitives)
    μ = ones(prim.N)
    for i = 2:prim.N
        μ[i] = μ[i-1]/(1 + prim.n)
    end 
    μ = μ ./ sum(μ)
    return μ
end 


function compute_transition_matrix(prim::Primitives, res::results)
    """ Computes the transition matrix based on policy function and Π . """
    @unpack nz, na, a_grid, Π , N = prim

    # Initialize the transition matrix of size (nz*na) x (nz*na) x N
    P = zeros(Float64, nz * na, nz * na, N)

    for j in 1:N
        for a_i in 1:na
            for z_i in 1:nz
                ap = res.g_a[a_i, z_i, j]  # get next period's asset from policy function
                ap_id = argmin(abs.(ap .- a_grid))  # Find the closest asset grid point

                for zp in 1:nz
                    P[a_i + na*(z_i -1), ap_id + na*(zp - 1), j] = Π[z_i, zp]
                end
            end
        end
    end 
    return P
end

function stationary_distribution(P, prim::Primitives)
    """ Computes the stationary distribution from the transition matrix. """

    @unpack e_dist, N , na, nz = prim

    na_nz = size(P, 1)
    ψ = zeros(na_nz, prim.N)
    ψ[1,1] = e_dist[1]
    ψ[na + 1, 1] = e_dist[2]

    for j in 2:N
        ψ[:,j] = P[:,:,j-1]' * ψ[:,j-1]
    end

    μ = gen_cohort_size(prim)

    ψ = ψ .* μ'
end

function solve_stationary_distribution(prim::Primitives, res::results)
    """ Solves for the stationary distribution given the policy function. """
    P = compute_transition_matrix(prim, res)  
    ψ = stationary_distribution(P, prim)  

    # Reshape ψ into (na, nz, N) format
    res.ψ_distr = reshape(ψ, (prim.na, prim.nz, prim.N))
end

function DP(prim::Primitives, res::results)
    """ Do 1 iteration of VFI """

    @unpack a_grid, na, nz, z_grid, J_r, N, β, Π = prim
    @unpack v_f, g_a, g_l = res

    # solves the geriatric problem 
    for age = N:-1:(J_r)
        @sync @distributed for i_a = 1:na
            a = a_grid[i_a]
            if age == N
                res.v_f[:, :, N] .= crra_o(prim, res, a, zeros(na))
                res.g_a[:,:, N] .= 0
            else
                res.v_f[i_a, :, age] .= maximum(crra_o(prim, res, a, a_grid) + β .* v_f[:, 1, age+1])
                i_a_max = argmax(crra_o(prim, res, a, a_grid) + β .* v_f[:, 1, age+1])
                res.g_a[i_a, :, age] .= a_grid[i_a_max]
                res.g_l[i_a, :, age] .= 0
            end
        end
    end

    # solves the workers problem 
    for age = (J_r-1):-1:1
        @sync @distributed for i_a = 1:na
            a = a_grid[i_a]
            for i_z = 1:nz
                z = z_grid[i_z]
                p_z = Π[i_z, :]
                res.v_f[i_a, i_z, age] = maximum(crra_y(prim, res, age, a, a_grid, z) + β .* (p_z[1] .* v_f[:, 1, age+1] + p_z[2] .* v_f[:, 2, age+1]))
                i_a_max = argmax(crra_y(prim, res, age, a, a_grid, z) + β .* (p_z[1] .* v_f[:, 1, age+1] + p_z[2] .* v_f[:, 2, age+1]))
                res.g_a[i_a, i_z, age] = a_grid[i_a_max]
                res.g_l[i_a, i_z, age] = labor_supply(prim, res, age, a, g_a[i_a, i_z, age], z)
            end
        end
    end
end


function fixed_point(prim::Primitives, res::results; tol = 0.001, err = 100 )
    @unpack g_a, g_l  = res 
    @unpack a_grid, z_grid, J_r , ef , na, nz, α , N, θ, δ = prim

    μ = gen_cohort_size(prim)

    # calculate prices 
    res.r = α*(res.K/res.L)^(α - 1) - δ
    res.w = (1 - α)*(res.K/res.L)^(α)
    res.b = θ*res.w*res.L/sum(μ[J_r:N])

    # find new K, L 
    DP(prim, res)
    solve_stationary_distribution(prim, res)

    K_new = 0
    for j = 1:N
        for z = 1:nz
            K_new += sum(res.ψ_distr[:,z,j] .* a_grid)
        end 
    end 

    L_new = 0 
    for j = 1:(J_r - 1)
        for z = 1:nz
            e = prim.ef[j]*prim.z_grid[z]
            L_new += sum(res.ψ_distr[:,z,j] .* (e * res.g_l[:,z,j]))
        end 
    end 

    err_K = abs(K_new - res.K)
    err_L = abs(L_new - res.L)
    err = max(err_K, err_L)

    n= 0
    while err > tol 
        n = n + 1 
        println("Iteration ", n, " error ", err, " K is ", res.K, " L is ", res.L)
        res.K = 0.75*res.K + 0.25*K_new
        res.L = 0.75*res.L + 0.25*L_new

        # calculate prices 
        res.r = α*(res.K/res.L)^(α - 1) - δ
        res.w = (1 - α)*(res.K/res.L)^(α)
        res.b = θ*res.w*res.L/sum(μ[J_r:N])

        # find new K, L 
        DP(prim, res)
        solve_stationary_distribution(prim, res)

        K_new = 0
        for j = 1:N
            for z = 1:nz
                K_new += sum(res.ψ_distr[:,z,j] .* a_grid)
            end 
        end 

        L_new = 0 
        for j = 1:(J_r - 1)
            for z = 1:nz
                e = prim.ef[j]*prim.z_grid[z]
                L_new += sum(res.ψ_distr[:,z,j] .* (e * res.g_l[:,z,j]))
            end 
        end 

        err_K = abs(K_new - res.K)
        err_L = abs(L_new - res.L)
        err = max(err_K, err_L)
    end 
    println("Converged")
end 



# run models ------------
my_table = []

#Benchmark


#=prim, res = initialize(3.0, 0.5, 0.42, 0.11, ef)
#DP(prim, res)

vf_50 = plot(prim.a_grid, res.v_f[:,1, 50], 
             xlabel = L"a",
             ylabel = L"V_{50}(a)",
             title = "Age 50 VF", legend = nothing)

g_a_20 = plot(prim.a_grid, res.g_a[:,:,20],
              xlabel = L"a",
              ylabel = L"g_{20}(a)",
              title = "Age 20 asset choice"  , label = [L"z_l", L"z_h"] )

g_savings_20 = plot(prim.a_grid, res.g_a[:,:,20] .- prim.a_grid,
              xlabel = L"a",
              ylabel = L"g_{20}(a)",
              title = "Age 20 asset choice"  , label = [L"z_l", L"z_h"] )

g_labor_20 = plot(prim.a_grid, res.g_l[:,:,20] ,
              xlabel = L"a",
              ylabel = L"g_{20}(a)",
              title = "Age 20 asset choice"  , label = [L"z_l", L"z_h"] )


savefig(vf_50, "./PS5/vf50.png")
savefig(g_a_20, "./PS5/g_a_20.png")
savefig(g_savings_20, "./PS5/g_s_20.png")
=# 

prim, res = initialize(3.0, 0.5, 0.42, 0.11, ef)
fixed_point(prim, res)


# Eliminate Social Security 
prim2, res2 = initialize(3.0, 0.5, 0.42, 0, ef)
DP(prim2, res2)
solve_stationary_distribution(prim2, res2)
fixed_point(prim2, res2; tol = 1e-3)

# No idiosyncratic risk 
prim3, res3 = initialize(0.5, 0.5, 0.42, 0.11, ef)
DP(prim3, res3)
solve_stationary_distribution(prim3, res3)
fixed_point(prim3, res3; tol = 1e-3)


# No idiosyncratic risk, no SS
prim4, res4 = initialize(0.5, 0.5, 0.42, 0, ef)
DP(prim4, res4)
solve_stationary_distribution(prim4, res4)
fixed_point(prim4, res4; tol = 1e-3)

# No idiosyncratic risk 
prim5, res5 = initialize(3.0, 0.5, 1.0, 0.11, ef)
DP(prim5, res5)
solve_stationary_distribution(prim5, res5)
fixed_point(prim5, res5)

# No idiosyncratic risk, no SS
prim6, res6 = initialize(3.0, 0.5, 1.0, 0, ef)
DP(prim6, res6)
solve_stationary_distribution(prim6, res6)
fixed_point(prim6, res6)


rmprocs(workers()) 
