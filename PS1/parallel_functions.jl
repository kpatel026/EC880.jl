#= 
This code is a parallelized version of the VFI code for the neoclassical growth model.
The main difference is that the Bellman operator is parallelized using the @distributed macro.
September 2024
=#


@everywhere @with_kw struct Primitives
    β::Float64 = 0.99
    δ::Float64 = 0.025
    α::Float64 = 0.36
    k_min::Float64 = 0.01
    k_max::Float64 = 90.0
    nk::Int64 = 1000
    k_grid::SharedVector{Float64} = SharedVector(collect(range(k_min, k_max, nk)))

    z_grid = SharedVector{Float64}([1.25, 0.2]) # productivity grid
    nz::Int64 = 2
    Π = [0.977 0.023 ; 0.074 0.926] # productivity transition matrix 
end

@everywhere @with_kw mutable struct Results
    val_func::SharedArray{Float64}
    pol_func::SharedArray{Float64}
end

@everywhere function Initialize()
    prim = Primitives()
    val_func = SharedArray(zeros(prim.nk,prim.nz))
    pol_func = SharedArray(zeros(prim.nk,prim.nz))
    res = Results(val_func, pol_func)
    prim, res
end

@everywhere function Bellman(prim::Primitives, res::Results)
    @unpack_Results res
    @unpack_Primitives prim
    
    v_next = SharedArray{Float64}(nk,nz)

    @sync @distributed for (k_index, z_index) in collect(Iterators.product(1:nk,1:nz))

        k = k_grid[k_index]
        z = z_grid[z_index]
        p_z = Π[z_index, : ]
        candidate_max = -Inf
        budget = z*k^α + (1-δ)*k
        
        for kp_index in 1:nk
            c = budget - k_grid[kp_index]
            if c > 0
                val = log(c) + β*(p_z[1]*val_func[kp_index,1] + p_z[2]*val_func[kp_index,2]) # expected compute value
                if val > candidate_max
                    candidate_max = val
                    res.pol_func[k_index,z_index] = k_grid[kp_index]
                end
            end
        end
        v_next[k_index,z_index] = candidate_max
    end
    v_next
end

function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)
    n = 0

    while err > tol
        v_next = Bellman(prim, res)
        err = maximum(abs.(v_next .- res.val_func))
        res.val_func .= v_next
        n += 1
        println(n)
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end