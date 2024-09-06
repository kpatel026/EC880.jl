using Parameters, Plots #import the libraries we want
include("02Growth_model.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

val_func_low_z = val_func[:,2]
val_func_high_z = val_func[:,1]


##############Make plots
#value function
Plots.plot(k_grid, val_func_low_z, title="Value Function, z = 0.2")
Plots.plot(k_grid, val_func_high_z, title="Value Function, z = 1.25")
Plots.savefig("02_Value_Functions.png")

#policy functions
Plots.plot(k_grid, pol_func, title="Policy Functions")
Plots.savefig("02_Policy_Functions.png")

#changes in policy function
pol_func_δ = copy(pol_func).-k_grid
Plots.plot(k_grid, pol_func_δ, title="Policy Functions Changes")
Plots.savefig("02_Policy_Functions_Changes.png")

println("All done!")
################################

