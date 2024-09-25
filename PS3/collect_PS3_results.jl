using Optim, Parameters, Plots, LinearAlgebra 

include("solve_hopenhayn_rogerson.jl")
include("solve_hopenhayn_rogerson_ev_shocks.jl")


# Solve the Standard Model -----------------------------------
println("Printing resuls for standard model")
prim, res = Initialize()
solve_model(prim, res)

standard_x_pr = res.x_pr

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

# Solve the EV Shock model for alpha = 1  -----------------------------------
println("")
println("Printing resuls for α = 1")
prim, res = Initialize()
solve_model_ev(prim, res)

ev_x_pr = res.x_pr

# main results for table 

mass_incumbents_ev = sum((1 .- res.x_pr) .* res.μ)
mass_exits_ev = sum(res.x_pr .* res.μ)
N_d_ev = ( prim.θ*res.p .* prim.s_grid ).^(1/(1 - prim.θ))
L_d_ev = sum(N_d_ev .* res.μ) + res.M*sum(N_d_ev .* prim.v_dist)
L_d_incumbents_ev = sum((1 .- res.x_pr) .* N_d_ev .* res.μ)
L_d_entrants_ev = res.M*sum(N_d_ev .* prim.v_dist)

println("price = ", res.p)
println("mass incumbents = ", mass_incumbents_ev)
println("mass entrants = ", res.M)
println("mass exits = ",mass_exits_ev )
println("Aggregate Labor Demand = ", L_d_ev)
println("Incumbent Labor Demand = ", L_d_incumbents_ev)
println("Entrant Labor Demand = ", L_d_entrants_ev)
println("Fraction Labor by Entrants = ", L_d_entrants_ev/L_d_ev)


#Solve the EV Shock model for alpha = 2  -----------------------------------
println("")
println("Printing resuls for α = 2")
prim, res = Initialize()
res.α = 2.0
solve_model_ev(prim, res)

ev2_x_pr = res.x_pr


mass_incumbents_ev2 = sum((1 .- res.x_pr) .* res.μ)
mass_exits_ev2 = sum(res.x_pr .* res.μ)
N_d_ev2 = ( prim.θ*res.p .* prim.s_grid ).^(1/(1 - prim.θ))
L_d_ev2 = sum(N_d_ev2 .* res.μ) + res.M*sum(N_d_ev2 .* prim.v_dist)
L_d_incumbents_ev2 = sum((1 .- res.x_pr) .* N_d_ev2 .* res.μ)
L_d_entrants_ev2 = res.M*sum(N_d_ev2 .* prim.v_dist)

println("price = ", res.p)
println("mass incumbents = ", mass_incumbents_ev2)
println("mass entrants = ", res.M)
println("mass exits = ",mass_exits_ev2 )
println("Aggregate Labor Demand = ", L_d_ev2)
println("Incumbent Labor Demand = ", L_d_incumbents_ev2)
println("Entrant Labor Demand = ", L_d_entrants_ev2)
println("Fraction Labor by Entrants = ", L_d_entrants_ev2/L_d_ev2)

# Plot the decision rules -------------------------------------------------------
exit_choices = hcat(standard_x_pr, ev_x_pr, ev2_x_pr)
plots_cf10 = plot(prim.s_grid, exit_choices,
     xlabel = "Productivity",
     ylabel = "Probability of Exit",
     label = ["Standard Model" "α = 1" "α = 2"] , 
     title = "Exit Decision Rules with Fixed cost = 10" )


# Raise cf = 15 -------------------------------------------------------

prim, res = Initialize() 
res.c_f = 15.0

solve_model(prim, res)
standard_x_pr_cf_h = res.x_pr

prim, res = Initialize() 
res.c_f = 15.0
res.α = 1.0
solve_model_ev(prim, res)
ev1_x_pr_cf_h = res.x_pr


prim, res = Initialize() 
res.c_f = 15.0
res.α = 2.0
solve_model_ev(prim, res)
ev2_x_pr = res.x_pr


exit_choices = hcat(standard_x_pr_cf_h, ev1_x_pr_cf_h, ev2_x_pr)
plots_cf15 = plot(prim.s_grid, exit_choices,
     xlabel = "Productivity",
     ylabel = "Probability of Exit",
     label = ["Standard Model" "α = 1" "α = 2"] , 
     title = "Exit Decision Rules with fixed cost = 15" )

savefig(plots_cf10, "./PS3/exit_rule_c_f_10.png")
savefig(plots_cf15, "./PS3/exit_rule_c_f_15.png")