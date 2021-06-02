#=
@task        : Solve a simple dynamic optimization problem using direct collocation in JuMP
@author      : Johannes Jaeschke
@contact     : jaschke@ntnu.no
@organization: Department of Chemical Engineering, NTNU, Norway
@since       : May, 2019
@change      :
@version     : v1.0
@todo        :
@requires    : Julia, JuMP, Ipopt PyPlot
=#

#−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−#
using JuMP
using Ipopt
#using PyPlot
using Plots;
#pyplot(size = (300,300), legend = false)
#−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−#
# Collocation parameters and radau time series
M = [0.19681547722366  0.39442431473909 0.37640306270047;
         -0.06553542585020  0.29207341166523 0.51248582618842;
          0.02377097434822 -0.04154875212600 0.11111111111111]
radau  = [0.15505 0.64495 1.00000]

nfe    = 30          # number of finite elements (control intervals)
ncp    = 3           # number of collocation points
th     = 10          # time horizon
h      = th/nfe      # length of one finite element on the time horizon

ts     = Vector{Float64}(undef,nfe) # time series for plotting
for i in 1:nfe
    ts[i] = h*i
end
#add a zero to the beginning of ts
ts = pushfirst!(ts,0)

#−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−#
# Initial conditions
x1_init = 0.0
x2_init = 1.0


# JuMP model
m = Model(with_optimizer(Ipopt.Optimizer, print_level = 5) )
                                       #max_iter = 20000
                                       #warm_start_init_point = "yes",
                                       #mu_init = 1e-3,
                                       #replace_bounds = "yes"

# Set up variables
@variables(m, begin
    x1[1:nfe, 1:ncp] >= -0.25
    x2[1:nfe, 1:ncp]
    -1 <= u1[1:nfe] <= 1
    x1dot[1:nfe, 1:ncp]
    x2dot[1:nfe, 1:ncp]
end)

# Set up initial guesses for solver
for i in 1:nfe
    for j in 1:ncp
        set_start_value(x1[i,j], 0)
        set_start_value(x2[i,j], 1)
        set_start_value(u1[i],   0)
    end
end

# Set up objective function
@NLobjective(m, Min, sum(h*(sum(((x1[i,j])^2 + x2[i,j]^2 )*M[j,ncp] for j in 1:ncp)) for i in 1:nfe))

#Set up the constraints
@NLconstraints(m, begin
    # set up differential equations
    dx1dt[i=1:nfe, j=1:ncp], x1dot[i,j] == x2[i,j]*x1[i,j] + u1[i]
    dx2dt[i=1:nfe, j=1:ncp], x2dot[i,j] == x1[i,j]-x2[i,j]

    # set up collocation equations - 2nd-to-nth point
    coll_x1_n[i=2:nfe, j=1:ncp], x1[i,j] == x1[i-1,ncp]+h*sum(M[k,j]*x1dot[i,k] for k in 1:ncp)
    coll_x2_n[i=2:nfe, j=1:ncp], x2[i,j] == x2[i-1,ncp]+h*sum(M[k,j]*x2dot[i,k] for k in 1:ncp)
    # set up collocation equations - 1st point
    coll_x1_0[i=1, j=1:ncp], x1[i,j] == x1_init+h*sum(M[k,j]*x1dot[i,k] for k in 1:ncp)
    coll_x2_0[i=1, j=1:ncp], x2[i,j] == x2_init+h*sum(M[k,j]*x2dot[i,k] for k in 1:ncp)
end)

#−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−#
# Solve the model
solveNLP = JuMP.optimize!
status = solveNLP(m)

# Print cost
Cost   = JuMP.objective_value(m)
println("\n−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−")
println("The cost is ", Cost, " price units")
println("−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−\n")

# Get values for plotting
x1_list   = JuMP.value.(x1[:,3])
x2_list   = JuMP.value.(x2[:,3])
u1_list   = JuMP.value.(u1)
u1first   = u1_list[1]

# Add initial values to the lists (for plotting)
x1_list   = pushfirst!(x1_list, x1_init)
x2_list   = pushfirst!(x2_list, x2_init)
u1_list   = pushfirst!(u1_list, NaN)


# plot the optimal openloop solution of the MPC
p1 = plot(x1_list, label = "x1")
p2 = plot(x2_list, label = "x2")
p3 = scatter(u1_list, label = "Input u")
plot(p1,p2,p3, layout = (3,1))


#−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−#
