using JuMP, DifferentialEquations,DiffEqSensitivity, Plots, Ipopt

#parameters
k10=2.0 * exp(7.0) #1/s
k20=8.0* exp(14.0)
E1=55000.0 #J/mol
E2=110000.0
R=8.314 #J/mol*K
alfa=E2/E1


# nx – number of states, nfe – number of intervals
nfe=600
nx=2
h=60.0 #Time span
dh=h/nfe
u0=1.0

#initial guess
x0_1=1.0
x0_2=0.0
x0_3=0.0

#Vectors
x0=Vector{Float64}(undef, 3)
x0=[x0_1,x0_2,x0_3]


#Function ODE
function F(x0_1,x0_2,x0_3,u,j)
x0 = [x0_1,x0_2,x0_3]

function f(xdot,x,u,t)
	xdot[1] =  -u*x[1]
	xdot[2] = u*x[1]- k20*(u^alfa)*x[2]/(k10^alfa)
	#objective function
	xdot[3] = xdot[2]
end

tspan = (0.0,dh) #integrating over the interval h

prob = ODEProblem(f,x0,tspan,u)
sol = DifferentialEquations.solve(prob,Vern9(),reltol=1e-8,abstol=1e-8,save_everystep=false)
xk = sol.u[end]

return xk[trunc(Int,j)]
end


# JuMP model
m = Model(with_optimizer(Ipopt.Optimizer, warm_start_init_point = "yes", print_level = 5) )

#registering user-defined function
JuMP.register(m, :F, 5, F, autodiff=true)

@variables(m, begin
# states – dependent variables
x[1:nx + 1, 1:nfe+1]
# input – independent variables
-1 <= u[1:nfe] <= 1
end)

#initial guess
for i in 1:nfe+1
for l in 1:nx+1
	set_start_value(x[l,i], x0[l])
end
if i < nfe + 1
	set_start_value(u[i], u0)
end
end

@NLobjective(m, Min, -x[3,end])

@NLconstraints(m, begin
# system dynamics
eq[l=1:nx + 1,i=2:nfe + 1], x[l,i] - F(x[1,i - 1],x[2,i - 1],x[3,i - 1],u[i - 1],l) == 0
 eq0[l=1:nx + 1], x[l,1] - x0[l] == 0

# set up operational constraints

end)


# Solve the model
solveNLP = JuMP.optimize!
status = solveNLP(m)

# Print cost
Cost   = JuMP.objective_value(m)
println("\n−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−")
println("The maximum B is ", -Cost, "")
println("−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−\n")

#Temperature
T=-E1/(R*log(JuMP.value.(u[end])/k10))

println("\n−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−")
println("The Temperature is ", T, "K")
println("−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−\n")

# Get values for plotting
x1_list   = JuMP.value.(x[1,:])
x2_list   = JuMP.value.(x[2,:])
x3_list   = JuMP.value.(x[3,:])
u1_list   = JuMP.value.(u)
u1first   = u1_list[1]

# Add initial values to the lists (for plotting)
x1_list   = pushfirst!(x1_list, 1)
x2_list   = pushfirst!(x2_list, 0)
u1_list   = pushfirst!(u1_list, NaN)


# plot the optimal openloop solution of the MPC
p1 = plot(x1_list, label = "x1")
#p2 = plot!(x2_list, label = "x2")
p3 = plot!(x3_list, label = "x2")
p4 = plot!(u1_list, label = "Input u")
