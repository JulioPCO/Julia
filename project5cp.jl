using JuMP, Plots, Ipopt

#parameters
k10=2.0 * exp(7.0) #1/s
k20=8.0* exp(14.0)
E1=55000.0 #J/mol
E2=110000.0
R=8.314 #J/mol*K
alfa=E2/E1


# nx – number of states, nfe – number of intervals
nfe=131
ncp=5
h=131.0 #Time span
dh=h/nfe


#collocation points gauss-radau
t=Vector{Float64}(undef,ncp)
t=[0.057104, 0.276843,0.583590,0.860240, 1.0]

#Matrix M
M1=Matrix{Float64}(undef,ncp,ncp)
for i in 1:ncp
    for j in 1:ncp
        if j==1
            M1[i,j]= t[i]
        end

        if j==2
            M1[i,j]=(t[i]^2.0)/2.0
        end

        if j==3
            M1[i,j]=(t[i]^3.0)/3.0
        end
        if j==4
            M1[i,j]=(t[i]^4.0)/4.0
        end

        if j==5
            M1[i,j]=(t[i]^5.0)/5.0
        end
    end
end

M2=Matrix{Float64}(undef,ncp,ncp)
for i in 1:ncp
    for j in 1:ncp
        if j==1
            M2[i,j]= 1
        end

        if j==2
            M2[i,j]=t[i]
        end

        if j==3
            M2[i,j]=(t[i]^2.0)
        end

        if j==4
            M2[i,j]=(t[i]^3.0)
        end
        if j==5
            M2[i,j]=(t[i]^4.0)
        end
    end
end
M2INV=inv(M2)

M=Matrix{Float64}(undef, ncp,ncp)
M=M1*M2INV
M=transpose(M)


ts     = Vector{Float64}(undef,nfe) # time series for plotting
for i in 1:nfe
    ts[i] = dh*i
end
#add a zero to the beginning of ts
ts = pushfirst!(ts,0)

#−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−#
# Initial conditions
x1_init = 1.0
x2_init = 0.0


# JuMP model
m = Model(with_optimizer(Ipopt.Optimizer, print_level = 5) )
                                       #max_iter = 20000
                                       #warm_start_init_point = "yes",
                                       #mu_init = 1e-3,
                                       #replace_bounds = "yes"

# Set up variables
@variables(m, begin
x1[1:nfe, 1:ncp]
x2[1:nfe, 1:ncp]
-1 <= u1[1:nfe] <= 1
x1dot[1:nfe, 1:ncp]
x2dot[1:nfe, 1:ncp]
end)

# Set up initial guesses for solver
for i in 1:nfe
for j in 1:ncp
set_start_value(x1[i,j], 1.0)
 set_start_value(x2[i,j], 0.0)
 set_start_value(u1[i],   1.0)
end
end


# Set up objective function
@NLobjective(m, Min, -x2[nfe,ncp])

#Set up the constraints
@NLconstraints(m, begin
    # set up differential equations
    dx1dt[i=1:nfe, j=1:ncp], x1dot[i,j] == -x1[i,j] * u1[i]
    dx2dt[i=1:nfe, j=1:ncp], x2dot[i,j] == u1[i]*x1[i,j]- k20*(u1[i]^alfa)*x2[i,j]/(k10^alfa)

    # set up collocation equations - 2nd-to-nth point
    coll_x1_n[i=2:nfe, j=1:ncp], x1[i,j] == x1[i-1,ncp]+dh*sum(M[k,j]*x1dot[i,k] for k in 1:ncp)
    coll_x2_n[i=2:nfe, j=1:ncp], x2[i,j] == x2[i-1,ncp]+dh*sum(M[k,j]*x2dot[i,k] for k in 1:ncp)
    # set up collocation equations - 1st point
    coll_x1_0[i=1, j=1:ncp], x1[i,j] == x1_init+dh*sum(M[k,j]*x1dot[i,k] for k in 1:ncp)
    coll_x2_0[i=1, j=1:ncp], x2[i,j] == x2_init+dh*sum(M[k,j]*x2dot[i,k] for k in 1:ncp)
end)

#−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−#
# Solve the model
solveNLP = JuMP.optimize!
status = solveNLP(m)

# Print cost
Cost   = JuMP.objective_value(m)
println("\n−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−")
println("The maximum of x2 is ", -Cost, " ")
println("−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−\n")

#Temperature
T=-E1/(R*log(JuMP.value.(u1[end])/k10))

println("\n−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−")
println("The Temperature is ", T, "K")
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
p2 = plot!(x2_list, label = "x2")
p3 = plot!(u1_list, label = "Input u")



#−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−#
