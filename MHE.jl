using JuMP, DifferentialEquations,DiffEqSensitivity ,Plots, Ipopt
using Distributions
nfe=600
tf=2.0



dh=tf/nfe

#parameters

DA1=3.0
DA3=1.0
x0_1=1.0
x0_2=0.0
xk=Matrix{Float64}(undef,2,nfe)

for i in 1:nfe
    xk[1,i]=x0_1
    xk[2,i]=x0_2
end

#ode x real
function F(x0_1,x0_2,u,j)
    x0 = [x0_1,x0_2]

        function f(xdot,x,u,t)
                xdot[1]=-(1+DA1)*x[1]
                xdot[2]=DA1*x[1]-(1+DA3)*x[2]
        end

        tspan = (0.0,dh) #integrating over the interval h

        prob = ODEProblem(f,x0,tspan,u)
        sol = DifferentialEquations.solve(prob,Vern9(),reltol=1e-8,abstol=1e-8,save_everystep=false)
        xkf = sol.u[end]

        return xkf[trunc(Int,j)]
end

for i in 2:nfe
    xk[1,i]=F(xk[1,i-1],xk[2,i-1],0,1)
    xk[2,i]=F(xk[1,i-1],xk[2,i-1],0,2)
end

#random number
mu = 0    #The mean of the truncated Normal
sigma = 0.05 #The standard deviation of the truncated Normal
lb = -0.5    #The truncation lower bound
ub = 0.5    #The truncation upper bound
d = Truncated(Normal(mu, sigma), lb, ub)  #Construct the distribution type

mn=Vector{Float64}(undef,nfe)
y=Vector{Float64}(undef,nfe)
for i in 1:nfe
	mn[i]=rand(d)
    y[i]=xk[2,i]+mn[i]
end

nx=2

plot(1:nfe, xk[1,:])
plot!(1:nfe,xk[2,:])
plot!(1:nfe,y[:])

x0_1=0.85
x0_2=0.15
x0=[x0_1,x0_2,0]
q=0.0025

Q=Matrix{Float64}(undef,nfe,nfe)
Q=zeros(nfe,nfe)
pk=Matrix{Float64}(undef,3,nfe)

for i in 1:nfe
    pk[1,i]=0.04
    pk[2,i]=0.0
    pk[3,i]=0.01
end

#ode P
function FP(P11,P12,P22,u,j)
    p0 = [P11,P12,P22]

    function fp(pdot,p,u,t)
        pdot[1]=-2*(1+DA1)*p[1]-q*p[2]^2
        pdot[2]=DA1*p[1]-(DA1+DA3+2)*p[2]-q*p[2]*p[3]
        pdot[3]=2*DA1*p[2]-2*(1+DA3)*p[3]-q*p[3]^2
    end

    tspan = (0.0,dh) #integrating over the interval h

    prob = ODEProblem(fp,p0,tspan,u)
    sol = DifferentialEquations.solve(prob,Vern9(),reltol=1e-8,abstol=1e-8,save_everystep=false)
    pkf = sol.u[end]

    return pkf[trunc(Int,j)]
end

for i in 2:nfe

    pk[1,i]=FP(pk[1,i-1],pk[2,i-1],pk[3,i-1],0,1)
    pk[2,i]=FP(pk[1,i-1],pk[2,i-1],pk[3,i-1],0,2)
    pk[3,i]=FP(pk[1,i-1],pk[2,i-1],pk[3,i-1],0,3)

end

plot(1:nfe, pk[1,:])
plot!(1:nfe, pk[2,:])
plot!(1:nfe, pk[3,:])


for i in 1:nfe
	Q[i,i]=mn[i]^(-2)
end


#Function ODE
function Fm(x0_1,x0_2,x0_3,p22,p12,qq,yy,j)
    x0 = [x0_1,x0_2,x0_3]
    u=[p22,p12,qq,yy]

    function f(xdot,x,u,t)
	       xdot[1]=-(1+DA1)*x[1]+u[2]*u[3]*(u[4]-x[2])
	          xdot[2]=DA1*x[1]-(1+DA3)*x[2]+u[1]*u[3]*(u[4]-x[2])
	             #objective function
	                xdot[3]=x[2]
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

JuMP.register(m, :Fm, 8, Fm, autodiff=true)

@variables(m, begin
# states – dependent variables
x[1:nx + 1, 1:nfe+1]
# input – independent variables
end)

#initial guess
for i in 1:nfe+1
    for l in 1:nx+1
	       set_start_value(x[l,i], x0[l])
     end
end

@NLobjective(m, Min, sum((y[i]-x[2,i])^2 for i in 1:nfe)+sum(x[1,i]^2 for i in 1:nfe))

@NLconstraints(m, begin
# system dynamics
eq[l=1:nx + 1,i=2:nfe + 1], x[l,i] - Fm(x[1,i - 1],x[2,i - 1],x[3,i - 1],pk[3,i-1],pk[2,i-1],Q[i-1,i-1],y[i-1],l) == 0
 eq0[l=1:nx + 1], x[l,1] - x0[l] == 0

# set up operational constraints
end)


# Solve the model
solveNLP = JuMP.optimize!
status = solveNLP(m)

#values for plotting
x1_list   = JuMP.value.(x[1,:])
x2_list   = JuMP.value.(x[2,:])
x3_list   = JuMP.value.(x[3,:])

p1 = plot(x1_list, label = "x1")
p2 = plot!(x2_list, label = "x2")
p3= plot!(1:nfe, xk[1,:])
p4= plot!(1:nfe, xk[2,:])
