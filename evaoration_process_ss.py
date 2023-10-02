import numpy as NP
import matplotlib.pyplot as plt

from casadi import *
from casadi.tools import *

## STEADY STATE COMPUTATION 
nx=2 #number of states
nu=2 #number of control inputs

# states = struct_symSX([
#             entry('x',shape=2)    # states
#             # entry('L')     # additional state for Lagrange polynomials
#          ])

# inputs = struct_symSX([
#             entry('u',shape=2) #inputs
#          ])

# X2 = states['x',0]
# P2 = states['x',1]
# P100 = inputs['u',0]
# F200 = inputs['u',1]

X2 = MX.sym("X2")
P2 = MX.sym("P2")
P100 = MX.sym("P100")
F200 = MX.sym("F200")

#parameters of the evaporation process

aa=0.5616
bb=0.3126
cc=48.43
dd=0.507
ee=55
ff=0.1538
gg=90
hh=0.16

M=20
C=4
UA2=6.84
Cp=0.07
lambdaa=38.5
lambda_s=36.6
X1=5


#constraints values
X2_min=25
X2_max=100
P2_max=80
P2_min=40
P100_min=100
P100_max=400
F200_min=100
F200_max=400

# fixed value quantities and definition of  other quantities
T1=40
T200=25
F1=10
C1=5
F3=50
T2=aa*P2+bb*X2+cc
T3=dd*P2+ee
T100=ff*P100+gg
UA1=hh*(F1+F3)
Q100=UA1*(T100-T2)
F4=(Q100-F1*Cp*(T2-T1))/lambdaa

Q200=(UA2*(T3-T200))/(1+(UA2/(2*Cp*F200)))
F5=Q200/lambdaa
F2=F1-F4
F100=Q100/lambda_s

# rhs = struct_SX(states) #definisco una struttura come quella di 'states'

# rhs["x"] = vertcat((F1*X1-F2*X2)/M, (F4-F5)/C) # dynamics
# rhs = vertcat((F1*X1-F2*X2)/M, (F4-F5)/C) # dynamics
rhs = [(F1*X1-F2*X2)/M, (F4-F5)/C] # dynamics
cost = (10.09*(F2+F3))+(600*F100)+(0.6*F200)+(10^(-4))*(P100*P100)    # stage cost

# ODE right hand side function
f = Function('f', [X2,P2,P100,F200],rhs)

# Objective function (meyer term)

# m = Function('m', [states,inputs],[states["L"]])
# m = Function('m', [states,inputs],[cost])

# Control bounds
u_min = [P100_min, F200_min]
u_max = [P100_max, F200_max]

# u_init = [250.0, 300.0 ]



# State bounds and initial guess
x_min =  [X2_min, P2_min]
x_max =  [X2_max, P2_max]


# x_init = [ X2_ss,  P2_ss] #condizioni iniziali
# x_init = [ 50.0,  70.0]

# NLP
opt_variables = [X2,P2,P100,F200]
g_ss = [(F1*X1-F2*X2)/M, (F4-F5)/C]
lbg_ss = [0,0] #equality constr
ubg_ss = [0,0] #equality constr
#definition of the NLP problem
# prova = rhs[0]
# prova1 = rhs[1]
nlp_ss = {'x':opt_variables, 'f':cost, 'g':f}
print(opt_variables)
print(cost)
print(g_ss)
## ----
## SOLVE THE NLP
## ----


# Allocate an NLP solver
solver_ss = nlpsol("solver_ss", "ipopt", nlp_ss)




arg = {}
# Bounds on x
arg["lbx"] = [X2_min,P2_min,P100_min,F200_min ]
arg["ubx"] = [X2_max,P2_max,P100_max,F200_max ]

# Bounds on g
arg["lbg"] = lbg_ss
arg["ubg"] = ubg_ss




# Initial condition
arg["x0"] = [0,0,0,0]



# Solve the problem
res_ss = solver_ss(**arg)

# Retrieve the stesy state values

print(res_ss)


