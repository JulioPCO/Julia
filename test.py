from sys import path
path.append(r"<yourpath>/casadi-py27-v3.5.1")
from casadi import *
x = MX.sym("x")
print(jacobian(sin(x),x))
