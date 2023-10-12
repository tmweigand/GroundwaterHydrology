import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


"""
This code solves the 1-dimensional Laplace Equation for First-type Boundary Conditions

Left |     ->x    | Right

"""

# Size of Domain [Lx,Ly]
L = 1000.

# Boundary Conditions
lBound = 100.  # Left
rBound = 85.  # Right

# Number of Internal Nodes
N = 251

K = 10 #m/day

# Spacing between Nodes
dX = L/(N-1)

# location of the Nodes
x = np.linspace(0, L, num=N)

# pre-calculate values for speed
invdX2 = 1./(dX*dX)

# Initialize arrays
A = np.zeros([N,N])
b = np.zeros(N)


print(x[100])
print(x[200])

### Loop through each node in domain and set values
for i in range(0,N):

    if i == 0: # Left Boundary
        b[i]       = lBound
        A[i, i]    = 1

    elif i == N-1: # Right Boundary
        b[i]       = rBound
        A[i, i]    = 1

    else: # Internal Node
        A[i, i-1]  = invdX2
        A[i, i]    = -2.*invdX2
        A[i, i+1]  = invdX2

b[100] = 0.18/K

# Solve linear system of equation
phi = np.linalg.solve(A,b)

def analytic(x,lBound,rBound,L):
    h = (rBound-lBound)/L*x + lBound
    return h

analyticHead = analytic(x,lBound,rBound,L)

print(dX)
print( np.average((phi-analyticHead)/analyticHead)*100)

plt.plot(x,analyticHead)
plt.plot(x,phi)
plt.ylabel("Hydraulc Head (m)")
plt.xlabel("Distance (m)")
plt.show()