import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


"""
This code solves the 2-dimensional Laplace Equation for First-type Boundary Conditions


          TOP
     ______________
     |     ^ y    |
     |     |      |
Left |     ->x    | Right
     |            |
     |            |
     ______________
         Bottom       
"""

# Size of Domain [Lx,Ly]
L = [1,1]

# Boundary Conditions
lBound = 1.01  # Left
rBound = 1.  # Right
tBound = 1.01  # Top
bBound = 1.  # Bottom

# Number of Internal Nodes
Nx = 10
Ny = 10
N = Nx*Ny

# Spacing between Nodes
dX = L[0]/(Nx+1)
dY = L[1]/(Ny+1)

# location of the Nodes
x = np.linspace(0, L[0], num=Nx+2)
y = np.linspace(0, L[1], num=Ny+2)
xInternal = np.linspace(0, L[0], num=Nx)
yInternal  = np.linspace(0, L[1], num=Ny)
X,Y = np.meshgrid(x, y)
XInt,YInt = np.meshgrid(xInternal, yInternal)

# porec-calculate values for speed
invdX2 = 1./(dX*dX)
invdY2 = 1./(dY*dY)

# Initialize arrays 
A = np.zeros([N,N])
b = np.zeros(N)


### Loop through each node in domain and set values
c = 0
for j in range(0,Ny):
    for i in range(0,Nx):
        if i == 0 and j == 0: # Left Bottom Corner
            b[c]       = -lBound*invdX2 - bBound*invdY2
            A[c, c]    = -2.*invdX2 - 2*invdY2
            A[c, c+1]  = invdX2
            A[c, c+Nx] = invdY2

        elif i == 0 and j == Ny-1: # Left Top Corner
            b[c]       = -lBound*invdX2 - tBound*invdY2
            A[c, c-Nx] = invdY2
            A[c, c]    = -2.*invdX2 - 2*invdY2
            A[c, c+1]  = invdX2

        elif i == Nx-1 and j == 0: # Right Bottom Corner
            b[c]       = -rBound*invdX2 - bBound*invdY2
            A[c, c-1]  = invdX2
            A[c, c]    = -2.*invdX2 - 2*invdY2
            A[c, c+Nx] = invdY2

        elif i == Nx-1 and j == Ny-1: # Right Top Corner
            b[c]       = -rBound*invdX2 - tBound*invdY2
            A[c, c-1]  = invdX2
            A[c, c-Nx] = invdY2
            A[c, c]    = -2.*invdX2 - 2*invdY2

        elif i == 0: # Left Boundary
            b[c]       = -lBound*invdX2
            A[c, c-Nx] = invdY2
            A[c, c]    = -2.*invdX2 - 2*invdY2
            A[c, c+1]  = invdX2
            A[c, c+Nx] = invdY2

        elif i == Nx-1: # Right Boundary
            b[c]       = -rBound*invdX2
            A[c, c-1]  = invdX2
            A[c, c-Nx] = invdY2
            A[c, c]    = -2.*invdX2 - 2*invdY2
            A[c, c+Nx] = invdY2

        elif j == 0: # Bottom Boundary
            b[c]       = -bBound*invdY2
            A[c, c-1]  = invdX2
            A[c, c]    = -2.*invdX2 - 2*invdY2
            A[c, c+1]  = invdX2
            A[c, c+Nx] = invdY2


        elif j == Ny-1: # Top Boundary
            b[c]       = -tBound*invdY2
            A[c, c-1]  = invdX2
            A[c, c-Nx] = invdY2
            A[c, c]    = -2.*invdX2 - 2*invdY2
            A[c, c+1]  = invdX2


        else: # Internal Node
            A[c, c-1]  = invdX2
            A[c, c-Nx] = invdY2
            A[c, c]    = -2.*invdX2 - 2*invdY2
            A[c, c+1]  = invdX2
            A[c, c+Nx] = invdY2

        c = c + 1 # Add Counter


# Solve linear system of equation
phi = np.linalg.solve(A,b)

# Reshpae solution variables
phi = phi.reshape(Nx,Ny)

# Append solutuion variables with boundary conditions
phi = np.pad(phi,1)
phi[:,0]  = bBound
phi[:,-1] = tBound
phi[0,:]  = lBound
phi[-1,:] = rBound

# Plot solution

fig, ax = plt.subplots()
CS = ax.contourf(Y, X, phi)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(Y, X, phi, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# plt.show()


v = np.zeros([Nx,Ny,2])
# Calaculate Velocity v = \del phi
for i in range(1,Nx):
    for j in range(1,Ny):
        v[i,j,0] = (phi[i+1,j] - phi[i-1,j])/(2*dX)
        v[i,j,1] = (phi[i,j+1] - phi[i,j-1])/(2*dY)

fig, ax = plt.subplots()
q = ax.quiver(YInt, XInt, v[:,:,0], v[:,:,1])
plt.show()
