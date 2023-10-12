import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import sympy
import pysr

# def darcysLaw(poro,pIn,pOut,length,K):
#     """
#     Darcy's Law:
#         e*v = -K*(p_out-p_in)/L
#     """
#     velocity = -K*(pOut-pIn)/(length*poro)
#     velocity = velocity #+ np.random.normal(0,100)
#     return velocity

# n = 100
# porosity = np.random.uniform(0.2,0.5,n)
# density = 997.*np.ones(n)  # kg/m^3
# pIn = np.random.uniform(100,1000,n) # Pa
# pOut = pIn - np.random.uniform(10,100.,n) # Pa
# length = np.random.uniform(0.5,10,n) # Pa  # m
# mu = 8.9e-4*np.ones(n)  # Pa-s
# K = np.random.uniform(0.1,1,n)

# vel = darcysLaw(porosity,pIn,pOut,length,K)

# X = np.zeros([n,6])
# X[:,0] = porosity
# X[:,1] = density
# X[:,2] = pIn
# X[:,3] = pOut
# X[:,4] = length
# X[:,5] = mu

# dataOut = np.zeros([n,7])
# dataOut[:,0:6] = X
# dataOut[:,6] = vel

# file = 'DarcyData.csv'
# np.savetxt(file, dataOut, delimiter=',', header='Porosity, Density, Pressure In, Pressure Out,Sample Length,Viscosity, Velocity',comments='')

file = 'DarcyData.csv'
data = np.loadtxt(file,delimiter=',',skiprows=1)
#print(data)

porosity = data[:,0]
density = data[:,1]
pIn = data[:,2]
pOut = data[:,3]
length = data[:,4]
mu = data[:,5]
vel = data[:,6]


vData = -(pOut-pIn)/(length*porosity)
plt.plot(vData,vel,'o')
z = np.polyfit(vData, vel, 1)
p = np.poly1d(z)
plt.plot(vData,p(vData),"r--")
# the line equation:
print("y=%.6fx+(%.6f)"%(z[0],z[1]))
plt.show()


model = pysr.PySRRegressor(
    niterations=100,  # < Increase me for better results
    binary_operators=["+", "*", '/', '-'],
    unary_operators=[],
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(data[:,0:6], vel)

print(model)

print(model.sympy())