import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fileName = 'DarcyData.csv'

data = pd.read_csv(fileName)

print(data.columns)


# plt.plot(data[' Velocity'],data['Porosity'],'o')
# plt.xlabel('Velocity')
# plt.ylabel('Porosity')
# plt.show()

gradP  = (data[' Pressure In'] - data[' Pressure Out'])/data['Sample Length']

term2 = gradP/data['Porosity'] 

# plt.plot(data[' Velocity'],term2,'o')
# plt.xlabel('Velocity')
# plt.ylabel('Pressure Gradient')
# plt.show()


vData = data[' Velocity']


plt.plot(vData,term2,'o')
z = np.polyfit(vData, term2, 1)
p = np.poly1d(z)
plt.plot(vData,p(vData),"r")
# the line equation:
print("vel=%.6fterm2+(%.6f)"%(z[0],z[1]))
plt.show()
