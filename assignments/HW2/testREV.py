import REV

# Set Domain Size
domain = [1.,1.]

# Set Desired Porosity
# For a low porosity value, the code may time out and stop. 
porosity = 0.6

# Set Distribution and Paramters for Solid Radius
# Distribution can be 'normal','uniform', or 'lognormal'
# Parameter Values are
# 'uniform' -> [minValue,maxValue]
# 'normal' -> [mean,standard deviation]
# 'lognormal' -> [mean,standard deviation]

dist ='lognormal' 
param = [0.01,0.001] 

# Generate Solids
packing = REV.circlePacking(domain,porosity,dist,param)
packing.generate_packing()
packing.plot_packing() 

# Digitize Porous Media
digi = REV.digitizedDomain(packing, nodes = [200,200])
digi.generate_domain()

# Plot the Domain
digi.plot_domain()

# Sample Porous Media and calculate porosity
digi.sample_domain()

# Plot porosity vs sample area
digi.plot_sample()

