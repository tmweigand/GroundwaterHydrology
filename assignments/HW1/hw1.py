import numpy as np
import pandas as pd
import dataretrieval.nwis as nwis
import matplotlib.pyplot as plt


### Downstream Site ID - HAW RIVER NEAR BYNUM, NC
downSiteID = '02096960'
downSite = nwis.get_record(sites=downSiteID, service='site')
print(downSite)

### Upstream Site ID - HAW RIVER AT HAW RIVER, NC
upSiteID = '02096500'
upSite = nwis.get_record(sites=upSiteID, service='site')
print(upSite)

### Dates for Start and End of Data
startDate = '2023-08-01'
endDate = '2023-08-14'

### Type of Data 
### See https://github.com/DOI-USGS/dataretrieval-python for options
dataType = 'dv'  

### For these two sites, we want discharge (ft^3/s)
### See USGS parameter codes
param = '00060'

### Grab Discharge Data
downData = nwis.get_record(sites=downSiteID, service=dataType, start=startDate, end=endDate, parameterCd=param)
upData = nwis.get_record(sites=upSiteID, service=dataType, start=startDate, end=endDate, parameterCd=param)

#### Intergrate Dischacharge Data to ft^3
secondsToday = 60.*60.*24 # Number of Seconds in a Day
downVolume = downData['00060_Mean']*secondsToday
upVolume = upData['00060_Mean']*secondsToday

### HAW RIVER AT HAW RIVER, NC also has Preciptation
param = '00045'

### Grab Rain Data (inches)
rainData = nwis.get_record(sites=upSiteID, service=dataType, start=startDate, end=endDate, parameterCd=param)

### Determine Drainage Area
downArea = downSite['drain_area_va'][0]
upArea = upSite['drain_area_va'][0]
rainArea = downArea-upArea

milesTofeet = 5280*5280 # Square Miles to Square feet
rainVolume = rainArea*milesTofeet * rainData['00045_Sum']/12.


storage = upVolume - downVolume + rainVolume
storageRate = storage/secondsToday
total = np.sum(storageRate)