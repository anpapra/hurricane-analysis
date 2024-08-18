#!/usr/bin/env python
# coding: utf-8

# ### Calculating the total uncertainties for HADSST.4.0.0.0

# In[1]:


import iris
import numpy as np
import matplotlib . pyplot as plt
import iris.analysis.cartography


# In[2]:


import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geocat.datafiles as gdf
import geocat.viz as gv
import cartopy.feature as cfeature
import shapely.geometry  # Import the shapely.geometry module
import fiona  # Import the fiona module
import netCDF4 as nc
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter, YearLocator
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import sem
from sklearn.metrics import r2_score


# In[14]:


len(uncertainty_data)


# In[18]:


alo = iris.load_cube('HadSST.4.0.0.0_total_uncertainty.nc')

# Extract only the uncertainty data
uncertainty_data = alo.data

# Define a constraint to extract only the uncertainty data
constraint = iris.Constraint(name='sea_water_temperature_anomaly standard_error')

# Apply the constraint to the cube
uncertainty_cube = alo.extract(constraint)

# Extract the uncertainty data
uncertainty_data = uncertainty_cube.data
print(uncertainty_data)


# In[47]:


is_zero_or_nan = np.logical_or(uncertainty_data == 0, np.isnan(uncertainty_data))

# Get the indices of these values
indices = np.where(is_zero_or_nan)

print("Indices of values that are either 0 or missing:", indices)


# In[ ]:





# In[19]:


alo = iris.load_cube('HadSST.4.0.0.0_total_uncertainty.nc')

# Extract latitude values
latitudes = alo.coord('latitude').points

longitudes = alo.coord('longitude').points

# Print the latitude values
print(len(longitudes))


# In[20]:


import numpy as np

# Assuming uncertainty_data is your array with shape (years, latitudes, longitudes)

# Calculate the mean uncertainty per year
mean_uncertainty_per_month = np.mean(uncertainty_data, axis=(1, 2))

# Print the resulting array
print(mean_uncertainty_per_month)


# In[21]:


# Assuming uncertainty_data is your array with shape (years, latitudes, longitudes)

# Calculate the mean uncertainty per year
mean_uncertainty_per_month = np.mean(uncertainty_data, axis=(1, 2))

# Print the resulting array
print(mean_uncertainty_per_month)

# Calculate the number of missing values to make it a multiple of 12
missing_values = 12 - len(mean_uncertainty_per_month) % 12

# Fill the missing values with zeros
mean_uncertainty_per_month = np.concatenate([mean_uncertainty_per_month, np.zeros(missing_values)])

# Reshape the array to have 12 values per row
reshaped_array = mean_uncertainty_per_month.reshape(-1, 12)

# Calculate the mean along the second axis (axis=1)
mean_uncertainty_per_year = np.mean(reshaped_array, axis=1)

# Print or use mean_uncertainty_per_year
print(mean_uncertainty_per_year)
np.savetxt('mean_uncertainty_per_year.txt', mean_uncertainty_per_year)


# In[46]:


for i in range(len(mean_uncertainty_per_month)):
    if mean_uncertainty_per_month[i] == 0:
        print(i)
print('length of array is', len(mean_uncertainty_per_month))


# In[23]:


missing_values


# In[24]:


latitudes_north = np.array([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5])
# Find the index where latitude is greater than or equal to 0 (northern hemisphere)
northern_indices = np.where(latitudes_north >= 0)[0]

# Extract data for the northern hemisphere
uncertainty_data_northern = uncertainty_data[:, northern_indices, :]

# Calculate the mean uncertainty per year for the northern hemisphere
mean_uncertainty_per_month_northern = np.mean(uncertainty_data_northern, axis=(1, 2))

# Print the resulting array
print(mean_uncertainty_per_month_northern)

# Calculate the number of missing values to make it a multiple of 12
missing_values_northern = 12 - len(mean_uncertainty_per_month_northern) % 12

# Fill the missing values with zeros
mean_uncertainty_per_month_northern = np.concatenate([mean_uncertainty_per_month_northern, np.zeros(missing_values_northern)])

# Reshape the array to have 12 values per row
reshaped_array_northern = mean_uncertainty_per_month_northern.reshape(-1, 12)

# Calculate the mean along the second axis (axis=1)
mean_uncertainty_per_year_northern = np.mean(reshaped_array_northern, axis=1)

# Print or use mean_uncertainty_per_year_northern
print(mean_uncertainty_per_year_northern)

# Save the resulting array to a text file
np.savetxt('mean_uncertainty_per_year_northern.txt', mean_uncertainty_per_year_northern)


# In[41]:


lower_bound = mean_uncertainty_per_year_northern[0]
upper_bound = mean_uncertainty_per_year_northern[77]

# Filter values within the specified interval
filtered_values = [value for value in values_array if lower_bound <= value <= upper_bound]

# Calculate the average of the filtered values
average_value = sum(filtered_values) / len(filtered_values) if filtered_values else 0
print(average_value)


# In[44]:


def calculate_average_interval(array):
    # Ensure the array has at least 78 elements
    if len(array) < 50:
        return "Array does not have enough elements."
    
    # Define the range within the array from the first element to the element at position 78 (index 77)
    interval_values = array[0:50]  # This slices the array from the first element to the 78th element
    
    # Calculate the average of values within this range
    average_value = sum(interval_values) / len(interval_values)
    
    return average_value

# Example usage
  # Your original array of values
average = calculate_average_interval(mean_uncertainty_per_year_northern)
print(f"The average of the specified interval is: {average}")


# In[39]:


def calculate_average_of_last_values(array, last_n_values=50):
    # Ensure the array has at least as many elements as specified
    if len(array) < last_n_values:
        return "Array does not have enough elements."
    
    # Select the last 'last_n_values' elements from the array
    selected_values = array[-last_n_values:]
    
    # Calculate the average of these selected values
    average_value = sum(selected_values) / len(selected_values)
    
    return average_value

# Example usage
 # Your array of values
average_of_last = calculate_average_of_last_values(mean_uncertainty_per_year_northern)

print(average_of_last)


# In[ ]:


mean_uncertainty_per_year_northern[0] 


# In[32]:


(mean_uncertainty_per_year_northern[0] - mean_uncertainty_per_year_northern[-1])/ mean_uncertainty_per_year_northern[0]*100


# In[33]:


mean_uncertainty_per_year_northern[-1]


# In[30]:


mean_uncertainty_per_year_northern[0]


# In[13]:


for k in range
    anoms = iris.load_cube('HadSST.4.0.0.0_median.nc')
    uncorrelated_unc = iris.load_cube('HadSST.4.0.0.0_uncorrelated_measurement_uncertainty.nc')
    sampling_unc = iris.load_cube('HadSST.4.0.0.0_sampling_uncertainty.nc')
    covariance = iris.load_cube('HadSST.4.0.0.0_error_covariance_185001.nc')
    # combine the uncorrelated - measurement - error and sampling uncertainties
    m_and_s_unc = uncorrelated_unc * uncorrelated_unc + sampling_unc * sampling_unc
    m_and_s_unc = iris.analysis.maths.exponentiate( m_and_s_unc , 0.5)
    fill_value = 9.96921E36
    # extract latitudes to a 2 -d array and convert to relative areas
    latitude = anoms.coord('latitude').points
    latitude_field = np.zeros((36 , 72))
    for i in range (0 ,36):latitude_field[i ,:] = latitude[i]
    area_field = np.cos( latitude_field * np . pi / 180.)
    # form a vector from the anomaly field
    anoms_1d = np.reshape(anoms.data[0,:,:],(2592,1))
    anoms_1d = anoms_1d.data
    
    # form a vector from the measurement and sampling uncertainty
    unc_1d = np.reshape(m_and_s_unc.data[0 ,: ,:],(2592 ,1))
    unc_1d = unc_1d.data
    
    # form a vector from the area field
    areas_1d = np.reshape (area_field.data ,(2592 ,1))
    
    # where there are missing data in the anomaly field , set the areas to zero
    areas_1d[ anoms_1d == fill_value] = 0.0
    unc_1d[anoms_1d == fill_value] = 0.0
    anoms_1d[anoms_1d == fill_value] = 0.0
    
    # convert the measurement and sampling uncertainty to a diagaonal covariance
    # matrix and add to the covariance
    covariance2 = np.diag(np.reshape( unc_1d * unc_1d ,(2592)))
    covariance = covariance + covariance2
    
    # turn the areas into gridcell weights for the area average calculation
    areas_1d = areas_1d/sum(areas_1d)
    
    # calculate the area average
    area_average = sum( anoms_1d * areas_1d ) / sum( areas_1d )
    area_average = area_average[0]
    
    # calculate the uncertainty in the area average . Have to do this in two stages
    # using numpy matrix multiplication
    area_average_uncertainty_pre = np.matmul( covariance.data , areas_1d )
    area_average_uncertainty_sq = np.matmul( np.matrix.transpose ( areas_1d ), area_average_uncertainty_pre)
    area_average_uncertainty = np.sqrt( area_average_uncertainty_sq.data)[0][0]
    
    # read ensemble and extract 1 st timestep for each member
    ensemble = iris.load('C:/Users/anapa/HadSST.4.0.0.0_ensemble_member_*.nc')
    for i in range(0 ,200): ensemble[i] = ensemble[i][0:1]
        
    # calculate the global area average timeseries for each ensemble member
    # the time series only has one time step though
    ensemble_gmt = np.zeros((200))
    for i in range(0 ,200):
        grid_areas = iris.analysis.cartography.area_weights( ensemble [i ])
        ts = ensemble[i ].collapsed (['longitude','latitude'] , iris.analysis.MEAN , weights = grid_areas )
        ensemble_gmt[i ] = ts.data[0]
    bias_unc = np.std( ensemble_gmt )
    area_average_uncertainty_sq += bias_unc * bias_unc
    area_average_uncertainty = np.sqrt( area_average_uncertainty_sq.data)[0][0]
    print('Global mean Jan 1850: {:4.2f} +/- {:4.2f}'.format(area_average, area_average_uncertainty))


# In[17]:





# In[20]:


area_average


# ### Calculate global mean SST time series

# In[7]:


anoms = iris.load_cube('HadSST.4.0.0.0_median.nc')
grid_areas = iris.analysis.cartography.area_weights(anoms)
timeseries = anoms.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights = grid_areas)
plt.plot(timeseries.data)
plt.show()


# In[11]:


print(m_and_s_unc)


# In[ ]:




