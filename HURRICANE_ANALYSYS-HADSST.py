#!/usr/bin/env python
# coding: utf-8

# ### Met Office Hadley sea-surface temperature trends
# 
# Two types of datasets are used in this analysis: HadCRUT (blend of land and sea-surface temperature) and HadSST (exclusively sea-surface temperatures). All data is collected from records and systematic measurements from buoys, ships, aircraft.
# 
# When using the HadSST data you must cite:
# Kennedy, J. J., Rayner, N. A., Atkinson, C. P., & Killick, R. E. (2019). An ensemble data set of sea‐surface temperature change from 1850: the Met Office Hadley Centre HadSST.4.0.0.0 data set. Journal of Geophysical Research: Atmospheres, 124. https://doi.org/10.1029/2018JD029867
# 
# The data used is the timeseries HadSST.4.0.0.0 annual data set for the globe and the Northern Hemisphere. The columns from left to right are as follows:
# 
# * For yearly data: year, anomaly (K), total uncertainty (K), uncorrelated uncertainty (K), correlated uncertainty (K), bias uncertainty (K), coverage uncertainty (K)
# * For mothly data: year, month, anomaly (K), total uncertainty (K), uncorrelated uncertainty (K), correlated uncertainty (K), bias uncertainty (K), coverage uncertainty (K)

# In[14]:


import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#geocat packages
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
from IPython.display import HTML


# #### HadCRUT blended temperatures

# In[15]:


#Open the data file stored in my local machine
file_path_hadcrut = 'C:/Users/anapa/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc'

#load the file using xarray library
ds = xr.open_dataset(file_path_hadcrut, decode_times=False)
#print(ds)

# Assuming 'temperature_anomaly' is the temperature variable, replace it with the actual variable name in your dataset
# Also, replace 'time=0' with the appropriate time index if needed
tee = ds.tas_mean.isel(time=2084)


# In[16]:


print(ds.data_vars)  #identify the variable corresponding to temperature
print(ds['tas_mean'])

# Extract latitude and longitude data
latitude = ds.latitude
longitude = ds.longitude

# Define the latitude and longitude ranges for the Atlantic Ocean
lat_range = slice(30, 80)  # Latitude range for the Atlantic Ocean
lon_range = slice(280, 20)  # Longitude range for the Atlantic Ocean

# Subset the dataset for the Atlantic Ocean region
atlantic_data = ds.sel(latitude=lat_range, longitude=lon_range)
print(atlantic_data)


# #### HadSST temperature anomaly

# The data is in a csv file without headers. From the documentation by the Met Office (https://www.metoffice.gov.uk/hadobs/hadsst4/HadSST.4.0.0.0_Product_User_Guide.pdf), the columns correspond to:
# * column 1: year; integer
# * column 2: anomaly. SST anomaly relative to 1961-1990; floating point
# * column 3: total_uncertainty. Uncertainty combining all sources of uncertainty: uncorrelated 
# measurement error, sampling error, correlated measurement erro 
# and residual bias uncertainty; floating poi.
# * column 4: uncorrelated_uncertainty. Uncertainty combining all sources of uncertainty: uncorrelated 
# measurement error, sampling error, correlated measurement erro 
# and residual bias uncertainty; floating poi.
# * column 5: correlated_uncertainty 1σ Uncertainty arising from correlated measurement errors; floating
# poin.
# * column 6: bias_uncertainty. 1σ residual bias uncertainty; floating point.
# 
# All columns except 1 are in unit of Kelvin (K).tntnt

# In[17]:


file_path_hadsst_north = 'C:/Users/anapa/HadSST.4.0.0.0_annual_NHEM.csv' #data for Norhtern Hemisphere only
file_path_hadsst_globe = 'C:/Users/anapa/HadSST.4.0.0.0_annual_GLOBE.csv'
file_path_hadsst_south = 'C:/Users/anapa/HadSST.4.0.0.0_annual_SHEM.csv'
file_path_hadsst_tropic = 'C:/Users/anapa/HadSST.4.0.0.0_annual_TROP.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path_hadsst_north)
dn = pd.read_csv(file_path_hadsst_globe)
ds = pd.read_csv(file_path_hadsst_south)
dt = pd.read_csv(file_path_hadsst_tropic)

# Extract columns 1, 2, and 3 into separate arrays NORTH
years_north = df.iloc[:, 0].values
ta_north = df.iloc[:, 1].values
totun_north = df.iloc[:, 2].values
column_4_north = df.iloc[:, 3].values
column_5_north = df.iloc[:, 4].values

#globe
years_globe = dn.iloc[:,0].values
ta_globe = dn.iloc[:,1].values
totun_globe = dn.iloc[:,2].values

#south
years_south = ds.iloc[:,0].values
ta_south = ds.iloc[:,1].values
totun_south = ds.iloc[:,2].values

#tropic
years_tropic = dt.iloc[:,0].values
ta_tropic = dt.iloc[:,1].values
totun_tropic = dt.iloc[:,2].values


# In[18]:


years_north


# In[19]:


#provided the year arrays for both North and Globe are the same, I can use only one of those arrays to define the x-axis values.
#proven by manual inspection of the arrays

#absolute values for uncertainties NORTH
uncertainties_north = np.abs(totun_north)
#absolute values for uncertainties GLOBE
uncertainties_globe = np.abs(totun_globe)
uncertainties_south = np.abs(totun_south)
uncertainties_tropic = np.abs(totun_tropic)

#plot temperature anomaly
plt.figure(figsize=(18, 6))
plt.plot(years_north, ta_north, linestyle='-', marker='.', color='red', label='North Hemisphere') #NORTH
plt.plot(years_south, ta_south, linestyle='-', marker='.', color='deepskyblue', label='South Hemisphere')
plt.plot(years_globe, ta_globe, linestyle='-', marker = '.', color='purple', label = 'Globe') #GLOBE
plt.plot(years_tropic, ta_tropic, linestyle='-', marker = '.', color='greenyellow', label = 'Tropic') #GLOBE
plt.axhline(0, linestyle='dashed', color='grey')


# Fill the area between the trend line and the upper/lower bounds
#plt.plot(column_1, column_2, color='red')

#plot uncertainties trend
#plt.plot(years_north, uncertainties_north,  color='purple', label = 'uncertainties North')
#plt.plot(years_north, uncertainties_globe, color = 'brown', label = 'uncertainties Globe')
#plt.plot(years_north, column_4, color='blue')
#plt.plot(years_north, column_5, color='green')

#general plot settings
#plt.xlabel('Years', fontsize=19)
plt.ylabel(r'Temperature anomaly ($^\circ$C)', fontsize=21)
plt.title('a) Sea Surface Temperature anomaly evolution', loc='left',fontsize=21)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize='x-large')
plt.grid(False)
plt.tight_layout()
plt.savefig('temp_anomaly_all_datasets.png')  #Save the plot as a PNG image file
plt.show()
# Display download link
HTML(f'<a href="temp_anomaly_all_datasets.png" download>Click here to download the plot</a>')


# Plot above: Temperature anomaly evolution for all data sets
# 
# Read in the calculated uncertainties produced in a different Jupyter Notebook, and asved in a .txt file named mean_uncertainty_per_year.txt. I will use these uncertainties in the computation of confidence interval for the global dataset.

# In[20]:


mean_global_uncertainty_filepath = 'C:/Users/anapa/mean_uncertainty_per_year.txt'
northern_uncertainty_filepath = 'C:/Users/anapa/mean_uncertainty_per_year_northern.txt'
mean_uncertainty_per_year_global = np.loadtxt(mean_global_uncertainty_filepath)
mean_uncertainty_per_year_northern = np.loadtxt(northern_uncertainty_filepath)


# In[21]:


#mean_uncertainty_per_year_global = mean_uncertainty_per_year_global[:-3].tolist()
#mean_uncertainty_per_year_northern = mean_uncertainty_per_year_northern[:-3].tolist()


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Assuming your data is already loaded and variables are defined
# years_north, ta_north, uncertainties_north
# ta_globe, uncertainties_globe

# Function to calculate confidence interval
def calculate_ci(data, uncertainties, confidence=0.95):
    degrees_of_freedom = len(data) - 1
    standard_error = uncertainties/ np.sqrt(len(data))
    margin_of_error = t.ppf((1 + confidence) / 2, degrees_of_freedom) * standard_error
    lower_bound = data - margin_of_error
    upper_bound = data + margin_of_error
    return lower_bound, upper_bound


#function to calculate error bounds
def errorbounds(data, uncertainties):
    lower_bound_ = data - uncertainties
    upper_bound_ = data + uncertainties
    return lower_bound_, upper_bound_
    
# Calculate 95% CI for ta_north
lower_bound_north, upper_bound_north = calculate_ci(ta_north, mean_uncertainty_per_year_northern) #uncertainties_north

# Calculate 95% CI for ta_globe
lower_bound_globe, upper_bound_globe = calculate_ci(ta_globe, mean_uncertainty_per_year_global)

#Calculate error bounds for ta_north
lower_bound_north_, upper_bound_north_ = errorbounds(ta_north, mean_uncertainty_per_year_northern)

#calculate error bounds for ta_globe
lower_bound_globe_, upper_bound_globe_ = errorbounds(ta_globe, mean_uncertainty_per_year_northern)

# Plotting
plt.figure(figsize=(18, 6))
plt.plot(years_north, ta_north, linestyle='-', marker='.', color='red', label='North Hemisphere')
plt.plot(years_north, ta_globe, linestyle='-', marker='.', color='purple', label='Globe')

# Fill the area between the trend line and the upper/lower bounds for CONFIDENCE INTERVAL
plt.fill_between(years_north, lower_bound_north, upper_bound_north, color='red', alpha=0.2, label='95% CI North')
plt.fill_between(years_north, lower_bound_globe, upper_bound_globe, color='purple', alpha=0.2, label='95% CI Globe')
plt.axhline(0, linestyle='dashed', color='grey')

# Fill the area between the trend line and the upper/lower bounds for ERROR BOUNDS
#plt.fill_between(years_north, lower_bound_north_, upper_bound_north_, color='lightgreen', alpha=0.2, label='error bounds North')
#plt.fill_between(years_north, lower_bound_globe_, upper_bound_globe_, color='lightblue', alpha=0.2, label='error bounds Globe')

#POLYNOMIAL FIT NORTH
degree = 4 # You can adjust the degree based on the trend you observe
coefficients = np.polyfit(years_north, ta_north, degree)
poly_function = np.poly1d(coefficients)
#plt.plot(years_north, poly_function(years_north), color='darkred', label='Polynomial fit North')

#POLYNOMIAL FIT GLOBE
degree = 4# You can adjust the degree based on the trend you observe
coefficients = np.polyfit(years_north, ta_globe, degree)
poly_function = np.poly1d(coefficients)
#plt.plot(years_north, poly_function(years_north), color='darkviolet',  label='Polynomial fit Globe')


# General plot settings
#plt.xlabel('Years', fontsize=21)
plt.ylabel(r'Temperature anomaly ($^\circ$C)', fontsize=21)
plt.title('b) Sea Surface Temperature anomaly evolution', loc= 'left', fontsize=21)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize='x-large')
plt.grid(False)
plt.tight_layout()
plt.savefig('temp_anomaly_nh_globe_poly.png')  #Save the plot as a PNG image file
plt.show()
# Display download link
HTML(f'<a href="temp_anomaly_nh_globe_poly.png" download>Click here to download the plot</a>')



# plot above is  Temperature anomaly evolution for North Hemisphere and Globe data sets with Confidence Interval

# In[23]:


# Assuming your data is already loaded and variables are defined
# years_north, ta_north, uncertainties_north
# ta_globe, uncertainties_globe

# Function to calculate confidence interval
def calculate_ci(data, uncertainties, confidence=0.95):
    degrees_of_freedom = len(data) - 1
    standard_error = uncertainties/ np.sqrt(len(data))
    margin_of_error = t.ppf((1 + confidence) / 2, degrees_of_freedom) * standard_error
    lower_bound = data - margin_of_error
    upper_bound = data + margin_of_error
    return lower_bound, upper_bound


#function to calculate error bounds
def errorbounds(data, uncertainties):
    lower_bound_ = data - uncertainties
    upper_bound_ = data + uncertainties
    return lower_bound_, upper_bound_
    
# Calculate 95% CI for ta_north
lower_bound_north, upper_bound_north = calculate_ci(ta_north, mean_uncertainty_per_year_northern) #uncertainties_north

# Calculate 95% CI for ta_globe
lower_bound_globe, upper_bound_globe = calculate_ci(ta_globe, mean_uncertainty_per_year_global)

#Calculate error bounds for ta_north
lower_bound_north_, upper_bound_north_ = errorbounds(ta_north, mean_uncertainty_per_year_northern)

#calculate error bounds for ta_globe
lower_bound_globe_, upper_bound_globe_ = errorbounds(ta_globe, mean_uncertainty_per_year_northern)

# Plotting
plt.figure(figsize=(18, 6))
plt.plot(years_north, ta_north, linestyle='-', marker='.', color='red', label='North Hemisphere')
plt.plot(years_north, ta_globe, linestyle='-', marker='.', color='purple', label='Global')

plt.axhline(0, linestyle='dashed', color='grey')

# Fill the area between the trend line and the upper/lower bounds for ERROR BOUNDS
plt.fill_between(years_north, lower_bound_north_, upper_bound_north_, color='orange', alpha=0.2, label='uncertainties North')
plt.fill_between(years_north, lower_bound_globe_, upper_bound_globe_, color='blue', alpha=0.2, label='uncertainties Globe')

#POLYNOMIAL FIT NORTH
degree = 1 # You can adjust the degree based on the trend you observe
coefficients = np.polyfit(years_north, ta_north, degree)
poly_function_north = np.poly1d(coefficients)
plt.plot(years_north, poly_function_north(years_north), color='darkred')

#POLYNOMIAL FIT GLOBE
degree = 1 # You can adjust the degree based on the trend you observe
coefficients = np.polyfit(years_north, ta_globe, degree)
poly_function_globe = np.poly1d(coefficients)
plt.plot(years_north, poly_function_globe(years_north), color='darkviolet')

#Confidence interval for Northern Hemisphere
y_pred_north = poly_function_north(years_north)
y_err_north = ta_north - y_pred_north
confidence_interval_north = 1.96 * np.std(y_err_north) / np.sqrt(len(years_north))
plt.fill_between(years_north, poly_function_north(years_north) - confidence_interval_north,
                 poly_function_north(years_north) + confidence_interval_north, color='lightcoral', alpha=0.5)

#Confidence interval for Globe
y_pred_globe = poly_function_globe(years_north)
y_err_globe = ta_globe - y_pred_globe
confidence_interval_globe = 1.96 * np.std(y_err_globe) / np.sqrt(len(years_north))
plt.fill_between(years_north, poly_function_globe(years_north) - confidence_interval_globe,
                 poly_function_globe(years_north) + confidence_interval_globe, color='mediumorchid', alpha=0.5)


#General plot settings
#plt.xlabel('Years', fontsize=21)
plt.ylabel('Temperature anomaly (K)', fontsize=21)
plt.title('Temperature anomaly evolution for Northern Hemisphere and Globe data sets with raw uncertainties', fontsize=21)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize='x-large')
plt.grid(False)
plt.tight_layout()
plt.savefig('temp_anomaly_nh_globe_raw_unc.png')  #Save the plot as a PNG image file
plt.show()
# Display download link
HTML(f'<a href="temp_anomaly_nh_globe_raw_unc.png" download>Click here to download the plot</a>')


# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Fitting a polynomial to sea surface temperature anomaly
degree = 3
coefficients_sst = np.polyfit(list(sea_surface_temp_anomaly.keys()), list(sea_surface_temp_anomaly.values()), degree)
poly_function_sst = np.poly1d(coefficients_sst)
x_sst = np.linspace(min(sea_surface_temp_anomaly.keys()), max(sea_surface_temp_anomaly.keys()), 100)
y_sst = poly_function_sst(x_sst)

# Fitting a polynomial to average wind speed
degree = 3
coefficients_avg = np.polyfit(list(avg_wind_speed_year.keys()), list(avg_wind_speed_year.values()), degree)
poly_function_avg = np.poly1d(coefficients_avg)
x_avg = np.linspace(min(avg_wind_speed_year.keys()), max(avg_wind_speed_year.keys()), 100)
y_avg = poly_function_avg(x_avg)

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plots for original data
plt.scatter(sea_surface_temp_anomaly.keys(), sea_surface_temp_anomaly.values(), label='Sea Surface Temp Anomaly')
plt.scatter(avg_wind_speed_year.keys(), avg_wind_speed_year.values(), label='Avg Wind Speed')

# Polynomial fits
plt.plot(x_sst, y_sst, color='blue', label='Polynomial Fit (SST)', linestyle='--')
plt.plot(x_avg, y_avg, color='red', label='Polynomial Fit (Avg Wind Speed)', linestyle='--')

plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Polynomial Fits for Sea Surface Temp Anomaly and Avg Wind Speed')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


len(mean_uncertainty_per_year_global)


# Important note for graph above: CI includes effects from measurement, sampling and bias-adjustment errors provided in the HadSST.4.0.0.0 dataset column 2. Using these errors instead of the standard deviation in the CI formula.

# In[ ]:


len(ta_globe)


# In[ ]:


ta_globe


# In[ ]:




