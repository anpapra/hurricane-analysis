#!/usr/bin/env python
# coding: utf-8

# ### Study of correlation between SSTs and Hurricane wind speeds

# In[1]:


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


# #### Import HADSST

# In[2]:


#importing HADSST 
file_path_hadsst_north = 'C:/Users/anapa/HadSST.4.0.0.0_annual_NHEM.csv' #data for Norhtern Hemisphere only
file_path_hadsst_globe = 'C:/Users/anapa/HadSST.4.0.0.0_annual_GLOBE.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path_hadsst_north)
dt = pd.read_csv(file_path_hadsst_globe)


# Extract columns 1, 2, and 3 into separate arrays NORTH
years_north = df.iloc[:, 0].values
ta_north = df.iloc[:, 1].values
totun_north = df.iloc[:, 2].values
column_4_north = df.iloc[:, 3].values
column_5_north = df.iloc[:, 4].values

#globe
years_globe = dt.iloc[:,0].values
ta_globe = dt.iloc[:,1].values
totun_globe = df.iloc[:,2].values


# #### Import HURDAT2

# In[3]:


#importing HURDAT2
# URL of the NOAA text website
url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-050423.txt"

# Fetch the webpage content
request = requests.get(url, allow_redirects=True)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(request.text, "html.parser")
request.status_code      #check website allows access

if request.status_code==200 and request.status_code==200:  #200 means the request is successful. 404 unsuccessful
    print('success!')
else: print('no access')

data = request.text  # Get the text content from the request
lines = data.split('\n')  # Split the text into lines
hurridata = [line.split(',') for line in lines]
print(hurridata[0][0])


# In[4]:


#IT WILL GIVE ERROR BC hurridata that starts with AL is out of range, but the lists are appended
hurricane_lists = []

for i in range(len(hurridata)):
    if hurridata[i][0].startswith('AL'): #if the first element of the list starts with 'AL', append the storm identifier
        identifier = hurridata[i][0]
    else: #measurement variable data included in a different list
        if hurridata[i][3] == ' HU': #if classified as hurricane, append desired variables of the list
            storm_info = [identifier, hurridata[i][0], hurridata[i][1], hurridata[i][2], hurridata[i][3], hurridata[i][4], hurridata[i][5], hurridata[i][6]]
            hurricane_lists.append(storm_info) #a list of lists of hurricane measurements
        if len(hurridata) == i:
            break


# In[5]:


wsall = {}  #Dictionary to store windspeed values for all storms as sublists that contain date and
            #all windspeeds for an individual storm
for sublist in hurricane_lists:
    key = sublist[1] #check that the list contains the storm date
    windspeed_value = float(sublist[7]) #append the windspeed

    if key not in wsall: #if the key doesn't match we're in the next storm, so initialise a new list for the storm
        wsall[key] = [key, [windspeed_value]]
    else: #if it has the same key, append the windspeed into the existing list
        wsall[key][1].append(windspeed_value) 

#Convert the dictionary values to a list of lists. Each sublist belongs to one storm.
wsall_lists = [data for key, data in wsall.items()]

#uncomment next line to check result
#print(wsall_lists)


# In[6]:


#List to store storm information including date with maximum, average, and standard deviation of the wind speed
hurricane_calculations = []

for storm_data in wsall_lists:
    storm_date = storm_data[0]
    wind_speed_values = storm_data[1]  #Extract wind speed values for the storm

    #Calculate maximum, average, and standard deviation of wind speed values
    max_wind_speed = max(wind_speed_values) #maximum
    avg_wind_speed = statistics.mean(wind_speed_values) #average
    std_dev_wind_speed = statistics.stdev(wind_speed_values) if len(wind_speed_values) > 1 else 0  #standard deviation, handling case for a single wind speed value
    
    #convert from knots to m/s units using conversion 
    storm_info = [storm_date, max_wind_speed*0.514444, avg_wind_speed*0.514444, std_dev_wind_speed*0.514444]  #Store information altogether in a list
    hurricane_calculations.append(storm_info) #list of lists

#Print the resulting list of lists. Uncomment next line to check results
#print(hurricane_calculations)


# In[7]:


max_wind_speeds = [item[1] for item in hurricane_calculations]

years = [datetime.strptime(str(item[0]), '%Y%m%d').year for item in hurricane_calculations]
max_wind_speeds = [float(item[1]) for item in hurricane_calculations]

# Aggregating maximum wind speeds by year
aggregated_data = {}
for year, speed in zip(years, max_wind_speeds):
    if year not in aggregated_data:
        aggregated_data[year] = [speed]
    else:
        aggregated_data[year].append(speed)

avg_wind_speeds = [float(item[2]) for item in hurricane_calculations]

# Aggregating average wind speeds by year
aggregated_data = {}
for year, speed in zip(years, avg_wind_speeds):
    if year not in aggregated_data:
        aggregated_data[year] = [speed]
    else:
        aggregated_data[year].append(speed)

# Calculating average wind speed for each year
avg_wind_speed_year = {year: statistics.mean(speeds) for year, speeds in aggregated_data.items()}

# Calculating maximum wind speed for each year
max_wind_speed_by_year = {year: max(speeds) for year, speeds in aggregated_data.items()}

years_hurdat2 = list(max_wind_speed_by_year.keys())  #ARRAY
max_wind_speed_by_year_hurdat2 = list(max_wind_speed_by_year.values()) #ARRAY
avg_wind_speed_by_year_hurdat2 = list(avg_wind_speed_year.values())


# Because the arrays HADSST and HURDAT2 are not the same lenght and HURDAT2 is missing values for 2 years (1907 and 1914) I exclude these values from the other sets)

# In[8]:


years_north_fix = []  # Initialize the new list
ta_north_fix_ = []
exclude_values = {1907, 1914, 2023}  # Set of values to exclude

for index, value in enumerate(years_north[:-1]):
    if value not in exclude_values:
        years_north_fix.append(value)
        nf = np.delete(ta_north, index)
        ta_north_fix_.append(nf)
#print(len(ta_north_fix))       
  


# In[9]:


exclude_values = {1907, 1914, 2023}

# Step 1: Find indices of exclude_values in years_north
indices_to_remove = [i for i, value in enumerate(years_north) if value in exclude_values]

# Step 2: Remove values from years_north
years_north = [value for i, value in enumerate(years_north) if i not in indices_to_remove]


# Step 3: Remove corresponding values from ta_north
ta_north = [value for i, value in enumerate(ta_north) if i not in indices_to_remove]
ta_globe = [value for i, value in enumerate(ta_globe) if i not in indices_to_remove]


# Important, there are years missing in the HURDAT2 array: 1907, 1914, 2023

# ### Pearson correlation coefficient

# In[10]:


# DATASETS USED
# years_north
# ta_north 
# totun_north 
# ta_globe 
# totun_globe 
# years_hurdat2 
# max_wind_speed_by_year_hurdat2 
# avg_wind_speed_by_year_hurdat2 

#correlation for max wind speed & temp anomaly NORTH
correlation_coefficient_max_ta_north = np.corrcoef(max_wind_speed_by_year_hurdat2, ta_north)[0, 1]

#correlation for max wind speed & temp anomaly GLOBE
correlation_coefficient_max_ta_globe = np.corrcoef(max_wind_speed_by_year_hurdat2, ta_globe)[0, 1]

#correlation for avg speed & temp anomaly NORTH
correlation_coefficient_avg_ta_north = np.corrcoef(avg_wind_speed_by_year_hurdat2 , ta_north)[0, 1]

#correlation for avg speed & temp anomaly GLOBE
correlation_coefficient_avg_ta_globe = np.corrcoef(avg_wind_speed_by_year_hurdat2 , ta_globe)[0, 1]


print("Pearson correlation coefficient for SST north and MAX WS North:", correlation_coefficient_avg_ta_globe)
print("Pearson correlation coefficient for SST globe and MAX WS globe:", correlation_coefficient_max_ta_globe)
print("Pearson correlation coefficient for SST north and MEAN WS north:", correlation_coefficient_avg_ta_north)
print("Pearson correlation coefficient for SST globe and MEAN WS globe", correlation_coefficient_avg_ta_north)


# In[11]:


print(len(max_wind_speed_by_year_hurdat2), len(ta_north))


# In[ ]:




