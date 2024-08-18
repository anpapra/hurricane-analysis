#!/usr/bin/env python
# coding: utf-8

# ## Analysing hurricane data - case study
# #### HadCRUT data
# 
# HadCRUT is a database providing gridded temperature anomalies across the world and hemispheres. The CRUTEM data is for land temperatures, whereas HadSST is for ocean regions.
# It is important to note these datasets were updated in 2020, increasing the number of weather stations and reducing the bias in sea temperatures. Refer to [1].
# 
# Values for the hemisphere are the weighted average of all the non-missing, grid-box anomalies in each hemisphere. The weights used are the cosines of the central latitudes of each grid box.
# 
# ##### Data collection and processing
# The database recovers information of temperature from weather stations, buoys and satellites across the world to generate a spatial grid that attributes a temperature to each cell. Each grid cell corresponds to a 5° latitude by 5° longitude area [1], and for those cells where measurements are missing, a temperature is obtained from data interpolation of adjacent (?) cells.
# 
# It calculates the difference in temperature with respect to a climatically stable period, for example, 1961-1990. The data is updated every month, and averaged temperatures are subtracted to the average temperature of the reference period for each grid cell, producing a temperature anomaly measurement for each timestep. Extended timeseries datasets are used to represent hemispheric and global temperature evolution.
#  
# Uncertainties are produced for the datasets. These are updated over time. The last update was 2020. Comparisons between studies of sea-surface temperatures (SST) recently showed the differences between the data sets surpassed the magnitude of the associated uncertainties, implying a bias in the measurements that is yet to be accounted for. HadSST3 and COBE-SST-2 underestimated global warming [1].
# 
# Assessments of uncertainty in global and regional average temperature changes have found that sparse data coverage is the most prominent source of uncertainty over monthly to decadal timescales (Brohan et al., 2006; Morice et al., 2012), outweighing those from measurement methods [1].
# 
# Methods to infer spatial fields from scattered observation data analysis use knowledge of the covariance structure of spatial fields to infer field values as weighted averages of observations in locations with strong covariation. Typically, weighting is based on a statistical model in which nearby locations are expected to covary strongly and distant locations weakly.[1]
# 
# Efforts are put towards the unification of datasets using bot in situ SST and satellite measurements. 

# In[25]:


#pip install xarray==2023.02.0  this is the compatible version with geocat
#pip install geocat-comp geocat-viz
#pip install nclpy
#pip install netCDF4
#pip install fiona


# In[26]:


import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# In[27]:


#geocat packages
import geocat.datafiles as gdf
import geocat.viz as gv

import cartopy.crs as ccrs
import matplotlib.animation as animation
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

import geocat.datafiles as gdf
import geocat.viz as gv


# In[28]:


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
from IPython.display import HTML


# #### Trying a Geocat example

# In[29]:


#solveeeeeed!! giiirl I am a geniuuuus!! B) HARD WORK


# In[30]:


# Open a netCDF data file using xarray default engine and load the data into xarrays
ds = xr.open_dataset(gdf.get("netcdf_files/atmos.nc"), engine='netcdf4', decode_times=False)
t = ds.TS.isel(time=0)


# In[31]:


wrap_t = gv.xr_add_cyclic_longitudes(t, "lon")


# In[32]:


# Generate figure (set its size (width, height) in inches)
fig = plt.figure(figsize=(10, 10))

# Generate axes using Cartopy and draw coastlines
ax = plt.axes(
    projection=ccrs.Mercator(central_longitude=0, min_latitude=-87.8638))

# Add coastlines
ax.coastlines(linewidths=0.5)

# Set extent of the projection
ax.set_extent([0, 359, -84.5, 89], crs=ccrs.PlateCarree())

# Draw gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5)

# Manipulate latitude and longitude gridline numbers and spacing
gl.ylocator = mticker.FixedLocator(np.arange(-84.5, 91, 20))
gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))

# Contourf-plot data (for filled contours)
wrap_t.plot.contourf(ax=ax,
                     transform=ccrs.PlateCarree(),
                     levels=12,
                     cmap='inferno',
                     add_colorbar=False)

# Contour-plot data (for borderlines)
wrap_t.plot.contour(ax=ax,
                    transform=ccrs.PlateCarree(),
                    levels=12,
                    linewidths=0.5,
                    cmap='black')

# Use geocat.viz.util convenience function to add titles to left and right
# of the plot axis.
gv.set_titles_and_labels(ax,
                         maintitle="Example of Mercator Projection",
                         lefttitle="Surface Temperature",
                         righttitle="K")

# Show the plot
plt.show()


# #### Trying another example

# In[33]:


# Open a netCDF data file using xarray default engine and load the data into xarrays
ds = xr.open_dataset(gdf.get("netcdf_files/atmos.nc"), decode_times=False)
t = ds.TS.isel(time=0)

# Fix the artifact of not-shown-data around 0 and 360-degree longitudes
wrap_t = gv.xr_add_cyclic_longitudes(t, "lon")


# In[34]:


#color maps
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])]


# In[35]:


# Open a netCDF data file using xarray default engine and load the data into xarrays
ds = xr.open_dataset(gdf.get("netcdf_files/atmos.nc"), decode_times=False)
t = ds.TS.isel(time=0)

# Fix the artifact of not-shown-data around 0 and 360-degree longitudes
wrap_t = gv.xr_add_cyclic_longitudes(t, "lon")

# Generate figure (set its size (width, height) in inches)
fig = plt.figure(figsize=(10, 10))

# Generate axes using Cartopy and draw coastlines
ax = plt.axes(projection=ccrs.Mollweide())
ax.coastlines(linewidths=0.5)

# Draw gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5)

# Import an NCL colormap
#newcmp = cmaps.gui_default
# Accessing the 'inferno' colormap from the list
selected_colormap = cmaps[0][1][3]  # Indexing: Group 0 -> 'Perceptually Uniform Sequential' -> 'inferno'

# Assign the selected colormap to newcmp
newcmp = plt.get_cmap(selected_colormap)

# Contourf-plot data (for filled contours)
temp = wrap_t.plot.contourf(ax=ax,
                            transform=ccrs.PlateCarree(),
                            levels=11,
                            cmap=newcmp,
                            add_colorbar=False)

# Add color bar
cbar_ticks = np.arange(220, 310, 10) #starting value of the colorbar, end value, step spacing between the colorbar number scale
cbar = plt.colorbar(temp, #plotting the mesh of temperature
                    orientation='horizontal',
                    shrink=0.8, #modifies length of color bar legend
                    pad=0.05, #spacing between map and colorbar
                    extendrect=True,
                    ticks=cbar_ticks,
                    drawedges=True)

cbar.ax.tick_params(labelsize=10) #font size of the colorbar values

# Contour-plot data (for borderlines)
wrap_t.plot.contour(ax=ax,
                    transform=ccrs.PlateCarree(),
                    levels=10, #number of lines that separate the colour region. 1 for only one line separating lower values. 10 for 10 lines
                    linewidths=0.5, #width of the lines
                    cmap='black') #line color

# Use geocat.viz.util convenience function to add titles to left and right of the plot axis.
gv.set_titles_and_labels(ax,
                         maintitle="Mollweide Projection",
                         lefttitle="Surface Temperature",
                         righttitle="K")

# Show the plot
plt.show()


# #### Another example for cylindrical projection

# In[36]:


# Open a netCDF data file using xarray default engine and load the data into xarrays
# Disable time decoding due to missing necessary metadata
ds = xr.open_dataset(gdf.get("netcdf_files/meccatemp.cdf"), decode_times=False)

tas = ds.t


# In[37]:


fig = plt.figure(figsize=(10, 8))
# Generate axes using Cartopy and draw coastlines
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=150))
ax.coastlines(linewidths=0.5)
ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

# Use geocat.viz.util convenience function to set axes limits & tick values
gv.set_axes_limits_and_ticks(ax,
                             xlim=(-180, 180),
                             ylim=(-90, 90),
                             xticks=np.linspace(-180, 180, 13),
                             yticks=np.linspace(-90, 90, 7))

# Use geocat.viz.util convenience function to add minor and major tick lines
gv.add_major_minor_ticks(ax, labelsize=10)

# Use geocat.viz.util convenience function to make latitude, longitude tick labels
gv.add_lat_lon_ticklabels(ax)

# create initial plot that establishes a colorbar
tas[0, :, :].plot.contourf(ax=ax,
                           transform=ccrs.PlateCarree(),
                           vmin=195,
                           vmax=328,
                           levels=53,
                           cmap="inferno",
                           cbar_kwargs={
                               "extendrect": True,
                               "orientation": "horizontal",
                               "ticks": np.arange(195, 332, 9),
                               "label": "",
                               "shrink": 0.90
                           })


# #### Visualising the hadCRUT data with geocat: Mollweide Projection

# In[38]:


# Open the NetCDF file
file_path = 'C:/Users/anapa/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc'  # Replace with your file path
data = nc.Dataset(file_path, 'r')  # 'r' for reading mode

# Get information about the file
print(data)  # This prints basic information about the NetCDF file
print('Available variables:')
print(data.variables.keys())


# In[39]:


#some statistics of the file to determine the range of temperatures in the plot
# Replace this with your file path
file_path = 'C:/Users/anapa/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc'

# Load the NetCDF file using xarray
ds = xr.open_dataset(file_path)

# Extract temperature variable
t = ds.tas_mean

# Calculate statistics to understand the data range
print("Minimum Temperature:", t.min().values)
print("Maximum Temperature:", t.max().values)
print("Mean Temperature:", t.mean().values)
print("Standard Deviation of Temperature:", t.std().values)


# In[40]:


len(ds.time)


# In[41]:


#load variables
tas_mean = ds['tas_mean']
latitude = ds['latitude']
longitude = ds['longitude']

#Identify missing data points
missing_data = np.isnan(tas_mean)

#print(missing_data)
#Now if we uncomment the previous line, we will find that many of the entries are missing.
#This means we have to fill in the values somehow. I choose to do interpolation because it provides a straightforward
#way of calculating the missing values using a variety of methods of choice: linear, nearest, cubic...
#Ideally I would like to calculate them using information of their nearest neighbours


# In[42]:


#Get the coordinates and values of the known data points
lon, lat = np.meshgrid(longitude, latitude)
coords = np.array([lon.flatten(), lat.flatten()]).T
values = tas_mean.values.flatten()


# In[43]:


len(coords)


# In[44]:


mask = ~missing_data.values.flatten()
print(mask)


# In[45]:


# Separate data and coordinates
lon = data.coords['longitude']
lat = data.coords['latitude']
data_array = data.values.flatten()
# Select valid data points (not missing)
valid_data = data_array[~missing.flatten()]
valid_lon = lon[~missing.flatten()]
valid_lat = lat[~missing.flatten()]


# In[ ]:





# In[46]:


# Assuming 'temperature_anomaly' is the temperature variable, replace it with the actual variable name in your dataset
# Also, replace 'time=0' with the appropriate time index if needed
t = ds.tas_mean.isel(time=2084)



# Fix the artifact of not-shown-data around 0 and 360-degree longitudes
wrap_t = gv.xr_add_cyclic_longitudes(t, "longitude")

# Generate figure
fig = plt.figure(figsize=(10, 10))

# Generate axes using Cartopy and draw coastlines
ax = plt.axes(projection=ccrs.Mollweide())
ax.coastlines(linewidth=0.5)

# Draw gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5)

# Contourf-plot data for filled contours
temp = wrap_t.plot.contourf(ax=ax,
                            transform=ccrs.PlateCarree(),
                            levels= (np.abs(t.min().values - t.max().values)/0.5),
                            cmap='inferno',
                            add_colorbar=False)

# Add color bar
cbar_ticks = np.arange(t.min().values, t.max().values, 0.5)  # Adjust based on the range of your temperature data 
cbar = plt.colorbar(temp,
                    orientation='horizontal',
                    shrink=0.8,
                    pad=0.05,
                    extendrect=True,
                    ticks=cbar_ticks,
                    drawedges=True)
cbar.ax.tick_params(labelsize=10)

# Contour-plot data for borderlines
wrap_t.plot.contour(ax=ax,
                    transform=ccrs.PlateCarree(),
                    levels=12, #number of lines that separate the colour region. 1 for only one line separating lower values. 10 for 10 lines
                    linewidths=0.5, #width of the lines
                    cmap='black') #color

# Add titles to the plot
gv.set_titles_and_labels(ax,
                         maintitle="Mollweide Projection of Sea-Surface Temperature",
                         #lefttitle="Surface Temperature",
                         #righttitle="K"
                        )

# Show the plot
plt.show()


# In[47]:


np.abs(t.min().values - t.max().values)/0.5


# #### Visualising the hadCRUT data with geocat: Cylindrical Projection

# In[48]:


# Open a netCDF data file using xarray default engine and load the data into xarrays
# Disable time decoding due to missing necessary metadata
file_path = 'C:/Users/anapa/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc'

# Load the NetCDF file using xarray
ds = xr.open_dataset(file_path, decode_times=False)

# Assuming 'temperature_anomaly' is the temperature variable, replace it with the actual variable name in your dataset
# Also, replace 'time=0' with the appropriate time index if needed
tee = ds.tas_mean.isel(time=2084)


# In[49]:


fig = plt.figure(figsize=(10, 8))

# Generate axes using Cartopy and draw coastlines
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=150))
ax.coastlines(linewidths=0.5)
ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

# Use geocat.viz.util convenience function to set axes limits & tick values
gv.set_axes_limits_and_ticks(ax,
                             xlim=(-180, 180),
                             ylim=(-90, 90),
                             xticks=np.linspace(-180, 180, 13),
                             yticks=np.linspace(-90, 90, 7))

# Use geocat.viz.util convenience function to add minor and major tick lines
gv.add_major_minor_ticks(ax, labelsize=10)

# Use geocat.viz.util convenience function to make latitude, longitude tick labels
gv.add_lat_lon_ticklabels(ax)

# create initial plot that establishes a colorbar
tee.plot.contourf(ax=ax,
                   transform=ccrs.PlateCarree(),
                   vmin= tee.min().values,
                   vmax= tee.max().values,
                   levels=53,
                   cmap="inferno",
                   cbar_kwargs={
                       "extendrect": True,
                       "orientation": "horizontal",
                       "ticks": np.arange(195, 332, 9),
                       "label": "",
                       "shrink": 0.90
                           })


# ## SCRAPING NOAA HURRICANE INTENSITY DATA

# In[50]:


import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter, YearLocator
from datetime import datetime



# In[51]:


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



# In[52]:


data = request.text  # Get the text content from the request
lines = data.split('\n')  # Split the text into lines
hurridata = [line.split(',') for line in lines]
print(hurridata[0][0])


# In[53]:


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


# In[54]:


print(len(hurricane_lists)) #check lists are appended
print(hurricane_lists[0])


# In[55]:


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


# In[56]:


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


# ### Raw data maximum wind speed agains time

# In[57]:


# Assuming you have storm_info_list as generated in the previous code

# Extracting storm identifier (date) and maximum wind speed for plotting
dates = [datetime.strptime(str(item[0]), '%Y%m%d') for item in hurricane_calculations]
max_wind_speeds = [item[1] for item in hurricane_calculations]

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(15, 4))

# Plot the data with a line connecting the dots
ax.plot(dates, max_wind_speeds, linestyle='-', color='darkblue')

# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Maximum Wind Speed (m/s)')
ax.set_title('Maximum wind speed achieved by hurricane storms over time in the Atlantic Ocean')

# Formatting x-axis with years
years = plt.dates.YearLocator()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(plt.dates.DateFormatter('%Y'))

#Adjust plot size in the x-axis onyl
plt.subplots_adjust(left=0.5, right=0.9, bottom=0.1, top=2)  # Adjust the parameters as needed

# Rotate and align the x-labels for better readability
#plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


# Create a DataFrame with dates and maximum wind speeds
data = {'Date': dates, 'MaxWindSpeed': max_wind_speeds}
df = pd.DataFrame(data)

# Extract year from the date
df['Year'] = df['Date'].dt.year

# Group by year and find the maximum wind speed for each year
max_speed_by_year = df.groupby('Year')['MaxWindSpeed'].max()

# Convert years to datetime objects for plotting
years = [datetime(year, 1, 1) for year in max_speed_by_year.index]


# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed

# Plot the maximum wind speed by year
ax.plot(years, max_speed_by_year, linestyle='-', marker='o', color='b')

# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Maximum Wind Speed')
ax.set_title('Maximum Wind Speed per Year')

# Formatting x-axis with years
years_locator = YearLocator()
ax.xaxis.set_major_locator(years_locator)
ax.xaxis.set_major_formatter(DateFormatter('%Y'))


# Assuming 'dates' contains the dates for the x-axis
years = YearLocator(base=10)  # Show years by decades
plt.gca().xaxis.set_major_locator(years)

date_format = DateFormatter("%Y")  # Format for displaying years
plt.gca().xaxis.set_major_formatter(date_format)
plt.xlim([datetime.datetime(1900, 1, 1), datetime.datetime(2020, 1, 1)])  # Set the range up to 2020

plt.tight_layout()  # Adjust layout for better spacing

plt.show()



# Plotting the data
plt.plot(dates, max_wind_speeds, linestyle='-', marker='', color='blue')  # Adjust the line plot settings

# Formatting the x-axis
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))  # Show ticks every 10 years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format x-axis to display years


plt.tight_layout()  # Adjust layout for better spacing
plt.show()



# In[58]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as datetime

# Convert dates to datetime objects
dates = [datetime.datetime.strptime(str(item[0]), '%Y%m%d') for item in hurricane_calculations]

# Plotting the data
plt.plot(dates, max_wind_speeds, linestyle='-', marker='', color='blue')  # Adjust the line plot settings

# Formatting the x-axis
years_to_display = [1850, 1870, 1890, 1910, 1920, 1940, 1960, 1980, 2000, 2020]
plt.xticks([datetime.datetime(year, 1, 1) for year in years_to_display], years_to_display)  # Set specific years as x-axis ticks

plt.xlim([datetime.datetime(1840, 1, 1), datetime.datetime(2030, 1, 1)])  # Set x-axis range

plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# ### Maximum wind speed per year against time
# Calculating the maximum wind speed in each year by aggregating the data, then plotting it against each year with only specific year labels showing.

# In[59]:


# Extracting and aggregating data by year
years = [datetime.datetime.strptime(str(item[0]), '%Y%m%d').year for item in hurricane_calculations]
max_wind_speeds = [float(item[1]) for item in hurricane_calculations]

# Aggregating maximum wind speeds by year
aggregated_data = {}
for year, speed in zip(years, max_wind_speeds):
    if year not in aggregated_data:
        aggregated_data[year] = [speed]
    else:
        aggregated_data[year].append(speed)

# Calculating maximum wind speed for each year
max_wind_speed_by_year = {year: max(speeds) for year, speeds in aggregated_data.items()}


plt.figure(figsize=(18, 6))  # Set the figure size
plt.plot(list(max_wind_speed_by_year.keys()), list(max_wind_speed_by_year.values()), linestyle='-', marker='', color='blue')
plt.xlabel('Year', fontsize=20)
plt.ylabel('Maximum Wind Speed (m/s)', fontsize=20)
plt.title('Maximum wind speed by year for hurricane storms in the Atlantic Ocean', fontsize=24)
plt.xticks([1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990, 2010, 2030], fontsize=18)  # Set specific years as x-axis ticks
plt.yticks(fontsize=18)
plt.grid(False)

plt.tight_layout()
plt.show()


# In[60]:


import numpy as np


# In[61]:


mwsy = np.array(list(max_wind_speed_by_year.values()))
years_x = np.array(list(max_wind_speed_by_year.keys()))
def my_mean(sample):
    return sum(sample) / len(sample)

n = len(sample)
z = 0.95
std = np.std(sample)
CI_high = my_mean(mwsy) + z*(std/np.sqrt(n))
CI_low = my_mean(mwsy) - z*(std/np.sqrt(n))
#CI equation CI = mean +- z*std/sqrt(n)


# In[62]:


# Print the coefficients
print(coefficients)


# In[63]:


from scipy import stats 


# In[64]:


max_wind_speed_by_year = {year: max(speeds) for year, speeds in aggregated_data.items()}

ci = []
for y in max_wind_speed_by_year:
    confidence_interval = stats.norm.interval(0.95, loc=np.mean(list(max_wind_speed_by_year.values())), scale=stats.sem(list(max_wind_speed_by_year.values())))
    confidence_interval[year] = confidence_interval
    ci.append(confidence_interval[year])


# In[65]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Extracting and aggregating data by year
years = [datetime.datetime.strptime(str(item[0]), '%Y%m%d').year for item in hurricane_calculations]
max_wind_speeds = [float(item[1]) for item in hurricane_calculations]

# Aggregating maximum wind speeds by year
aggregated_data = {}
for year, speed in zip(years, max_wind_speeds):
    if year not in aggregated_data:
        aggregated_data[year] = [speed]
    else:
        aggregated_data[year].append(speed)

# Calculating maximum wind speed for each year
max_wind_speed_by_year = {year: max(speeds) for year, speeds in aggregated_data.items()}

# Fitting a polynomial
degree = 10  # You can adjust the degree based on the trend you observe
coefficients = np.polyfit(list(max_wind_speed_by_year.keys()), list(max_wind_speed_by_year.values()), degree)

# Creating a polynomial function
poly_function = np.poly1d(coefficients)

# Calculating the 95% confidence interval
y_pred = poly_function(list(max_wind_speed_by_year.keys()))
y_err = np.array(list(max_wind_speed_by_year.values())) - y_pred
confidence_interval = 1.96 * sem(y_err)

# Plotting the data and polynomial fit
plt.figure(figsize=(18, 6))
plt.plot(list(max_wind_speed_by_year.keys()), list(max_wind_speed_by_year.values()), linestyle='-', marker='', color='blue')
plt.plot(list(max_wind_speed_by_year.keys()), poly_function(list(max_wind_speed_by_year.keys())), color='red', label='Polynomial Fit')
plt.fill_between(list(max_wind_speed_by_year.keys()), poly_function(list(max_wind_speed_by_year.keys())) - confidence_interval,
                 poly_function(list(max_wind_speed_by_year.keys())) + confidence_interval, color='red', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Year', fontsize=20)
plt.ylabel('Maximum Wind Speed (m/s)', fontsize=20)
plt.title('Maximum wind speed by year for hurricane storms in the Atlantic Ocean', fontsize=24)
plt.xticks([1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990, 2010, 2030], fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.show()


# In[81]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Extracting and aggregating data by year
years = [datetime.datetime.strptime(str(item[0]), '%Y%m%d').year for item in hurricane_calculations]
max_wind_speeds = [float(item[1]) for item in hurricane_calculations]

# Aggregating maximum wind speeds by year
aggregated_data = {}
for year, speed in zip(years, max_wind_speeds):
    if year not in aggregated_data:
        aggregated_data[year] = [speed]
    else:
        aggregated_data[year].append(speed)

# Calculating maximum wind speed for each year
max_wind_speed_by_year = {year: max(speeds) for year, speeds in aggregated_data.items()}

# Fitting a polynomial
degree = 2  # You can adjust the degree based on the trend you observe
coefficients = np.polyfit(list(max_wind_speed_by_year.keys()), list(max_wind_speed_by_year.values()), degree)

# Creating a polynomial function
poly_function = np.poly1d(coefficients)

# Calculating the 95% confidence interval for the polynomial fit
y_pred = poly_function(list(max_wind_speed_by_year.keys()))
y_err = np.array(list(max_wind_speed_by_year.values())) - y_pred
confidence_interval = 1.96 * sem(y_err)

# Calculating the 95% confidence interval for the data
data_err = np.array(max_wind_speeds)[:len(y_pred)] - y_pred
data_confidence_interval = 1.96 * sem(data_err)

# Plotting the data and polynomial fit
plt.figure(figsize=(18, 6))
plt.plot(list(max_wind_speed_by_year.keys()), list(max_wind_speed_by_year.values()), linestyle='-', marker='.', color='blue', label='Maxima')
plt.plot(list(max_wind_speed_by_year.keys()), poly_function(list(max_wind_speed_by_year.keys())), color='red', label='PF')
plt.fill_between(list(max_wind_speed_by_year.keys()), poly_function(list(max_wind_speed_by_year.keys())) - confidence_interval,
                 poly_function(list(max_wind_speed_by_year.keys())) + confidence_interval, color='red', alpha=0.2, label='95% CI of the PF')
plt.fill_between(list(max_wind_speed_by_year.keys()), np.array(list(max_wind_speed_by_year.values())) - data_confidence_interval,
                 np.array(list(max_wind_speed_by_year.values())) + data_confidence_interval, color='blue', alpha=0.2, label='95% CI of the Maxima')
plt.xlabel('Year', fontsize=21)
plt.ylabel('Wind Speed (m/s)', fontsize=21)
plt.title('d) Maximum Hurricane Wind Speed in the North Atlantic Ocean Basin',loc='left', fontsize=21)
plt.xticks([1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025], fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize='x-large')
plt.grid(False)

plt.tight_layout()
plt.savefig('max_wind_speed_Atlantic.png')  #Save the plot as a PNG image file
plt.show()
# Display download link
HTML(f'<a href="max_wind_speed_Atlantic.png" download>Click here to download the plot</a>')


# In[67]:


print("Slope of the polynomial fit:", coefficients[0])


# Plot above: Maximum wind speed by year for hurricane storms in the Atlantic Ocean

# ### Moving averages for maximum wind speed

# In[68]:


# Assuming 'dates' is a list of datetime objects and 'wind_speed' contains corresponding wind speed values

# Create a DataFrame from the data
data = {'Date': dates, 'MaxWindSpeed': max_wind_speeds}
df = pd.DataFrame(data)

# Set the 'Date' column as the DataFrame index (if not already set)
df.set_index('Date', inplace=True)

# Calculate the moving average using rolling window
window_size = 30  # Adjust window size as needed
df['Moving Average'] = df['MaxWindSpeed'].rolling(window=window_size).mean()

# Plot the original data and the moving average
plt.figure(figsize=(20, 6))
plt.plot(df.index, df['MaxWindSpeed'], label='Original Wind Speed')
plt.plot(df.index, df['Moving Average'], label=f'Moving Average (Window Size={window_size})', color='red')
plt.xlabel('Date')
plt.ylabel('MaxWindSpeed')
plt.title('MaxWindSpeed with Moving Average')
plt.legend()
plt.grid(True)
plt.show()


# ### Fourier analysis to detect oscillatory patters in the data

# In[69]:


import numpy as np
import matplotlib.pyplot as plt


# In[70]:


#Step 1: Data Preparation (Already given 'dates' and 'max_wind_speeds' arrays)

#list(max_wind_speed_by_year.keys()) contains maximum wind speed by year
#list(max_wind_speed_by_year.values()) contains year dates

# Assuming max_wind_speed_by_year.keys() and max_wind_speed_by_year.values() contain the data
data_keys = np.array(list(max_wind_speed_by_year.keys()))
data_values = np.array(list(max_wind_speed_by_year.values()))

# Step 2: Apply Fourier Transform
fft_result = np.fft.fft(data_keys)
freqs = np.fft.fftfreq(len(data_values))


# In[71]:


# Step 3: Identify Periodic Components
power_spectrum = np.abs(fft_result) ** 2
threshold = 0.01 * np.max(power_spectrum)  # Example threshold (Adjust as needed)

significant_freqs = freqs[power_spectrum > threshold]

# Step 4: Visualize Results
plt.figure(figsize=(10, 6))

plt.subplot(211)
plt.plot(dates, max_wind_speeds, marker='o', linestyle='-')
plt.title('Original Wind Speed Data')
plt.xlabel('Year')
plt.ylabel('Max Wind Speed')

plt.subplot(212)
plt.xlim(-0.025,0.025)
plt.plot(freqs, power_spectrum)
plt.title('Power Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Power')

# Step 5: Remove Periodic Components (Not implemented in this example)

# Step 6: Validate Results (Not implemented in this example)

plt.tight_layout()
plt.show()


# In[72]:


significant_freqs


# In[73]:


# Perform FFT on the wind speed data
fft_result = np.fft.fft(list(max_wind_speed_by_year.keys()))
power_spectrum = np.abs(fft_result) ** 2

# Calculate frequency range
N = len(list(max_wind_speed_by_year.keys()))
time_step = 1  # assuming data collected at regular intervals
freqs = np.fft.fftfreq(N, d=time_step)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(freqs, power_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.xlim(-0.001,0.0075)
plt.title('Power Spectrum of Hurricane Wind Speed Data')
plt.grid(True)
plt.show()


# ### HURDAT2 average wind speed analysis

# In[74]:


import datetime
import statistics
import matplotlib.pyplot as plt

# Assuming hurricane_calculations is already defined

# Extracting and aggregating data by year
dates = [datetime.datetime.strptime(str(item[0]), '%Y%m%d') for item in hurricane_calculations]
avg_wind_speeds = [float(item[2]) for item in hurricane_calculations]

# Aggregating average wind speeds by year
aggregated_data = {}
for date, speed in zip(dates, avg_wind_speeds):
    year = date.year
    if year not in aggregated_data:
        aggregated_data[year] = [speed]
    else:
        aggregated_data[year].append(speed)

# Calculating average wind speed for each year
avg_wind_speed_year = {year: statistics.mean(speeds) for year, speeds in aggregated_data.items()}

# Plotting the data
plt.figure(figsize=(18, 6))
plt.plot(list(avg_wind_speed_year.keys()), list(avg_wind_speed_year.values()), linestyle='-', marker='.', color='green')
plt.xlabel('Year', fontsize=20)
plt.ylabel('Average Wind Speed (m/s)', fontsize=20)
plt.title('Average wind speed by year for hurricane storms in the Atlantic Ocean', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

plt.tight_layout()
plt.show()


# In[75]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Extracting and aggregating data by year for avg_wind_speed_year dataset
years_avg = [date.year for date in dates]
avg_wind_speeds = [float(item[2]) for item in hurricane_calculations]

# Aggregating average wind speeds by year
aggregated_data_avg = {}
for year, speed in zip(years_avg, avg_wind_speeds):
    if year not in aggregated_data_avg:
        aggregated_data_avg[year] = [speed]
    else:
        aggregated_data_avg[year].append(speed)

# Calculating average wind speed for each year
avg_wind_speed_year = {year: statistics.mean(speeds) for year, speeds in aggregated_data_avg.items()}

# Fitting a polynomial
degree = 1  # You can adjust the degree based on the trend you observe
coefficients_avg = np.polyfit(list(avg_wind_speed_year.keys()), list(avg_wind_speed_year.values()), degree)

# Creating a polynomial function
poly_function_avg = np.poly1d(coefficients_avg)

# Calculating the 95% confidence interval for the polynomial fit
y_pred_avg = poly_function_avg(list(avg_wind_speed_year.keys()))
y_err_avg = np.array(list(avg_wind_speed_year.values())) - y_pred_avg
confidence_interval_avg = 1.96 * sem(y_err_avg)

# Calculating the 95% confidence interval for the data
data_err_avg = np.array(avg_wind_speeds)[:len(y_pred_avg)] - y_pred_avg
data_confidence_interval_avg = 1.96 * sem(data_err_avg)

# Plotting the data and polynomial fit
plt.figure(figsize=(18, 6))
plt.plot(list(avg_wind_speed_year.keys()), list(avg_wind_speed_year.values()), linestyle='-', marker='.', color='green', label='Data')
plt.plot(list(avg_wind_speed_year.keys()), poly_function_avg(list(avg_wind_speed_year.keys())), color='purple', label='Polynomial Fit')
plt.fill_between(list(avg_wind_speed_year.keys()), poly_function_avg(list(avg_wind_speed_year.keys())) - confidence_interval_avg,
                 poly_function_avg(list(avg_wind_speed_year.keys())) + confidence_interval_avg, color='purple', alpha=0.2, label='95% Confidence Interval (Fit)')
plt.fill_between(list(avg_wind_speed_year.keys()), np.array(list(avg_wind_speed_year.values())) - data_confidence_interval_avg,
                 np.array(list(avg_wind_speed_year.values())) + data_confidence_interval_avg, color='green', alpha=0.2, label='95% Confidence Interval (Data)')
#plt.xlabel('Year', fontsize=21)
plt.ylabel('Average Wind Speed (m/s)', fontsize=21)
plt.title('c) ', fontsize=24)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize='x-large')  # Increase legend font size
plt.grid(False)

plt.tight_layout()



# Plot above: Average wind speed by year for hurricane storms in the Atlantic Ocean

# In[80]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Extracting and aggregating data by year for avg_wind_speed_year dataset
years_avg = [date.year for date in dates]
avg_wind_speeds = [float(item[2]) for item in hurricane_calculations]

# Aggregating average wind speeds by year
aggregated_data_avg = {}
for year, speed in zip(years_avg, avg_wind_speeds):
    if year not in aggregated_data_avg:
        aggregated_data_avg[year] = [speed]
    else:
        aggregated_data_avg[year].append(speed)

# Calculating average wind speed for each year
avg_wind_speed_year = {year: statistics.mean(speeds) for year, speeds in aggregated_data_avg.items()}
median_wind_speed_year = {year: statistics.median(speeds) for year, speeds in aggregated_data_avg.items()}

# Fitting a polynomial
degree = 2  # You can adjust the degree based on the trend you observe
coefficients_avg = np.polyfit(list(avg_wind_speed_year.keys()), list(avg_wind_speed_year.values()), degree)

# Creating a polynomial function
poly_function_avg = np.poly1d(coefficients_avg)

# Calculating the 95% confidence interval for the polynomial fit
y_pred_avg = poly_function_avg(list(avg_wind_speed_year.keys()))
y_err_avg = np.array(list(avg_wind_speed_year.values())) - y_pred_avg
confidence_interval_avg = 1.96 * sem(y_err_avg)

# Calculating the 95% confidence interval for the data
data_err_avg = np.array(avg_wind_speeds)[:len(y_pred_avg)] - y_pred_avg
data_confidence_interval_avg = 1.96 * sem(data_err_avg)

# Plotting the data, polynomial fit, and median
plt.figure(figsize=(18, 6))
plt.plot(list(avg_wind_speed_year.keys()), list(avg_wind_speed_year.values()), linestyle='-', marker='.', color='green', label='Mean')
plt.plot(list(median_wind_speed_year.keys()), list(median_wind_speed_year.values()), linestyle='-', marker='', color='orange', label='Median')
plt.plot(list(avg_wind_speed_year.keys()), poly_function_avg(list(avg_wind_speed_year.keys())), color='purple', label='PF of the Mean')
plt.fill_between(list(avg_wind_speed_year.keys()), poly_function_avg(list(avg_wind_speed_year.keys())) - confidence_interval_avg,
                 poly_function_avg(list(avg_wind_speed_year.keys())) + confidence_interval_avg, color='purple', alpha=0.2, label='95% CI of the PF')
plt.fill_between(list(avg_wind_speed_year.keys()), np.array(list(avg_wind_speed_year.values())) - data_confidence_interval_avg,
                 np.array(list(avg_wind_speed_year.values())) + data_confidence_interval_avg, color='green', alpha=0.2, label='95% CI of the Mean')
#plt.xlabel('Year', fontsize=21)
plt.ylabel('Wind Speed (m/s)', fontsize=21)
plt.title('c) Mean Hurricane Wind Speed in the North Atlantic Ocean Basin',  loc='left', fontsize=21)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize='x-large')  # Increase legend font size
plt.grid(False)

plt.tight_layout()
plt.savefig('mean_median_wind_speed_atlantic.png')  #Save the plot as a PNG image file
plt.show()
# Display download link
HTML(f'<a href="mean_median_wind_speed_atlantic.png" download>Click here to download the plot</a>')


# In[77]:


print("Slope of the polynomial fit:", coefficients_avg[0])


# Note for above plot: Average wind speed by year for hurricane storms in the Atlantic Ocean with median and polynomial fit of degree 4
# 

# #### References
# 
# [1] https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JD032361
# 
# [2] https://github.com/NCAR/geocat-datafiles/tree/main/netcdf_files my geocat repo

# In[ ]:




