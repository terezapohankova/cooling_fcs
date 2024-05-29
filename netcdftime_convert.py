import netCDF4 as nc
from datetime import datetime, timedelta
from pprint import pprint

# Function to convert hours since 1900-01-01 to a readable date
def hours_since_1900_to_date(hours):
    reference_date = datetime(1900, 1, 1)
    return reference_date + timedelta(hours=hours)

# Open the netCDF file
file_path = '/home/tereza/Downloads/copy.nc'

dataset = nc.Dataset(file_path, 'r')
# Print the structure of the netCDF file
#print("Variables in the netCDF file:")
#for var_name in dataset.variables:
    #print(f"{var_name}: {dataset.variables[var_name].dimensions}")

# Extract the time variable (assuming the variable is named 'time')
time_var = dataset.variables['time'][:]
#print("Time variable:", time_var)

# Extract the band data (assuming the bands are named starting with 'tcmw')
band_names = [var for var in dataset.dimensions]
#print("Band names:", band_names)

# Create a dictionary to store band values and corresponding dates
band_data = {}

# Loop over all band names and their corresponding data
for band_index, band_name in enumerate(band_names, start=1):
    band_values = dataset.variables[band_name][:]
    #print(f"Values for {band_name}: {band_values}")
    band_data[band_index] = []  # Initialize the list for this band
    for i, value in enumerate(band_values):
        #pprint(i)
        hours_since_1900 = time_var[i]
        date = hours_since_1900_to_date(int(hours_since_1900))
        band_data[band_index].append((value, date))

# Close the dataset
dataset.close()

# Print the resulting dictionary
#for band, values in band_data.items():
    #print(f"Band {band}:")
    #for value, date in values:
        #print(f"  Value: {value}, Date: {date}")

pprint(band_data[3])

import csv
with open('/home/tereza/Downloads/dict.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in band_data.items():
       writer.writerow([key, value])