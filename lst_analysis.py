import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fiona
import sys
import os
from pprint import pprint

gpkg_path = r'/home/tereza/Documents/data/LANDSAT/RESULTS/2022/lst_analysis/lst_2022.gpkg'

# Create a list to store temporal data
temporal_data = []

# Read each layer and extract statistics
for layer in fiona.listlayers(gpkg_path):
    gdf = gpd.read_file(gpkg_path, layer=layer)
    date = os.path.basename(layer)  # Extract date from layer name
    
    # Calculate mean values for each category
    stats = gdf.groupby('cat')['_mean'].mean().reset_index()
    stats['date'] = pd.to_datetime(date)
    temporal_data.append(stats)

# Combine all temporal data
df_temporal = pd.concat(temporal_data)

# Define custom colors for categories
category_colors = {
    'agricultural': 'orange',
    'artificial': 'red',
    'vegetation': 'green',
    'meadows': 'brown',
    'water': 'blue',
    'other': 'gray'
}

# Plot temperature trends
plt.figure(figsize=(14, 7))
sns.set_style("darkgrid")
sns.lineplot(
    data=df_temporal, 
    x='date', 
    y='_mean', 
    hue='cat', 
    palette=category_colors, 
    ci=95,
    
)
plt.title('Surface Temperature in 2022')
plt.xlabel('Date')
plt.ylabel('Mean Temperature [˚C]')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate summary statistics for each category
summary_stats = df_temporal.groupby('cat').agg({
    '_mean': ['mean', 'std', 'min', 'max']
}).round(2)

print("\
Summary Statistics by Category:")
print(summary_stats)

# Create box plots for temperature distributions by category
plt.figure(figsize=(14, 7))
sns.boxplot(
    data=df_temporal, 
    x='cat', 
    y='_mean', 
    palette=category_colors
)
plt.title('Temperature Distributions by Category')
plt.xlabel('Category')
plt.ylabel('Mean Temperature [˚C]')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()