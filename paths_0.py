import os                   # Importing the OS module for file and directory operations
import process_func         # Importing custom processing functions

# Define the main input folder where results will be stored
INPUT_FOLDER = r'/home/tereza/Documents/data/LANDSAT/RESULTS'

# Define the path for the directory containing clipped band images
INPUT_DATA = os.path.join(INPUT_FOLDER, 'clipped_bands')

# Get file path for processing Landsat images
JSON_MTL_PATH = process_func.getfilepath(INPUT_DATA, 'MTL.json')  

# Define paths for auxiliary data
AUX_DATA = r'aux_data'
METEOROLOGY = os.path.join(AUX_DATA, 'weather_2024_olomouc.csv')  # Path for meteorological data
HILLSHADE = os.path.join(AUX_DATA, 'hillshade_olomouc_32633.tif')  # Path for hillshade data

# Define a dictionary mapping short folder names to descriptive names
#{working name : local folder name on disc}

FOLDERS = {
    'lst': 'land_surface_temperature', 
    'vegIndices': 'vegetation_indices', 
    'radiation': 'solar_radiation',  
    'preprocess': 'img_preprocessing',  
    'albedo': 'albedo',  
    'fluxes': 'heat_fluxes',  
    'et': 'evapotranspiration',  
    'bowen': 'bowen_ratio',  
    'cci': 'cooling_capacity_index',  
    'lse': 'land_surface_emissivity'  
}
