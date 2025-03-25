import os, sys
import json
from pprint import pprint
import fiona
import rasterio as rio
from osgeo import gdal
import pandas as pd
import shutil

import process_func
import time
import warnings
warnings.filterwarnings('ignore')

# Start time to measure execution duration
start_time = time.time()
pprint('I solemnly swear that I am up to no good...')

# User inputs

OUTPUT_PATH = input(f"Enter the path for the output directory: ").strip() 
INPUT_FOLDER = input(f"Enter the path for the input folder: ").strip() 
AREA_MASK = input(f"Enter the path for the area mask: ").strip() 

JSON_MTL_PATH = process_func.get_band_filepath(INPUT_FOLDER, 'MTL.json') #['root/snimky_L9_testovaci/LC09_L2SP_190025_20220518_20220520_02_T1/LC09_L2SP_190025_20220518_20220520_02_T1_MTL.json']
ORIGINAL_IMG = process_func.get_band_filepath(INPUT_FOLDER, '.TIF') #['root/snimky_L9_testovaci/18052022/LC09_L2SP_190025_20220518_20220518_02_T1_SZA.TIF']
OUT_CLIP_FOLDER = os.path.join(OUTPUT_PATH,'clipped_bands')

# Verify paths are valid
if not os.path.exists(INPUT_FOLDER):
    print("Error: The input folder does not exist.")
    sys.exit(1)
if not os.path.exists(AREA_MASK):
    print("Error: The area mask file does not exist.")
    sys.exit(1)



# Create output folders
os.makedirs(OUT_CLIP_FOLDER, exist_ok = True)


# Initialize empty structures to store metadata and clipped image paths
mtlJSONFile = {}
sensingDate = []
CLIPPED_IMG_PATHS = []

# Load JSON MTL files with metadata and create a dictionary keyed by sensing date {sensingdate : {metadatafile}} for level 2 (level 2 MTL json includes level 1 MTL data)
for jsonFile in JSON_MTL_PATH:
    if 'L2SP' in jsonFile: # Filter for Level-2 metadata files
        loadJSON = process_func.load_json(jsonFile)
        sensDate = jsonFile.split('_')[5] # 20220518
        
        
        # If the sensing date is new, create a corresponding folder in the output directory
        if sensDate not in sensingDate:
            sensingDate.append(sensDate)
            target_path = os.path.join(OUT_CLIP_FOLDER, sensDate)
            
            if not os.path.exists(target_path): 
                os.makedirs(os.path.join(OUT_CLIP_FOLDER, sensDate))
        
        # Store metadata for the sensing date and copy JSON to the output folder
        mtlJSONFile[sensDate] = loadJSON
        shutil.copy(jsonFile, os.path.join(target_path, os.path.basename(jsonFile)))
        
        if os.path.isfile(os.path.join(target_path, os.path.basename(jsonFile))):
            pprint(f'Copying JSON for file {os.path.basename(jsonFile)}')
            continue
        else:
            pprint(f'Error: JSON MTL file not copied: {target_path}')
            pprint(f'Stopping the programme -- JSON copy.')
            sys.exit()

# Create output path for clipped images by pairing sensing date from JSON metadata file and sensing date on original images
for inputBand in ORIGINAL_IMG:
    date = inputBand.split('/')[-1].split('_')[4]
    
   
# Work ith each individual band in each subfolder
for inputBand in ORIGINAL_IMG:
    
    # if date on original input band equals date sensing date from json mtl, then append it to the list
    image_basename = os.path.basename(inputBand) # 'LC09_L1TP_190025_20220518_20220518_02_T1_B6_clipped.TIF'  

    # Filter QA_PIXEL band
    if 'L2SP' and 'QA_PIXEL' in image_basename:
        date = image_basename.split('_')[3] #get sensing date from image name
        
        clippedImgPath = os.path.join(OUT_CLIP_FOLDER, date, 'clipped_' + os.path.basename(inputBand)) # create path to clippe images
        #pprint(clippedImgPath)
        
        process_func.clipimage(AREA_MASK, inputBand, clippedImgPath, True, False) # Clip QA_PIXEl to ROI
        
        #check if clipping worked
        if os.path.isfile(clippedImgPath):
            pprint(f'Clipping QA_PIXEL for file {os.path.basename(inputBand)}')
            continue
            
        else:
            pprint(f'Clipping QA_PIXEL failed for file {os.path.basename(inputBand)}')
            pprint(f'Stopping the programme -- Clipping QA.')
            sys.exit()
        
    # Filter data bands, only clip the ones later used
    elif 'B2' in image_basename or \
        'B3' in image_basename or 'B4' in image_basename or 'B5' in image_basename or \
        'B6' in image_basename or 'B7' in image_basename or 'B10' in image_basename or \
        'B11' in image_basename: 
        
        date = image_basename.split('_')[3] # '20220612'
        
        clippedImgPath = os.path.join(OUT_CLIP_FOLDER, date, 'clipped_' + os.path.basename(inputBand)) # create path to clipped images                                                   
        #pprint(clippedImgPath)
        process_func.clipimage(AREA_MASK, inputBand, clippedImgPath, True, False) #clip the images according to area
        if os.path.isfile(clippedImgPath):
            pprint(f'Clipping bands for file {os.path.basename(image_basename)}')
            continue
        else:
            pprint(f'Clipping bands failed for file {os.path.basename(image_basename)}')
            pprint(f'Stopping the programme -- clipping Bands.')
            sys.exit()

"""""
GET CSV WITH QA_PIXEL BAND STATISTICS FOR CLOUD INFORMATION
"""""
# Locate QA band
path_to_QA_img = process_func.get_qa_filepath(OUT_CLIP_FOLDER, '.TIF')
existing_dates = set()

if os.path.exists('clouds_stats.csv'):
    existing_data = pd.read_csv('clouds_stats.csv')
    existing_dates.update(existing_data['sensing_date'].unique())


# For each QA_band
for file in path_to_QA_img:   
    date_str = (file.split('_')[5])  # Extract date from filename

    if date_str in existing_dates:
        pprint(f'Date {date_str} already processed, skipping.')
        continue
    
    dataset = gdal.Open(file)
    band = dataset.GetRasterBand(1) # open raster bands
    get_data = band.ReadAsArray()   # read as array
    data = band.ReadAsArray().flatten()
    gt = dataset.GetGeoTransform()

    reference_pixel_area = gt[1] * -gt[5]   # area of one pixel
    width = dataset.RasterXSize             # get width in pixels
    height = dataset.RasterYSize            # get heigth in pixels

    pixel_frequency = pd.Series(data).value_counts(dropna = True) # calculate number of occurence of pixel value; ignore NA
   
    
    if pixel_frequency is not None:
        # Calculating areas
        pixel_value_area_m2 = process_func.calc_area_qa_pixels_m2(reference_pixel_area, pixel_frequency)  # calculating m2 area for pixel value
        pixel_value_area_percent = process_func.calc_area_qa_pixels_percent(width, height, pixel_frequency)  # calculating % area for pixel value

        pixel_frequency.name = date_str  # Set DataFrame name to date
        unique_pixel_values = pixel_frequency.index.to_numpy()  # Get unique values from QA_BAND
        
        # Convert unique_pixel_values to binary strings
        binary_pixel_values = [bin(pixel)[2:].zfill(16) for pixel in unique_pixel_values]  # Convert to binary and pad to 16 bits
        
        # Assuming you have the following variables defined:
        # date_str, unique_pixel_values, binary_pixel_values 

        qa_df = pd.DataFrame({
            'sensing_date': date_str,
            'pixel_value': unique_pixel_values, 
            'pixel_binary': binary_pixel_values,
            'pixel_frequency': pixel_frequency.values,
            'pixel_area_m2': pixel_value_area_m2,
            'pixel_area_%': pixel_value_area_percent,  
            #'cloud_pres': preprocess_func.get_bit_index(binary_pixel_values, -5),  # Extract single bit
            'cloud_conf': process_func.get_combined_index(binary_pixel_values, -7, -6),  # Extract and combine two bits
            'shadow_conf': process_func.get_combined_index(binary_pixel_values, -9, -8),  # Extract and combine two bits    
                  
        })

        # Filter from qa_df only pixels that have cloud present in moderate or high confidence, according to:
        # Department of the Interior U.S. Geological Survey. Landsat 8 (L8) Data Users Handbook U.S. Geological Survey (2016). 

        clouds_df = qa_df[
            (((qa_df['cloud_conf'] == '11') | (qa_df['cloud_conf'] == '10')) & 
            ((qa_df['shadow_conf'] == '11') | (qa_df['shadow_conf'] == '10')) & 
            (qa_df['pixel_area_%'] >= 10))
        ]                           
    
        # if there are cloudy pixels, export information to csv
        output_csv_name = 'clouds_stats'
        if not clouds_df.empty:
            csv_df = process_func.export_df_cloud_csv(output_csv_name, clouds_df)
            existing_dates.add(date_str)

        if os.path.exists(output_csv_name + '.csv'):
            pprint(f'Exporting CSV for file {os.path.basename(image_basename)}')
            continue
        else:
            pprint(f'Exporting file to CSV failed')
            pprint(f'Stopping the programme -- Export to CSV.')
            sys.exit()


    
end = time.time()
pprint('Mischief managed...')
print("The time of execution of above program is :",
      (end-start_time))
