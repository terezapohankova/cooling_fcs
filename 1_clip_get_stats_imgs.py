import os, sys
import json
from pprint import pprint
import fiona
import rasterio as rio
from osgeo import gdal
import pandas as pd
import shutil


import preprocess_func
import supportlib_v2
import time
start_time = time.time()

OUTPUT_PATH = r'/home/tereza/Documents/data/LANDSAT/RESULTS'

INPUT_FOLDER = r'/media/tereza/e69216f4-59db-4b9c-a9ea-eec828d59ee5/home/tereza/Documents/LANDSAT_2023'
JSON_MTL_PATH = supportlib_v2.getfilepath(INPUT_FOLDER, 'MTL.json') #['root/snimky_L9_testovaci/LC09_L2SP_190025_20220518_20220520_02_T1/LC09_L2SP_190025_20220518_20220520_02_T1_MTL.json']
AREA_MASK = r'/home/tereza/Documents/gh_repos/cooling_fcs/aux_data/olomouc_32633.gpkg'


OUT_CLIP_FOLDER = os.path.join(OUTPUT_PATH,'clipped_bands')
os.makedirs(OUT_CLIP_FOLDER, exist_ok = True)


# get paths to images and jsons
JSON_MTL_PATH = supportlib_v2.get_band_filepath(INPUT_FOLDER, 'MTL.json') #['root/snimky_L9_testovaci/LC09_L2SP_190025_20220518_20220520_02_T1/LC09_L2SP_190025_20220518_20220520_02_T1_MTL.json']
ORIGINAL_IMG = supportlib_v2.get_band_filepath(INPUT_FOLDER, '.TIF') #['root/snimky_L9_testovaci/18052022/LC09_L2SP_190025_20220518_20220518_02_T1_SZA.TIF']

CLOUD_PIXELS = [22280, 24088, 24216, 23344, 24472, 55052]

# empty stuff
mtlJSONFile = {}
sensingDate = []
#nni knstanta
CLIPPED_IMG_PATHS = []

# load JSON MTL file with metadata into dictionary {sensingdate : {metadatafile}} for level 2 (level 2 MTL json includes level 1 MTL data)
for jsonFile in JSON_MTL_PATH:
    if 'L2SP' in jsonFile:
        loadJSON = supportlib_v2.load_json(jsonFile)
        #pprint(jsonFile.split('_'))
        sensDate = jsonFile.split('_')[4] # 20220518
        
        if sensDate not in sensingDate:
            sensingDate.append(sensDate)
            #pprint(sensingDate)
            if not os.path.exists(os.path.join(OUT_CLIP_FOLDER, sensDate)): 
                os.makedirs(os.path.join(OUT_CLIP_FOLDER, sensDate))
        mtlJSONFile[sensDate] = loadJSON   
        shutil.copy(jsonFile, os.path.join(OUT_CLIP_FOLDER, sensDate))


# create output path for clipped images by pairing sensing date from JSON metadata file and sensing date on original images
for inputBand in ORIGINAL_IMG:
    for date in sensingDate:
        
        if os.path.basename(inputBand).split('_')[3] == date: # if date on original input band equals date sensing date from json mtl, then append it to the list
            #pprint(os.path.join(OUTPUT_PATH, OUT_CLIP_FOLDER, date, 'clipped_' + os.path.basename(inputBand)))
            CLIPPED_IMG_PATHS.append(os.path.join(OUT_CLIP_FOLDER, date, 'clipped_' + os.path.basename(inputBand)))
            # pprint(CLIPPED_IMG_PATHS)

for inputBand in ORIGINAL_IMG:
    
    # if date on original input band equals date sensing date from json mtl, then append it to the list
    
    image_basename = os.path.basename(inputBand) # 'LC09_L1TP_190025_20220518_20220518_02_T1_B6_clipped.TIF'
    
    if 'L2SP' and 'QA_PIXEL' in image_basename:
        date = image_basename.split('_')[3]
        clippedImgPath = os.path.join(OUT_CLIP_FOLDER, date, 'clipped_' + (os.path.basename(inputBand)))
        supportlib_v2.clipimage(AREA_MASK, inputBand, clippedImgPath, True, False)
        #pprint(clippedImgPath)
    
        
    elif 'B2' in image_basename or \
        'B3' in image_basename or 'B4' in image_basename or 'B5' in image_basename or \
        'B6' in image_basename or 'B7' in image_basename or 'B10' in image_basename: 
        #image_name = image_basename.replace('.TIF','') #'LC09_L2SP_189026_20220612_20220614_02_T1_SR_B1'
                   
        date = image_basename.split('_')[3] # '20220612'   
        clippedImgPath = os.path.join(OUT_CLIP_FOLDER, date, 'clipped_' + (os.path.basename(inputBand))) # '/home/tereza/Documents/testy_VSC/clipped_bands/20220612/clipped_LC09_L2SP_189026_20220612_20220614_02_T1_SR_B1.TIF'                                                    
        supportlib_v2.clipimage(AREA_MASK, inputBand, clippedImgPath, True, False)

      
    
### create dictionary for QA_PIXEL band that will get the frequency of each pixel value for each sensing date




"""""
GET CSV WITH QA_PIXEL BAND STATISTICS FOR CLOUD MASKING
"""""

"""path_to_QA_img = supportlib_v2.get_qa_filepath(OUT_CLIP_FOLDER, '.TIF')


for file in path_to_QA_img:
   
    date_str = (file.split('_')[5])  # Extract date from filename
    
    dataset = gdal.Open(file)
    band = dataset.GetRasterBand(1)
    get_data = band.ReadAsArray()   # read as array
    data = band.ReadAsArray().flatten()
    gt = dataset.GetGeoTransform()


    reference_pixel_area = gt[1] * -gt[5]   # area of one pixel
    width = dataset.RasterXSize             # get width in pixels
    height = dataset.RasterYSize            # get heigth in pixels

    pixel_frequency = pd.Series(data).value_counts(dropna = True)
   
    if pixel_frequency is not None:
        pixel_value_area_m2 = preprocess_func.calc_area_qa_pixels_m2(reference_pixel_area, pixel_frequency)  # calculating m2 area for pixel value
        pixel_value_area_percent =  preprocess_func.calc_area_qa_pixels_percent(width, height, pixel_frequency) # calculating % area for pixel value
        
        pixel_frequency.name = date_str  # Set DataFrame name to date
        unique_pixel_values = pixel_frequency.index.to_numpy() #get unique values from QA_BAND

        # create data frame with columns
        qa_df = pd.DataFrame({
            'pixel_value': unique_pixel_values,  
            'pixel_frequency': pixel_frequency,
            'sensing_date': date_str,
            'pixel_area_m2': pixel_value_area_m2,
            'pixel_area_%': pixel_value_area_percent,
              })


        cloud_coverage_df = preprocess_func.filter_df_cloud_pixels(qa_df, 30, CLOUD_PIXELS) #filter cloud pixels if cloud coverage is larger than value (default 30 %)
        
        # if the dataframe is not empty, then export it to csv 
        if not cloud_coverage_df.empty:
            preprocess_func.export_df_cloud_csv(date_str, cloud_coverage_df)
                  
        dataset.Close()"""
        
end = time.time()       
print("The time of execution of above program is :",
      (end-start_time))     
