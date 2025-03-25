import os
from pprint import pprint
import numpy as np
import process_func
import time
import sys
import const_param_0
import paths_0
 
pprint('I solemnly swear that I am up to no good...')
start_time = time.time()

## Paths are loaded from paths.py
## Constants and paramaters are loaded from const_param.py

for folder in paths_0.FOLDERS.values():  
    os.makedirs(os.path.join(paths_0.INPUT_FOLDER, folder), exist_ok=True)  



mtlJSONFile = {}
sensing_date_list = []
clipped_img_path = []


esun_sum = sum(const_param_0.esun.values())


meteorologyDict = process_func.createmeteodict(paths_0.METEOROLOGY) #{{'20220518': {'avg_temp': '14.25','max_temp': '19.8','min_temp': '8.7','relHum': '65.52','wind_sp': '1.25'}},
#pprint(meteorologyDict)
# e.g. meteorologyDict[date]['avg_temp']


ORIGINAL_IMG = process_func.getfilepath(paths_0.INPUT_DATA, '.TIF') #['root/snimky_L9_testovaci/18052022/LC09_L2SP_190025_20220518_20220518_02_T1_SZA.TIF']

# load JSON MTL file with metadata into dictionary {sensingdate : {metadatafile}} for level 2 (level 2 MTL json includes level 1 MTL data)

for img in ORIGINAL_IMG:
    if img.endswith('.TIF'):
        sensing_date = img.split('_')[5]
        if sensing_date not in sensing_date_list:
            sensing_date_list.append(sensing_date)


for jsonFile in paths_0.JSON_MTL_PATH:
    if 'L2SP' in jsonFile:
        loadJSON = process_func.loadjson(jsonFile)
        sensDate = jsonFile.split('_')[4]
        #pprint(sensDate)
        if sensDate not in sensing_date_list:
            sensing_date_list.append(sensDate)
        mtlJSONFile[sensDate] = loadJSON

    
# create output path for clipped images by pairing sensing date from JSON metadata file and sensing date on original images
for inputBand in ORIGINAL_IMG:
    
    for date in sensing_date_list:
        
        if os.path.basename(inputBand).split('_')[4] == date: # if date on original input band equals date sensing date from json mtl, then append it to the list
            
            clipped_img_path.append(os.path.join(paths_0.INPUT_FOLDER, date, os.path.basename(inputBand)))
        #pprint(os.path.basename(inputBand).split('_')[4])
        
imgDict = {} # {sensingdate : {path : path, radiance : int ...} }


for inputBand in ORIGINAL_IMG:
    # if date on original input band equals date sensing date from json mtl, then append it to the list
    
    image_basename = os.path.basename(inputBand) # 'LC09_L1TP_190025_20220518_20220518_02_T1_B6_clipped.TIF'
    #pprint(image_basename)

    if 'B' in image_basename:
        image_name = image_basename.replace('.TIF','') #'LC09_L2SP_189026_20220612_20220614_02_T1_SR_B1'
        date = image_basename.split('_')[4] # '20220612'
        
        
        #.split('_')[-1] - last splitted value which should be B1 - B10 
        band = image_basename.replace('.TIF','').split('_')[-1] # 'B1
        
        
        # from basename by splitting keep L1TP, by [:2] keep just L1
        image_level = image_basename.split('_')[2][:2] # 'L2'
        #pprint(image_level)
        
        clippedImgPath = os.path.join(paths_0.INPUT_DATA, date, image_basename) 
        #pprint(clippedImgPath)
     
        band_level_key = f'{band}_{image_level}' # 'B4_L2'
        #pprint(band_level_key)
        if date not in imgDict:    
            imgDict.setdefault(date, {})
        

        imgDict[date][band_level_key] = {
            'clipped_path' : clippedImgPath,
            'imageName' : image_name,
            'RADIANCE_ADD' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'].get(f'RADIANCE_ADD_BAND_{band[1:]}')), #[1:] - delete B
            'RADIANCE_MULT' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'].get(f'RADIANCE_MULT_BAND_{band[1:]}') ),
            'KELVIN_CONS_1' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS'].get(f'K1_CONSTANT_BAND_{band[1:]}') or 0),
            'KELVIN_CONS_2' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS'].get(f'K2_CONSTANT_BAND_{band[1:]}') or 0),
            'REFLECTANCE_ADD': float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['LEVEL2_SURFACE_REFLECTANCE_PARAMETERS'].get(f'REFLECTANCE_ADD_BAND_{band[1:]}') or 0),
            'REFLECTANCE_MULT' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['LEVEL2_SURFACE_REFLECTANCE_PARAMETERS'].get(f'REFLECTANCE_MULT_BAND_{band[1:]}') or 0),
            'dES' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES'].get('EARTH_SUN_DISTANCE')),
            'sunAzimuth' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES'].get('SUN_AZIMUTH')),
            'sunElev' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES'].get('SUN_ELEVATION')),
            }

pprint("Dictionary for imagery metadata has been created.")
pprint('Mischief managed...')