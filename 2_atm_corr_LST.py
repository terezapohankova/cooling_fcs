
import math
import os
import sys
from pprint import pprint
import numpy as np
import supportlib_v2
import tifffile as tf
import time
start_time = time.time()

INPUT_FOLDER = r'/home/tereza/Documents/data/LANDSAT/RESULTS/clipped_bands'
METEOROLOGY = r'/home/tereza/Documents/GitHub/repos/cooling_fcs/aux_data/weather_2023_olomouc.csv'   
JSON_MTL_PATH = supportlib_v2.getfilepath(INPUT_FOLDER, 'MTL.json') #['root/snimky_L9_testovaci/LC09_L2SP_190025_20220518_20220520_02_T1/LC09_L2SP_190025_20220518_20220520_02_T1_MTL.json']


OUTPUT_FOLDER = r'/home/tereza/Documents/data/LANDSAT/RESULTS'
OUT_LST_FOLDER = os.path.join(OUTPUT_FOLDER,'lst')
os.makedirs(OUT_LST_FOLDER, exist_ok = True)

OUT_VEG_INDEX_FOLDER = os.path.join(OUTPUT_FOLDER,'vegIndices')
os.makedirs(OUT_VEG_INDEX_FOLDER, exist_ok = True)




mtlJSONFile = {}
sensing_date_list = []
CLIPPED_IMG_PATHS = []
B_GAMMA = 1324 # [K]

#thetaX_dict["a"]
# https://www.mdpi.com/2072-4292/10/3/431
theta1_dict = {
    "a": 4.4729730361,
    "b": -0.0000748260,
    "c": 0.0466282124,
    "d": 0.0231691781,
    "e": -0.0000496173,
    "f": -0.0262745276,
    "g": -2.4523205637,
    "h": 0.0000492124,
    "i": -7.2121979375
}

theta2_dict = {
  "a": -30.3702785256,
  "b": 0.0009118768,
  "c": -0.5731956714,
  "d": -0.7844419527,
  "e": 0.0014080695,
  "f": 0.2157797227,
  "g": 106.5509303783,
  "h": -0.0003760208,
  "i": 89.6156888857
}

theta3_dict = {
  "a": -3.7618398628,
  "b": -0.0001417749,
  "c": 0.0911362208,
  "d": 0.5453487543,
  "e": -0.0009095018,
  "f": 0.0418090158,
  "g": -79.9583806096,
  "h": -0.0001047275,
  "i": -14.6595491055
}

meteorologyDict = supportlib_v2.createmeteodict(METEOROLOGY) #{{'20220518': {'avg_temp': '14.25','max_temp': '19.8','min_temp': '8.7','relHum': '65.52','wind_sp': '1.25'}},
#pprint(meteorologyDict)
# e.g. meteorologyDict[date]['avg_temp']


ORIGINAL_IMG = supportlib_v2.get_band_filepath(INPUT_FOLDER, '.TIF') #['root/snimky_L9_testovaci/18052022/LC09_L2SP_190025_20220518_20220518_02_T1_SZA.TIF']

# load JSON MTL file with metadata into dictionary {sensingdate : {metadatafile}} for level 2 (level 2 MTL json includes level 1 MTL data)

for img in ORIGINAL_IMG:
    if img.endswith('.TIF'):
        
        sensing_date = img.split('_')[5]
        
        if sensing_date not in sensing_date_list:
            sensing_date_list.append(sensing_date)


for jsonFile in JSON_MTL_PATH:
    for jsonFile in JSON_MTL_PATH:
        if 'L2SP' in jsonFile:
            loadJSON = supportlib_v2.load_json(jsonFile)
            sensDate = jsonFile.split('_')[4] # 20220518
            #pprint(sensDate)
            if sensDate not in sensing_date_list:
                sensing_date_list.append(sensDate)
        mtlJSONFile[sensDate] = loadJSON


# create output path for clipped images by pairing sensing date from JSON metadata file and sensing date on original images
for inputBand in ORIGINAL_IMG:
    
    for date in sensing_date_list:
        
        if os.path.basename(inputBand).split('_')[4] == date: # if date on original input band equals date sensing date from json mtl, then append it to the list
            #pprint(os.path.join(OUTPUT_PATH, OUT_CLIP_FOLDER, date, 'clipped_' + os.path.basename(inputBand)))
            CLIPPED_IMG_PATHS.append(os.path.join(INPUT_FOLDER, date, os.path.basename(inputBand)))
            
        
imgDict = {} # {sensingdate : {path : path, radiance : int ...} }


for inputBand in ORIGINAL_IMG:
    # if date on original input band equals date sensing date from json mtl, then append it to the list
    
    image_basename = os.path.basename(inputBand) # 'LC09_L1TP_190025_20220518_20220518_02_T1_B6_clipped.TIF'
    

    if 'B' in image_basename:
        image_name = image_basename.replace('.TIF','') #'LC09_L2SP_189026_20220612_20220614_02_T1_SR_B1'
        date = image_basename.split('_')[4] # '20220612'
        #pprint(date)
        
        #.split('_')[-1] - last splitted value which should be B1 - B10 
        band = image_basename.replace('.TIF','').split('_')[-1] # 'B1
        #pprint(band)
        
        # from basename by splitting keep L1TP, by [:2] keep just L1
        image_level = image_basename.split('_')[2][:2] # 'L2'
        #pprint(image_level)
        
        clippedImgPath = os.path.join(INPUT_FOLDER, date, image_basename) 
        # '/home/tereza/Documents/testy_VSC/clipped_bands/20220612/clipped_LC09_L2SP_189026_20220612_20220614_02_T1_SR_B1.TIF'
        #pprint(clippedImgPath)
        # band_level_key must be unique
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
            #'dES' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES'].get('EARTH_SUN_DISTANCE')),
            #'sunAzimuth' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES'].get('SUN_AZIMUTH')),
            'sunElev' : float(mtlJSONFile[date]['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES'].get('SUN_ELEVATION')),
            }
       
for date in sensing_date_list:
    pprint(date)
    pprint('============')
    tbright = supportlib_v2.bt(imgDict[date]['B10_L1']['KELVIN_CONS_1'],
                            imgDict[date]['B10_L1']['KELVIN_CONS_2'],
                            imgDict[date]['B10_L1']['RADIANCE_ADD'],
                            imgDict[date]['B10_L1']['RADIANCE_MULT'],
                            os.path.join(OUTPUT_FOLDER, 'lst', 
                            os.path.basename(imgDict[date]['B10_L1']['clipped_path']).replace('B10.TIF', 'BT' + '.TIF')),
                            imgDict[date]['B10_L1']['clipped_path'],
                            imgDict[date]['B5_L2']['clipped_path'])
    
    
    
    
    sens_radiance = supportlib_v2.sensor_radiance(tbright, 
                                                  imgDict[date]['B10_L1']['KELVIN_CONS_1'], 
                                                  imgDict[date]['B10_L1']['KELVIN_CONS_2'],
                                                  os.path.join(OUTPUT_FOLDER, 'lst', 
                                                        os.path.basename(imgDict[date]['B10_L1']['clipped_path']).replace('B10.TIF', 'L_sens' + '.TIF')),
                                                  imgDict[date]['B5_L2']['clipped_path'])
    
    

    gamma_cal = supportlib_v2.gamma(tbright, B_GAMMA, sens_radiance)
    delta_cal = supportlib_v2.delta(tbright, B_GAMMA, gamma_cal, sens_radiance)

    ndvi = supportlib_v2.ndvi(imgDict[date]['B5_L2']['clipped_path'],
                              imgDict[date]['B4_L2']['clipped_path'],
                              os.path.join(OUT_VEG_INDEX_FOLDER, 
                                    os.path.basename(imgDict[date]['B5_L2']['clipped_path']).replace('B5.TIF', 'ndvi' + '.TIF')))
   
    pv = supportlib_v2.pv(ndvi,
                          os.path.join(OUT_VEG_INDEX_FOLDER, 
                                    os.path.basename(imgDict[date]['B5_L2']['clipped_path']).replace('B5.TIF', 'pv' + '.TIF')),
                                    imgDict[date]['B4_L2']['clipped_path'])
    
    lse = supportlib_v2.emis(ndvi, 
                             pv,
                             os.path.join(OUT_LST_FOLDER,
                                    os.path.basename(imgDict[date]['B5_L2']['clipped_path']).replace('B5.TIF', 'lse' + '.TIF')),
                                    imgDict[date]['B5_L2']['clipped_path'])

    # from kg m-2 to g cm-2
    water_vap_cm = (float(meteorologyDict[date]['total_col_vat_wap_kg']) / 10)
    ta_kelvin = meteorologyDict[date]['avg_temp'] + 273.15
    theta1_terms = {
        "a": theta1_dict["a"],
        "b": theta1_dict["b"] * (ta_kelvin ** 2) * (water_vap_cm ** 2),
        "c": theta1_dict["c"] * ta_kelvin * (water_vap_cm ** 2),
        "d": theta1_dict["d"] * ta_kelvin * water_vap_cm,
        "e": theta1_dict["e"] * (ta_kelvin ** 2) * water_vap_cm,
        "f": theta1_dict["f"] * ta_kelvin,
        "g": theta1_dict["g"] * water_vap_cm,
        "h": theta1_dict["h"] * (ta_kelvin ** 2),
        "i": theta1_dict["i"] * (water_vap_cm ** 2)
    }

    theta2_terms = {
        "a": theta2_dict["a"],
        "b": theta2_dict["b"] * (ta_kelvin ** 2) * (water_vap_cm ** 2),
        "c": theta2_dict["c"] * ta_kelvin * (water_vap_cm ** 2),
        "d": theta2_dict["d"] * ta_kelvin * water_vap_cm,
        "e": theta2_dict["e"] * (ta_kelvin ** 2) * water_vap_cm,
        "f": theta2_dict["f"] * ta_kelvin,
        "g": theta2_dict["g"] * water_vap_cm,
        "h": theta2_dict["h"] * (ta_kelvin ** 2),
        "i": theta2_dict["i"] * (water_vap_cm ** 2),
    }

    theta3_terms = {
        "a": theta3_dict["a"],
        "b": theta3_dict["b"] * (ta_kelvin ** 2) * (water_vap_cm ** 2),
        "c": theta3_dict["c"] * ta_kelvin * (water_vap_cm ** 2),
        "d": theta3_dict["d"] * ta_kelvin * water_vap_cm,
        "e": theta3_dict["e"] * (ta_kelvin ** 2) * water_vap_cm,
        "f": theta3_dict["f"] * ta_kelvin,
        "g": theta3_dict["g"] * water_vap_cm,
        "h": theta3_dict["h"] * (ta_kelvin ** 2),
        "i": theta3_dict["i"] * (water_vap_cm ** 2),
    }

    theta1_vals = sum(value for value in theta1_terms.values())
    theta2_vals = sum(value for value in theta2_terms.values())
    theta3_vals = sum(value for value in theta3_terms.values())
    #theta3_vals = -1.5
   

    lst = gamma_cal * (1 / lse * ((theta1_vals * sens_radiance) + theta2_vals) + theta3_vals ) + delta_cal
    lst_C = lst - 273.15
    supportlib_v2.savetif(lst_C, os.path.join(OUT_LST_FOLDER, 
                                    os.path.basename(imgDict[date]['B5_L2']['clipped_path']).replace('B5.TIF', 'lst' + '.TIF')),
                                    imgDict[date]['B5_L2']['clipped_path'])
end = time.time()
print("The time of execution of above program is :",
      (end-start_time))    
    
    