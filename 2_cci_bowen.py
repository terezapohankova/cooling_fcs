
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
METEOROLOGY = r'aux_data/weather_2023_olomouc.csv'   
HILLSHADE = r'aux_data/hillshade_olomouc_32633.tif'
JSON_MTL_PATH = supportlib_v2.getfilepath(INPUT_FOLDER, 'MTL.json') #['root/snimky_L9_testovaci/LC09_L2SP_190025_20220518_20220520_02_T1/LC09_L2SP_190025_20220518_20220520_02_T1_MTL.json']


OUTPUT_FOLDER = r'/home/tereza/Documents/data/LANDSAT/RESULTS'

FOLDERS = ['lst', 'vegIndices', 'radiation', 'preprocess', 'albedo', 'fluxes', 'et']
for folder in FOLDERS:
    os.makedirs(os.path.join(OUTPUT_FOLDER, folder), exist_ok=True)

Z = 219
MEASURING_HEIGHT = 2
BLENDING_HEIGHT = 200   
AVG_VEG_METEOSTATION = 4 #average vegetation height around the meteostation


mtlJSONFile = {}
sensing_date_list = []
CLIPPED_IMG_PATHS = []
B_GAMMA = 1324 # [K]
SOLAR_CONSTANT = 1367 #[W/m2]

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


####ESUN
# 10.3390/rs12030498
esun = {'B2' : 2067,
        'B3' : 1893,
        'B4' : 1603,
        'B5' : 972.6,
        'B6' : 245,
        'B7' : 79.72}

esun_sum = sum(esun.values())



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


"""for jsonFile in JSON_MTL_PATH:
    for jsonFile in JSON_MTL_PATH:
        if 'L2SP' in jsonFile:
            loadJSON = supportlib_v2.load_json(jsonFile)
            sensDate = jsonFile.split('_')[4] # 20220518
            #pprint(sensDate)
            if sensDate not in sensing_date_list:
                sensing_date_list.append(sensDate)
        mtlJSONFile[sensDate] = loadJSON"""

for jsonFile in JSON_MTL_PATH:
    if 'L2SP' in jsonFile:
        loadJSON = supportlib_v2.load_json(jsonFile)
        sensDate = jsonFile.split('_')[4]
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


for date in sensing_date_list:
    pprint('============')
    pprint(date)
    reference_img = imgDict[date]['B5_L2']['clipped_path']
    

    
    # from kg m-2 to g cm-2
    water_vap_cm = (float(meteorologyDict[date]['total_col_vat_wap_kg']) / 10)
    ta_kelvin = meteorologyDict[date]['avg_temp'] + 273.15
    zenithAngle = supportlib_v2.zenithAngle(imgDict[date]['B2_L2']['sunElev'])
    inverseSE = supportlib_v2.dr(imgDict[date]['B2_L2']['dES'])
    

       # Atmospheric coefficient for atmospheric correction of thermal band
    theta_terms = {
        "theta1": {
            "a": theta1_dict["a"],
            "b": theta1_dict["b"] * (ta_kelvin ** 2) * (water_vap_cm ** 2),
            "c": theta1_dict["c"] * ta_kelvin * (water_vap_cm ** 2),
            "d": theta1_dict["d"] * ta_kelvin * water_vap_cm,
            "e": theta1_dict["e"] * (ta_kelvin ** 2) * water_vap_cm,
            "f": theta1_dict["f"] * ta_kelvin,
            "g": theta1_dict["g"] * water_vap_cm,
            "h": theta1_dict["h"] * (ta_kelvin ** 2),
            "i": theta1_dict["i"] * (water_vap_cm ** 2)
        },
        "theta2": {
            "a": theta2_dict["a"],
            "b": theta2_dict["b"] * (ta_kelvin ** 2) * (water_vap_cm ** 2),
            "c": theta2_dict["c"] * ta_kelvin * (water_vap_cm ** 2),
            "d": theta2_dict["d"] * ta_kelvin * water_vap_cm,
            "e": theta2_dict["e"] * (ta_kelvin ** 2) * water_vap_cm,
            "f": theta2_dict["f"] * ta_kelvin,
            "g": theta2_dict["g"] * water_vap_cm,
            "h": theta2_dict["h"] * (ta_kelvin ** 2),
            "i": theta2_dict["i"] * (water_vap_cm ** 2),
        },
        "theta3": {
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
    }

    theta_vals = {key: sum(value.values()) for key, value in theta_terms.items()}
  
    
    ########## METEOROLOGICAL PARAMETERS ######################
    #pprint(f"Calculating Atmosphere Emissiivty for {date}")
    
    emissivity_atmos = supportlib_v2.atmemis(ta_kelvin)
    pprint(f' emissivity_atmos : {emissivity_atmos} pro {date}')

    transmis_atm = 0.75+2*(10**-5)*219#supportlib_v2.atm_transmiss(theta_vals['theta1'])
    pprint(f' transmis_atm : {transmis_atm} pro {date}')

    e0 = supportlib_v2.e0(meteorologyDict[date]['avg_temp'])
    pprint(f' e0 : {e0} pro {date}')

    es = supportlib_v2.es(meteorologyDict[date]['max_temp'], meteorologyDict[date]['min_temp'], meteorologyDict[date]['avg_temp'])

    dealta_es = supportlib_v2.slopeVapPress(meteorologyDict[date]['avg_temp'])
    pprint(f' dealta_es : {dealta_es} pro {date}')

    p = supportlib_v2.atmPress(Z)
    pprint(f' atmPress : {p} + {date}')

    psychro = supportlib_v2.psychroCons(p)
    pprint(f' psychro : {psychro} + {date}')

    mom_rough_len_path = os.path.join(OUTPUT_FOLDER, FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'z0m.TIF')))
    mom_rough_len = supportlib_v2.z0m(AVG_VEG_METEOSTATION, mom_rough_len_path)
    

    

    ########## SATELLITE PARAMETERS ######################
    pprint(f"Calculating Brightness Temperature for {date}")
    tbright_path = os.path.join(OUTPUT_FOLDER, FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'tbright.TIF')))
    tbright = supportlib_v2.bt(imgDict[date]['B10_L1']['KELVIN_CONS_1'],
                            imgDict[date]['B10_L1']['KELVIN_CONS_2'],
                            imgDict[date]['B10_L1']['RADIANCE_ADD'],
                            imgDict[date]['B10_L1']['RADIANCE_MULT'],
                            tbright_path,
                            imgDict[date]['B10_L1']['clipped_path'],
                            reference_img)
    
    
    
    pprint(f"Calculating Sensor Radiance for {date}")
    sens_radiance_path = os.path.join(OUTPUT_FOLDER, FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'sens_redaince')))
    sens_radiance = supportlib_v2.sensor_radiance(tbright, 
                                                  imgDict[date]['B10_L1']['KELVIN_CONS_1'], 
                                                  imgDict[date]['B10_L1']['KELVIN_CONS_2'],
                                                  sens_radiance_path,
                                                  reference_img)
    

    gamma_cal = supportlib_v2.gamma(tbright, B_GAMMA, sens_radiance)
    delta_cal = supportlib_v2.delta(tbright, B_GAMMA, gamma_cal, sens_radiance)

    pprint(f"Calculating NDVI for {date}")

    
    ndvi_path = os.path.join(OUTPUT_FOLDER, FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'ndvi.TIF')))
    ndvi = supportlib_v2.ndvi(imgDict[date]['B5_L2']['clipped_path'],
                              imgDict[date]['B4_L2']['clipped_path'],
                              ndvi_path
                              )
   
    pprint(f"Calculating Fraction og Vegetation Cover for {date}")
    pv_path = os.path.join(OUTPUT_FOLDER, FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'pv.TIF')))
    pv = supportlib_v2.pv(ndvi, pv_path, reference_img)

    pprint(f"Calculating Surface Emissivity for {date}")
    lse_path = os.path.join(OUTPUT_FOLDER, FOLDERS[0], os.path.basename(reference_img.replace('B5.TIF', 'lse.TIF')))
    lse = supportlib_v2.emis(ndvi, pv, lse_path, reference_img)


    pprint(f"Calculating Surface Temeperature for {date}")
    lst_path = os.path.join(OUTPUT_FOLDER, FOLDERS[0], os.path.basename(reference_img.replace('B5.TIF', 'lst.TIF')))
    lst = gamma_cal * (1 / lse * ((theta_vals['theta1'] * sens_radiance) + theta_vals['theta2']) + theta_vals['theta3'] ) + delta_cal
    lst_C = lst - 273.15
    supportlib_v2.savetif(lst, lst_path, reference_img)
    

    pprint(f"Calculating Albedo for {date}")
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    esun_values = [esun[band] for band in bands]

    lb_values = {}

    for band in bands:
        band_path = os.path.join(OUTPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', f'L_{band}.TIF')))
        lb_values[band] = supportlib_v2.lb_band(
            imgDict[date][f'{band}_L1']['RADIANCE_ADD'],
            imgDict[date][f'{band}_L1']['RADIANCE_MULT'],
            imgDict[date][f'{band}_L1']['clipped_path'],
            band_path
        )
        
    # Reflectivity calculations for all bands
    reflectivity = {}
    for i, band in enumerate(bands):
        reflectivity[band] = supportlib_v2.reflectivity_band(
            lb_values[band], esun_values[i], inverseSE,
            imgDict[date]['B2_L1']['clipped_path'], zenithAngle,
            os.path.join(OUTPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', f'R_{band}.TIF')))
        )
        
  
    pb = {band: supportlib_v2.pb(esun[band], esun_sum) for band in bands}
    albedo_toa = sum(supportlib_v2.albedo_toa_band(pb[band], reflectivity[band]) for band in bands)

    albedo_path = os.path.join(OUTPUT_FOLDER, FOLDERS[4], os.path.basename(reference_img.replace('B5.TIF', 'albedo.TIF')))
    albd = supportlib_v2.albedo(albedo_toa, transmis_atm, albedo_path, reference_img)

    # Calculate Longwave Radiation Outwards
    
    rad_long_out_path = os.path.join(OUTPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'RadLongOut.TIF')))
    rad_long_out = supportlib_v2.longout(lse, lst, reference_img, rad_long_out_path)

    # Calculate Longwave Radiation Inwards
    rad_long_in_path = os.path.join(OUTPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'RadLongIn.TIF')))
    rad_long_in = supportlib_v2.longin(emissivity_atmos, lst, reference_img, rad_long_in_path)

    # Calculate Shortwave Radiation Inwards
    rad_short_in = supportlib_v2.shortin(SOLAR_CONSTANT, zenithAngle, inverseSE, transmis_atm)
    pprint(rad_short_in)
    # Calculate Shortwave Radiation Outwards
    rad_short_out_path = os.path.join(OUTPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'RadShortOut.TIF')))
    rad_short_out = supportlib_v2.shortout(albd, rad_short_in, reference_img, rad_short_out_path)

    # Calculate Net Radiation
    net_radiation_path = os.path.join(OUTPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'netRadiation.TIF')))
    net_radiation = supportlib_v2.netradiation(rad_short_in, rad_short_out, rad_long_in, rad_long_out, reference_img, net_radiation_path\
                                               , albd, lse)
    
    net_radiation_path_MJ = os.path.join(OUTPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'netRadiation_MJ.TIF')))
    #net_radiation_MJ = (((net_radiation / 10 ** 6)) * 3600 * 24) 
    net_radiation_MJ = (net_radiation * 86400) / 10000000 #daily MJ/m2/den
    supportlib_v2.savetif(net_radiation_MJ, net_radiation_path_MJ, reference_img)

    G_MJ_path= os.path.join(OUTPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'G_MJ.TIF')))
    g_path = os.path.join(OUTPUT_FOLDER,FOLDERS[5], os.path.basename(reference_img.replace('B5.TIF', 'G.TIF')))
    G_flux = supportlib_v2.soilGFlux(lst_C, albd, ndvi, net_radiation, g_path, reference_img)
    G_flux_MJ = (G_flux * 86400) / 10000000
    supportlib_v2.savetif(G_flux_MJ, G_MJ_path, reference_img)

    fric_vel_2m = supportlib_v2.u_fric_vel_measure(meteorologyDict[date]['wind_sp'], MEASURING_HEIGHT, mom_rough_len) #friction velocity at height emasurement (2 m)
    #pprint(np.nanmean(fric_vel_2m))

    wind_sp_200m = supportlib_v2.wind_speed_blending(BLENDING_HEIGHT, mom_rough_len, fric_vel_2m) # wind speed at blending height )200 m)
    #pprint(np.nanmean(fwind_sp_200m))

    fric_vel_pixel = supportlib_v2.u_fric_vel_pix(BLENDING_HEIGHT, wind_sp_200m, mom_rough_len) # friction velocity for 200 m

    #rah_outputpath = os.path.join(OUTPUT_FOLDER,FOLDERS[5], os.path.basename(imgDict[date]['B5_L2']['clipped_path']).replace('B5.TIF', 'friction_velocity.TIF'))
    r_ah = supportlib_v2.rah(0.1, 2.0, fric_vel_pixel)

    fraction_path = os.path.join(OUTPUT_FOLDER,FOLDERS[5], os.path.basename(reference_img.replace('B5.TIF', 'etf.TIF')))
    fraction = supportlib_v2.ef(lst, ndvi, fraction_path, reference_img)

    pmet0_path = os.path.join(OUTPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'et0_day_mm.TIF')))
    pm_et0 = supportlib_v2.ET0(e0, dealta_es, meteorologyDict[date]['wind_sp'], es, G_flux_MJ, psychro, net_radiation_MJ, meteorologyDict[date]['avg_temp'], pmet0_path, reference_img)

    savi_path = os.path.join(OUTPUT_FOLDER,FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'savi.TIF')))
    savi_index = supportlib_v2.savi(imgDict[date]['B4_L2']['clipped_path'], imgDict[date]['B5_L2']['clipped_path'], savi_path, reference_img)
    
    lai_path = os.path.join(OUTPUT_FOLDER,FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'lai.TIF')))
    lai_index = supportlib_v2.lai(savi_index, lai_path, reference_img)
    
    kc_path = os.path.join(OUTPUT_FOLDER,FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'Kc_lai.TIF')))
    kc_lai = supportlib_v2.Kc_LAI(lai_index, kc_path, reference_img)

    eti_path = os.path.join(OUTPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'eti.TIF')))
    eti = supportlib_v2.ETI(kc_lai, pm_et0, eti_path, reference_img)

    cci_path = os.path.join(OUTPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'cci.TIF')))
    cci = supportlib_v2.CCi(albd, eti, HILLSHADE, cci_path, reference_img)

    ETa_path = os.path.join(OUTPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'eta_ssebop.TIF')))
    ETa = supportlib_v2.ea(pm_et0, fraction, ndvi, ETa_path, reference_img)

end = time.time()
print("The time of execution of above program is :",
      (end-start_time))    
    
    