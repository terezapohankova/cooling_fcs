import os
from pprint import pprint
import numpy as np
import process_func
import time
import sys
import const_param_0

start_time = time.time()

INPUT_FOLDER = r'/home/tereza/Documents/data/LANDSAT/RESULTS'

INPUT_DATA = os.path.join(INPUT_FOLDER, 'clipped_bands')
JSON_MTL_PATH = process_func.getfilepath(INPUT_DATA, 'MTL.json') #['root/snimky_L9_testovaci/LC09_L2SP_190025_20220518_20220520_02_T1/LC09_L2SP_190025_20220518_20220520_02_T1_MTL.json']

AUX_DATA = r'aux_data'
METEOROLOGY = os.path.join(AUX_DATA, 'weather_2024_olomouc.csv') 
HILLSHADE = os.path.join(AUX_DATA, 'hillshade_olomouc_32633.tif')


FOLDERS = ['lst', 'vegIndices', 'radiation', 'preprocess', 'albedo', 'fluxes', 'et', 'bowen']
for folder in FOLDERS:
    os.makedirs(os.path.join(INPUT_FOLDER, folder), exist_ok=True)


mtlJSONFile = {}
sensing_date_list = []
clipped_img_path = []


esun_sum = sum(const_param_0.esun.values())


meteorologyDict = process_func.createmeteodict(METEOROLOGY) #{{'20220518': {'avg_temp': '14.25','max_temp': '19.8','min_temp': '8.7','relHum': '65.52','wind_sp': '1.25'}},
#pprint(meteorologyDict)
# e.g. meteorologyDict[date]['avg_temp']


ORIGINAL_IMG = process_func.getfilepath(INPUT_DATA, '.TIF') #['root/snimky_L9_testovaci/18052022/LC09_L2SP_190025_20220518_20220518_02_T1_SZA.TIF']

# load JSON MTL file with metadata into dictionary {sensingdate : {metadatafile}} for level 2 (level 2 MTL json includes level 1 MTL data)

for img in ORIGINAL_IMG:
    if img.endswith('.TIF'):
        sensing_date = img.split('_')[5]
        if sensing_date not in sensing_date_list:
            sensing_date_list.append(sensing_date)


for jsonFile in JSON_MTL_PATH:
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
            
            clipped_img_path.append(os.path.join(INPUT_FOLDER, date, os.path.basename(inputBand)))
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
        
        clippedImgPath = os.path.join(INPUT_DATA, date, image_basename) 
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

    
#pprint(imgDict)


for date in sensing_date_list:
    pprint('===================================')
    pprint(f'Sensing Date Calculated: {date}')
    pprint('===================================')
    
    #pprint(meteorologyDict)
    reference_img = imgDict[date]['B5_L2']['clipped_path']

    # from kg m-2 to g cm-2
    #water_vap_cm = ((meteorologyDict[date]['total_col_vat_wap_kg']) / 10)
    
    ta_kelvin = meteorologyDict[date]['avg_temp'] + 273.15

    zenithAngle = process_func.zenithAngle(imgDict[date]['B2_L2']['sunElev'])
    inverseSE = process_func.dr(imgDict[date]['B2_L2']['dES'])
    

       # Atmospheric coefficient for atmospheric correction of thermal band for single channel
    """theta_terms = {
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

    theta_vals = {key: sum(value.values()) for key, value in theta_terms.items()}"""
  
    
    ########## METEOROLOGICAL PARAMETERS ######################
    #pprint(f"Calculating Atmosphere Emissiivty for {date}")
    
    emissivity_atmos = process_func.atmemis(ta_kelvin)
    pprint(f' emissivity_atmos : {emissivity_atmos}')

    transmis_atm = 0.75+2*(10**-5)*219  #process_func.atm_transmiss(theta_vals['theta1'])
    pprint(f' transmis_atm : {transmis_atm} ')

    e0 = process_func.e0(meteorologyDict[date]['avg_temp'])
    pprint(f' e0 : {e0} ')

    es = process_func.es(meteorologyDict[date]['avg_temp'])
    pprint(f' es : {es} ')

    water_vap_cm = (0.0981 * (10 * e0 * meteorologyDict[date]['rel_hum']) + 0.1697)/10
   
    pprint(f' total water vapour : {water_vap_cm} ')

    slope_vap_press = process_func.slopeVapPress(meteorologyDict[date]['avg_temp'])
    pprint(f' slope_vap_press : {slope_vap_press} ')

    p = process_func.atmPress(const_param_0.Z)
    pprint(f' atmPress : {p} ')

    rho = process_func.densityair(meteorologyDict[date]['atm_press'], meteorologyDict[date]['avg_temp'], meteorologyDict[date]['rel_hum'])
    pprint(f' airDensitz : {rho} ')

    psychro = process_func.psychroCons(p)
    pprint(f' psychro : {psychro} ')

    tw_bulb = process_func.Tw_bulb(meteorologyDict[date]['avg_temp'], 
                                      meteorologyDict[date]['rel_hum'])
    pprint(f' T_wet bulb : {tw_bulb} ')

    
    ########## VEGETATION  ######################
    
    pprint(f"Calculating NDVI")

        
    ndvi_path = os.path.join(INPUT_FOLDER, FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'ndvi.TIF')))
    ndvi = process_func.ndvi(imgDict[date]['B5_L2']['clipped_path'],
                            imgDict[date]['B4_L2']['clipped_path'],
                            ndvi_path
                            )


    pprint(f"Calculating ed_fraction_ssebi og Vegetation Cover")
    pv_path = os.path.join(INPUT_FOLDER, FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'pv.TIF')))
    pv = process_func.pv(ndvi, pv_path, reference_img)

    savi_path = os.path.join(INPUT_FOLDER,FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'savi.TIF')))
    savi_index = process_func.savi(imgDict[date]['B4_L2']['clipped_path'], imgDict[date]['B5_L2']['clipped_path'], savi_path, reference_img)
    
    lai_path = os.path.join(INPUT_FOLDER,FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'lai.TIF')))
    lai_index = process_func.lai(ndvi, lai_path, reference_img)

    kc_path = os.path.join(INPUT_FOLDER,FOLDERS[1], os.path.basename(reference_img.replace('B5.TIF', 'kc.TIF')))
    kc = process_func.Kc_LAI(lai_index, kc_path, reference_img)
    
    ########## EMISSIVITY ######################

   

    pprint(f"Calculating Surface Emissivity")
    lse_path = os.path.join(INPUT_FOLDER, FOLDERS[0], os.path.basename(reference_img.replace('B5.TIF', 'lse_b10.TIF')))
    lse_b10 = process_func.emis(const_param_0.emissivity["vegetation"][0], pv, const_param_0.emissivity["bare_soil"][0], lse_path, reference_img)
    
    lse_path_b11 = os.path.join(INPUT_FOLDER, FOLDERS[0], os.path.basename(reference_img.replace('B5.TIF', 'lse_b11.TIF')))
    lse_b11 = process_func.emis(const_param_0.emissivity["vegetation"][1], pv, const_param_0.emissivity["bare_soil"][1], lse_path, reference_img)

    ########## THERMAL CORRECTIONS ######################
    pprint(f"Calculating Sensor Radiance for")
    sens_radiance_path = os.path.join(INPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', 'sens_redaince')))
    sens_radiance = process_func.sensor_radiance(imgDict[date]['B10_L1']['RADIANCE_MULT'], 
                                                  imgDict[date]['B10_L1']['clipped_path'],
                                                  imgDict[date]['B10_L1']['RADIANCE_ADD'],
                                                  sens_radiance_path,
                                                  reference_img)
    
    sens_radiance_path_b11 = os.path.join(INPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', 'sens_redaince_b11')))
    sens_radiance_b11 = process_func.sensor_radiance(imgDict[date]['B11_L1']['RADIANCE_MULT'], 
                                                  imgDict[date]['B11_L1']['clipped_path'],
                                                  imgDict[date]['B11_L1']['RADIANCE_ADD'],
                                                  sens_radiance_path_b11,
                                                  reference_img)
    

    
    
    pprint(f"Calculating Brightness Temperature")
    tbright_path = os.path.join(INPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', 'tbright.TIF')))
    tbright = process_func.bt(imgDict[date]['B10_L1']['KELVIN_CONS_1'],
                            imgDict[date]['B10_L1']['KELVIN_CONS_2'],
                            sens_radiance,
                            tbright_path,
                            imgDict[date]['B10_L1']['clipped_path'],
                            reference_img)
    
    tbright_path_b11 = os.path.join(INPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', 'tbright_b11.TIF')))
    tbright_b11 = process_func.bt(imgDict[date]['B11_L1']['KELVIN_CONS_1'],
                            imgDict[date]['B11_L1']['KELVIN_CONS_2'],
                            sens_radiance,
                            tbright_path_b11,
                            imgDict[date]['B11_L1']['clipped_path'],
                            reference_img)
    
    
    
    #gamma_cal = process_func.gamma(tbright_b11, B_GAMMA, sens_radiance)
    #delta_cal = process_func.delta(tbright, B_GAMMA, gamma_cal, sens_radiance)

    
     ########## LAND SURFACE TEMPERATURE ######################


    pprint(f"Calculating Split Window Surface Temeperature")

    lse_avg = 0.5 * (lse_b10 + lse_b11)
    lse_diff = lse_b10 - lse_b11

    tbright_diff = tbright - tbright_b11


    lst_sw_path = os.path.join(INPUT_FOLDER, FOLDERS[0], os.path.basename(reference_img.replace('B5.TIF', 'lst_sw.TIF')))
    
    lst_sw = process_func.LST_sw(tbright, tbright_diff, lse_avg, lse_diff, water_vap_cm, const_param_0.c_coeffs["c0"], const_param_0.c_coeffs["c1"], const_param_0.c_coeffs["c2"],
                                 const_param_0.c_coeffs["c3"], const_param_0.c_coeffs["c4"], const_param_0.c_coeffs["c5"], const_param_0.c_coeffs["c6"],
                                 lst_sw_path, reference_img)

    #single channel
    #lst = gamma_cal * (1 / lse_b10 * ((theta_vals['theta1'] * sens_radiance) + theta_vals['theta2']) + theta_vals['theta3'] ) + delta_cal
    #lst_C = lst - 273.15
    #process_func.savetif(lst, lst_path, reference_img)

     ########## ALBEDO ######################

    pprint(f"Calculating Albedo")
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    esun_values = [const_param_0.esun[band] for band in bands]
    

    lb_values = {}

    for band in bands:
        band_path = os.path.join(INPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', f'L_{band}.TIF')))
        lb_values[band] = process_func.lb_band(
            imgDict[date][f'{band}_L1']['RADIANCE_ADD'],
            imgDict[date][f'{band}_L1']['RADIANCE_MULT'],
            imgDict[date][f'{band}_L1']['clipped_path'],
            band_path
        )
        
    # Reflectivity calculations for all bands
    reflectivity = {}
    for i, band in enumerate(bands):
        reflectivity[band] = process_func.reflectivity_band(
            lb_values[band], esun_values[i], inverseSE,
            imgDict[date]['B2_L1']['clipped_path'], zenithAngle,
            os.path.join(INPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', f'R_{band}.TIF')))
        )
        
  
    pb = {band: process_func.pb(const_param_0.esun[band], esun_sum) for band in bands}
    albedo_toa = sum(process_func.albedo_toa_band(pb[band], reflectivity[band]) for band in bands)
    process_func.savetif(albedo_toa, os.path.join(INPUT_FOLDER, FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', 'albedo_toa.TIF'))),
                          reference_img)
    albedo_path = os.path.join(INPUT_FOLDER, FOLDERS[4], os.path.basename(reference_img.replace('B5.TIF', 'albedo.TIF')))
    albd = process_func.albedo(albedo_toa, transmis_atm, albedo_path, reference_img)

    
    ########## SOLAR NET RADIATION  ######################
    
    # Calculate Longwave Radiation Outwards
    
    rad_long_out_path = os.path.join(INPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'RadLongOut.TIF')))
    rad_long_out = process_func.longout(lse_b10, lst_sw, reference_img, rad_long_out_path)

    # Calculate Longwave Radiation Inwards
    rad_long_in_path = os.path.join(INPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'RadLongIn.TIF')))
    rad_long_in = process_func.longin(emissivity_atmos, lst_sw, reference_img, rad_long_in_path)

    # Calculate Shortwave Radiation Inwards
    rad_short_in = process_func.shortin(const_param_0.SOLAR_CONSTANT, zenithAngle, inverseSE, transmis_atm)
    #pprint(rad_short_in)
    # Calculate Shortwave Radiation Outwards
    rad_short_out_path = os.path.join(INPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'RadShortOut.TIF')))
    rad_short_out = process_func.shortout(albd, rad_short_in, reference_img, rad_short_out_path)

    # Calculate Net Radiation
    net_radiation_path = os.path.join(INPUT_FOLDER,FOLDERS[2], os.path.basename(reference_img.replace('B5.TIF', 'netRadiation.TIF')))
    net_radiation = process_func.netradiation(rad_short_in, rad_short_out, rad_long_in, rad_long_out, reference_img, net_radiation_path\
                                               , albd, lse_b10)
    ################################
    ### S-SEBI  ##
    ################################

    ### Paths
    g_path_ssebi = os.path.join(INPUT_FOLDER,FOLDERS[5], os.path.basename(reference_img.replace('B5.TIF', 'g_ssebi.TIF')))
    ef_ssebi_path = os.path.join(INPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'ef_ssebi.TIF')))
    le_ssebi_path = os.path.join(INPUT_FOLDER,FOLDERS[5], os.path.basename(reference_img.replace('B5.TIF', 'le_ssebi.TIF')))
    h_ssebi_path = os.path.join(INPUT_FOLDER,FOLDERS[5], os.path.basename(reference_img.replace('B5.TIF', 'h_ssebi.TIF')))
    bowen_path_ssebi = os.path.join(INPUT_FOLDER,FOLDERS[7], os.path.basename(reference_img.replace('B5.TIF', 'bowen_ssebi.TIF')))
    eta_ssebi_path = os.path.join(INPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'eta_ssebi.TIF')))
    
    g_flux_ssebi = process_func.soilGFlux_ssebi(lst_C, albd, ndvi, net_radiation, g_path_ssebi, reference_img) #soil heat flux
    ef_ssebi = process_func.ef_ssebi(albd, lst_sw, ef_ssebi_path, band_path) #evaporative fraction
    le_flux_ssebi = process_func.le_ssebi(ef_ssebi, net_radiation, g_flux_ssebi, le_ssebi_path, reference_img)    # latent heat flux
    h_flux_ssebi = process_func.h_ssebi(ef_ssebi, net_radiation, g_flux_ssebi, h_ssebi_path, reference_img) #sensible heat flux
    bowen_ssebi = process_func.bowenIndex(h_flux_ssebi, le_flux_ssebi, bowen_path_ssebi, band_path) #bowen ratio


    net_radiation_MJ = process_func.W_to_MJday(net_radiation) #convert W/m2 to MJ/s/day
    g_flux_MJ = process_func.W_to_MJday(g_flux_ssebi)  #convert W/m2 to MJ/s/day
    le_flux_MJ = process_func.W_to_MJday(le_flux_ssebi)

    eta_ssebi = process_func.et_a_day_ssebi(le_flux_MJ, net_radiation_MJ,
                                             const_param_0.lheat_vapor * (10**6), eta_ssebi_path, reference_img)
    
    
    ################################
    ### PRIESTLEY-TAYLOR  ##
    ################################
    et_pt_path = os.path.join(INPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'et_pt.TIF')))
    eta_pt_day = 0.68 * ((slope_vap_press / (slope_vap_press + psychro)) * (net_radiation_MJ / const_param_0.lheat_vapor) - (g_flux_MJ / const_param_0.lheat_vapor))

    
    process_func.savetif(eta_pt_day * kc, et_pt_path, reference_img)




    ################################
    ### PENMAN-MONETITH  ##
    ################################

    et0_PM_path = os.path.join(INPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'et0_day_PM.TIF')))
    PM_ea_path = os.path.join(INPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'eta_day_PM.TIF')))

    ET0_pm = process_func.ET0(e0,slope_vap_press, meteorologyDict[date]['wind_sp'], 
                                   es, g_flux_MJ, psychro, net_radiation_MJ, 
                                   meteorologyDict[date]['avg_temp'], et0_PM_path, reference_img)

    PM_ea_day = process_func.ea_pm(ET0_pm, kc, PM_ea_path, reference_img)


    ################################
    ### SSEBOP  ##
    ################################

    etf_ssebop_path = os.path.join(INPUT_FOLDER,FOLDERS[7], os.path.basename(reference_img.replace('B5.TIF', 'etf_ssebop.TIF')))
    eta_ssebop_path = os.path.join(INPUT_FOLDER,FOLDERS[7], os.path.basename(reference_img.replace('B5.TIF', 'eta_ssebop.TIF')))


    le_ssebop_path = os.path.join(INPUT_FOLDER,FOLDERS[7], os.path.basename(reference_img.replace('B5.TIF', 'le_ssebop.TIF')))
    h_ssebop_path = os.path.join(INPUT_FOLDER,FOLDERS[7], os.path.basename(reference_img.replace('B5.TIF', 'h_ssebop.TIF')))

    
    cold_pix, hot_pix = process_func.identify_cold_hot_pixels(lst_C, ndvi, albd)
    pprint(cold_pix)
    pprint(hot_pix)

    etf_ssebop = process_func.etf_ssebop(lst_C, hot_pix, cold_pix, etf_ssebop_path, reference_img)
    le_ssebop = process_func.le_ssebop(etf_ssebop, net_radiation, g_flux_ssebi, le_ssebop_path, reference_img)
    h_ssebop = process_func.h_ssebop(etf_ssebop, net_radiation, g_flux_ssebi, h_ssebop_path, reference_img)
    eta_ssebop = process_func.et_a_day_sssebop(process_func.W_to_MJday(le_ssebop), net_radiation_MJ, const_param_0.lheat_vapor * (10**6), eta_ssebop_path, reference_img)
    ################################
    ### CCI  ##
    ################################

    eti_path = os.path.join(INPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'eti.TIF')))
    eti = process_func.ETI(kc, ET0_pm, eti_path, reference_img)

    cci_path = os.path.join(INPUT_FOLDER,FOLDERS[6], os.path.basename(reference_img.replace('B5.TIF', 'cci.TIF')))
    cci = process_func.CCi(albd, eti, HILLSHADE, cci_path, reference_img)
    
end = time.time()
print("The time of execution of above program is :", (end-start_time))    
