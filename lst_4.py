from pprint import pprint
import img_prep_2
import process_func
import os
import paths_0
import const_param_0
import meteo_3


pprint('===================================')
pprint("Calculating LST")
pprint('===================================')
for date in img_prep_2.sensing_date_list:
########## VEGETATION  ######################

    pprint('===================================')
    pprint(f"'Sensing Date Calculated: {date}'")
    pprint('===================================')

    #pprint(meteorologyDict)
    reference_img = img_prep_2.imgDict[date]['B5_L2']['clipped_path']
    
    
    ndvi = process_func.ndvi(img_prep_2.imgDict[date]['B5_L2']['clipped_path'],
                            img_prep_2.imgDict[date]['B4_L2']['clipped_path'],
                            process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['vegIndices'], 
                                                            'ndvi.TIF'))
    
    pv = process_func.pv(ndvi, 
                         process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['vegIndices'], 
                                                            'Pv.TIF'),
                                                            reference_img)
    
    
    
    
    lse_b10 = process_func.emis(const_param_0.emissivity["vegetation"][0], 
                                pv, 
                                const_param_0.emissivity["bare_soil"][0], 
                                process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['lse'], 
                                                            'lse_b10.TIF'),
                                                            reference_img)
    
    
    lse_b11 = process_func.emis(const_param_0.emissivity["vegetation"][1], 
                                pv, 
                                const_param_0.emissivity["bare_soil"][1], 
                                process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['lse'], 
                                                            'lse_b11.TIF'),
                                                            reference_img)

    ########## THERMAL CORRECTIONS ######################

    sens_radiance = process_func.sensor_radiance(img_prep_2.imgDict[date]['B10_L1']['RADIANCE_MULT'], 
                                                  img_prep_2.imgDict[date]['B10_L1']['clipped_path'],
                                                  img_prep_2.imgDict[date]['B10_L1']['RADIANCE_ADD'],
                                                  process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['preprocess'], 
                                                            'sens_radiance_b10.TIF'),
                                                            reference_img)
    
    sens_radiance_b11 = process_func.sensor_radiance(img_prep_2.imgDict[date]['B11_L1']['RADIANCE_MULT'], 
                                                  img_prep_2.imgDict[date]['B11_L1']['clipped_path'],
                                                  img_prep_2.imgDict[date]['B11_L1']['RADIANCE_ADD'],
                                                  process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['preprocess'], 
                                                            'sens_radiance_b11.TIF'),
                                                            reference_img)

    
    tbright = process_func.bt(img_prep_2.imgDict[date]['B10_L1']['KELVIN_CONS_1'],
                            img_prep_2.imgDict[date]['B10_L1']['KELVIN_CONS_2'],
                            sens_radiance,
                            img_prep_2.imgDict[date]['B10_L1']['clipped_path'],
                            process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['preprocess'], 
                                                            'tbright_b10.TIF'),
                            reference_img)
    
    
    tbright_b11 = process_func.bt(img_prep_2.imgDict[date]['B11_L1']['KELVIN_CONS_1'],
                            img_prep_2.imgDict[date]['B11_L1']['KELVIN_CONS_2'],
                            sens_radiance,
                            img_prep_2.imgDict[date]['B11_L1']['clipped_path'],
                            process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['preprocess'], 
                                                            'tbright_b11.TIF'),
                            reference_img)
    

    lse_avg = 0.5 * (lse_b10 + lse_b11)
    lse_diff = lse_b10 - lse_b11

    tbright_diff = tbright - tbright_b11


    
    
    lst_sw = process_func.LST_sw(tbright, 
                                 tbright_diff, 
                                 lse_avg, 
                                 lse_diff, 
                                 meteo_3.water_vap_cm, 
                                 const_param_0.c_coeffs["c0"], 
                                 const_param_0.c_coeffs["c1"], 
                                 const_param_0.c_coeffs["c2"],
                                 const_param_0.c_coeffs["c3"], 
                                 const_param_0.c_coeffs["c4"], 
                                 const_param_0.c_coeffs["c5"], 
                                 const_param_0.c_coeffs["c6"],
                                 process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['lst'], 
                                                            'lst_sw.TIF'), 
                                 reference_img)