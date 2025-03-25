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
    
    pprint(f"Calculating NDVI")

        
    ndvi_path = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[1], 
                             os.path.basename(reference_img.replace('B5.TIF', 'ndvi.TIF')))
    
    ndvi = process_func.ndvi(img_prep_2.imgDict[date]['B5_L2']['clipped_path'],
                            img_prep_2.imgDict[date]['B4_L2']['clipped_path'],
                            ndvi_path
                            )


    pprint(f"Calculating ed_fraction_ssebi og Vegetation Cover")
    pv_path = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[1], 
                           os.path.basename(reference_img.replace('B5.TIF', 'pv.TIF')))
    
    pv = process_func.pv(ndvi, 
                         pv_path, 
                         reference_img)

    savi_path = os.path.join(paths_0.INPUT_FOLDER,paths_0.FOLDERS[1], 
                             os.path.basename(reference_img.replace('B5.TIF', 'savi.TIF')))
    
    savi_index = process_func.savi(img_prep_2.imgDict[date]['B4_L2']['clipped_path'], 
                                   img_prep_2.imgDict[date]['B5_L2']['clipped_path'], 
                                   savi_path, 
                                   reference_img)
    
    lai_path = os.path.join(paths_0.INPUT_FOLDER,paths_0.FOLDERS[1], 
                            os.path.basename(reference_img.replace('B5.TIF', 'lai.TIF')))
    
    lai_index = process_func.lai(ndvi, 
                                 lai_path, 
                                 reference_img)

    kc_path = os.path.join(paths_0.INPUT_FOLDER,paths_0.FOLDERS[1], 
                           os.path.basename(reference_img.replace('B5.TIF', 'kc.TIF')))
    
    kc = process_func.Kc_LAI(lai_index, 
                             kc_path, 
                             reference_img)
    
  

   

    pprint(f"Calculating Surface Emissivity")
    lse_path = os.path.join(paths_0.INPUT_FOLDER, 
                            paths_0.FOLDERS[0], 
                            os.path.basename(reference_img.replace('B5.TIF', 'lse_b10.TIF')))
    
    lse_b10 = process_func.emis(const_param_0.emissivity["vegetation"][0], 
                                pv, 
                                const_param_0.emissivity["bare_soil"][0], 
                                lse_path, 
                                reference_img)
    
    lse_path_b11 = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[0], 
                                os.path.basename(reference_img.replace('B5.TIF', 'lse_b11.TIF')))
    
    lse_b11 = process_func.emis(const_param_0.emissivity["vegetation"][1], 
                                pv, 
                                const_param_0.emissivity["bare_soil"][1], 
                                lse_path, reference_img)

    ########## THERMAL CORRECTIONS ######################
    pprint(f"Calculating Sensor Radiance for")
    sens_radiance_path = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[3], 
                                            os.path.basename(reference_img.replace('B5.TIF', 'sens_redaince')))

    sens_radiance = process_func.sensor_radiance(img_prep_2.imgDict[date]['B10_L1']['RADIANCE_MULT'], 
                                                  img_prep_2.imgDict[date]['B10_L1']['clipped_path'],
                                                  img_prep_2.imgDict[date]['B10_L1']['RADIANCE_ADD'],
                                                  sens_radiance_path,
                                                  reference_img)
    
    sens_radiance_path_b11 = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[3], 
                                            os.path.basename(reference_img.replace('B5.TIF', 'sens_redaince_b11')))
    
    sens_radiance_b11 = process_func.sensor_radiance(img_prep_2.imgDict[date]['B11_L1']['RADIANCE_MULT'], 
                                                  img_prep_2.imgDict[date]['B11_L1']['clipped_path'],
                                                  img_prep_2.imgDict[date]['B11_L1']['RADIANCE_ADD'],
                                                  sens_radiance_path_b11,
                                                  reference_img)
    

    
    
    pprint(f"Calculating Brightness Temperature")
    tbright_path = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[3], 
                                    os.path.basename(reference_img.replace('B5.TIF', 'tbright.TIF')))
    
    tbright = process_func.bt(img_prep_2.imgDict[date]['B10_L1']['KELVIN_CONS_1'],
                            img_prep_2.imgDict[date]['B10_L1']['KELVIN_CONS_2'],
                            sens_radiance,
                            tbright_path,
                            img_prep_2.imgDict[date]['B10_L1']['clipped_path'],
                            reference_img)
    
    tbright_path_b11 = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[3], 
                                    os.path.basename(reference_img.replace('B5.TIF', 'tbright_b11.TIF')))
    
    tbright_b11 = process_func.bt(img_prep_2.imgDict[date]['B11_L1']['KELVIN_CONS_1'],
                            img_prep_2.imgDict[date]['B11_L1']['KELVIN_CONS_2'],
                            sens_radiance,
                            tbright_path_b11,
                            img_prep_2.imgDict[date]['B11_L1']['clipped_path'],
                            reference_img)
    
    

    pprint(f"Calculating Split Window Surface Temperature")

    lse_avg = 0.5 * (lse_b10 + lse_b11)
    lse_diff = lse_b10 - lse_b11

    tbright_diff = tbright - tbright_b11


    lst_sw_path = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[0], 
                               os.path.basename(reference_img.replace('B5.TIF', 'lst_sw.TIF')))
    
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
                                 lst_sw_path, reference_img)