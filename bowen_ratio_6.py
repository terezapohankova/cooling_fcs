from pprint import pprint
import img_prep_2
import process_func
import os
import paths_0
import const_param_0
import meteo_3
import lst_4
import cci_5


pprint('===================================')
pprint("Calculating Bowen Ratio")
pprint('===================================')
for date in img_prep_2.sensing_date_list:

    pprint('===================================')
    pprint(f"'Sensing Date Calculated: {date}'")
    pprint('===================================')

    reference_img = img_prep_2.imgDict[date]['B5_L2']['clipped_path']   
    
    ef_ssebi = process_func.ef_ssebi(cci_5.albd, 
                                     lst_4.lst_sw, 
                                     process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['et'], 
                                                            'EF_SSEBI.TIF'), 
                                     reference_img) #evaporative fraction
    
    le_flux_ssebi = process_func.le_ssebi(ef_ssebi, 
                                          cci_5.net_radiation, 
                                          cci_5.g_flux_ssebi, 
                                          process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['fluxes'], 
                                                            'LE_SSEBI.TIF'), 
                                          reference_img)    # latent heat flux
    
    h_flux_ssebi = process_func.h_ssebi(ef_ssebi, 
                                        cci_5.net_radiation, 
                                        cci_5.g_flux_ssebi, 
                                        process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['fluxes'], 
                                                            'H_SSEBI.TIF'), 
                                        reference_img) #sensible heat flux

    bowen_ssebi = process_func.bowenIndex(h_flux_ssebi, 
                                          le_flux_ssebi, 
                                          process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['bowen'], 
                                                            'BR.TIF'), 
                                          reference_img) #bowen ratio