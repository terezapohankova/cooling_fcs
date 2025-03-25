import os
import logging
from pprint import pprint
import img_prep_2
import process_func
import paths_0
import const_param_0
import lst_4
import cci_5
import bowen_ratio_6
import sys


pprint('===================================')
pprint("Calculating S-SEBI and SSEBop")
pprint('===================================')
for date in img_prep_2.sensing_date_list:

    pprint('===================================')
    pprint(f"'Sensing Date Calculated: {date}'")
    pprint('===================================')

    reference_img = img_prep_2.imgDict[date]['B5_L2']['clipped_path']
    
   
    g_flux_MJ = process_func.W_to_MJday(cci_5.g_flux_ssebi)  #convert W/m2 to MJ/s/day
    le_flux_MJ = process_func.W_to_MJday(bowen_ratio_6.le_flux_ssebi)

    
    eta_ssebi = process_func.et_a_day_ssebi(le_flux_MJ, 
                                            cci_5.net_radiation_MJ,
                                            const_param_0.lheat_vapor * (10**6), 
                                            process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['et'], 
                                                            'ETA_SSEBI.TIF'), 
                                                            reference_img)

    
    cold_pix, hot_pix = process_func.identify_cold_hot_pixels(lst_4.lst_sw, 
                                                              lst_4.ndvi, 
                                                              cci_5.albd)
    etf_ssebop = process_func.etf_ssebop(lst_4.lst_sw, 
                                         hot_pix, 
                                         cold_pix, 
                                         process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['et'], 
                                                            'ETF_SSEBOP.TIF'), 
                                         reference_img)
    
    le_ssebop = process_func.le_ssebop(etf_ssebop, 
                                       cci_5.net_radiation, 
                                       cci_5.g_flux_ssebi, 
                                       process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['et'], 
                                                            'LE_SSEBOP.TIF'), 
                                       reference_img)
    
    h_ssebop = process_func.h_ssebop(etf_ssebop, 
                                     cci_5.net_radiation, 
                                     cci_5.g_flux_ssebi, 
                                     process_func.generate_image_path(img_prep_2.imgDict, 
                                                            date,
                                                            paths_0.INPUT_FOLDER, 
                                                            paths_0.FOLDERS['et'], 
                                                            'H_SSEBOP.TIF'), 
                                     reference_img)
    
    
    eta_ssebop = process_func.et_a_day_sssebop(process_func.W_to_MJday(le_ssebop), 
                                               cci_5.net_radiation_MJ, 
                                               const_param_0.lheat_vapor * (10**6), 
                                               process_func.generate_image_path(img_prep_2.imgDict, 
                                                                                date,
                                                                                paths_0.INPUT_FOLDER, 
                                                                                paths_0.FOLDERS['et'], 
                                                                                'eta_ssebop.TIF', 
                                                                                ))
