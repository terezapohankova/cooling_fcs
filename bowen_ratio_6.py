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

# Loop through each available sensing date
for date in img_prep_2.sensing_date_list:

    pprint('===================================')
    pprint(f"'Sensing Date Calculated: {date}'")
    pprint('===================================')

    # Get the reference image path for spatial alignment
    reference_img = img_prep_2.imgDict[date]['B5_L2']['clipped_path']

    # Calculate Evaporative Fraction (EF) using the SSEBI method
    ef_ssebi = process_func.ef_ssebi(
        cci_5.albd,  
        lst_4.lst_sw,  
        process_func.generate_image_path(img_prep_2.imgDict, date, 
                                         paths_0.INPUT_FOLDER, paths_0.FOLDERS['et'], 
                                         'EF_SSEBI.TIF'),  
        reference_img) 

    # Compute Latent Heat Flux (LE) using the SSEBI method
    le_flux_ssebi = process_func.le_ssebi(
        ef_ssebi,  
        cci_5.net_radiation,  
        cci_5.g_flux_ssebi, 
        process_func.generate_image_path(img_prep_2.imgDict, date, 
                                         paths_0.INPUT_FOLDER, paths_0.FOLDERS['fluxes'], 
                                         'LE_SSEBI.TIF'),  
        reference_img)  

    # Compute Sensible Heat Flux (H) using the SSEBI method
    h_flux_ssebi = process_func.h_ssebi(
        ef_ssebi,  
        cci_5.net_radiation,  
        cci_5.g_flux_ssebi,  
        process_func.generate_image_path(img_prep_2.imgDict, date, 
                                         paths_0.INPUT_FOLDER, paths_0.FOLDERS['fluxes'], 
                                         'H_SSEBI.TIF'),  
        reference_img ) 

    # Compute Bowen Ratio using the Sensible Heat Flux and Latent Heat Flux
    bowen_ssebi = process_func.bowenIndex(
        h_flux_ssebi,  
        le_flux_ssebi,  
        process_func.generate_image_path(img_prep_2.imgDict, date, 
                                         paths_0.INPUT_FOLDER, paths_0.FOLDERS['bowen'], 
                                         'BR.TIF'),  
        reference_img  
    ) 