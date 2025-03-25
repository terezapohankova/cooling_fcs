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

    ef_ssebi_path = os.path.join(paths_0.INPUT_FOLDER,
                                 paths_0.FOLDERS[6], 
                                 os.path.basename(reference_img.replace('B5.TIF', 'ef_ssebi.TIF')))
    
    le_ssebi_path = os.path.join(paths_0.INPUT_FOLDER,
                                 paths_0.FOLDERS[5], 
                                 os.path.basename(reference_img.replace('B5.TIF', 'le_ssebi.TIF')))
    
    h_ssebi_path = os.path.join(paths_0.INPUT_FOLDER,
                                paths_0.FOLDERS[5], 
                                os.path.basename(reference_img.replace('B5.TIF', 'h_ssebi.TIF')))
    
    bowen_path_ssebi = os.path.join(paths_0.INPUT_FOLDER,
                                    paths_0.FOLDERS[7], 
                                    os.path.basename(reference_img.replace('B5.TIF', 'bowen_ssebi.TIF')))
    
    eta_ssebi_path = os.path.join(paths_0.INPUT_FOLDER,
                                  paths_0.FOLDERS[6], 
                                  os.path.basename(reference_img.replace('B5.TIF', 'eta_ssebi.TIF')))
    
    
    ef_ssebi = process_func.ef_ssebi(cci_5.albd, 
                                     lst_4.lst_sw, 
                                     ef_ssebi_path, 
                                     reference_img) #evaporative fraction
    
    le_flux_ssebi = process_func.le_ssebi(ef_ssebi, 
                                          cci_5.net_radiation, 
                                          cci_5.g_flux_ssebi, 
                                          le_ssebi_path, 
                                          reference_img)    # latent heat flux
    
    h_flux_ssebi = process_func.h_ssebi(ef_ssebi, 
                                        cci_5.net_radiation, 
                                        cci_5.g_flux_ssebi, 
                                        h_ssebi_path, 
                                        reference_img) #sensible heat flux

    bowen_ssebi = process_func.bowenIndex(h_flux_ssebi, 
                                          le_flux_ssebi, 
                                          bowen_path_ssebi, 
                                          reference_img) #bowen ratio