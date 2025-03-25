from pprint import pprint
import img_prep_2
import process_func
import os
import paths_0
import const_param_0
import meteo_3
import lst_4


pprint('===================================')
pprint("Calculating CCI")
pprint('===================================')
for date in img_prep_2.sensing_date_list:

    pprint('===================================')
    pprint(f"'Sensing Date Calculated: {date}'")
    pprint('===================================')

    reference_img = img_prep_2.imgDict[date]['B5_L2']['clipped_path']

    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    esun_values = [const_param_0.esun[band] for band in bands]
    esun_sum = sum(const_param_0.esun.values())
    

    lb_values = {}

    for band in bands:
        band_path = os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', f'L_{band}.TIF')))
        
        lb_values[band] = process_func.lb_band(
            img_prep_2.imgDict[date][f'{band}_L1']['RADIANCE_ADD'],
            img_prep_2.imgDict[date][f'{band}_L1']['RADIANCE_MULT'],
            img_prep_2.imgDict[date][f'{band}_L1']['clipped_path'],
            band_path
        )
        
    # Reflectivity calculations for all bands
    reflectivity = {}
    for i, band in enumerate(bands):
        reflectivity[band] = process_func.reflectivity_band(
            lb_values[band], 
            esun_values[i], 
            meteo_3.inverseSE,
            img_prep_2.imgDict[date]['B2_L1']['clipped_path'], 
            meteo_3.zenithAngle,
            os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[3], os.path.basename(reference_img.replace('B5.TIF', f'R_{band}.TIF')))
        )
        
  
    pb = {band: process_func.pb(const_param_0.esun[band], esun_sum) for band in bands}

    albedo_toa = sum(process_func.albedo_toa_band(pb[band], 
                                                  reflectivity[band]) for band in bands)

    process_func.savetif(albedo_toa, 
                         os.path.join(paths_0.INPUT_FOLDER, paths_0.FOLDERS[3], 
                                      os.path.basename(reference_img.replace('B5.TIF', 'albedo_toa.TIF'))),
                          reference_img)
    
    albedo_path = os.path.join(paths_0.INPUT_FOLDER, 
                               paths_0.FOLDERS[4], 
                               os.path.basename(reference_img.replace('B5.TIF', 'albedo.TIF')))
    
    albd = process_func.albedo(albedo_toa, 
                               meteo_3.transmis_atm, 
                               albedo_path, 
                               reference_img)

    
    ########## SOLAR NET RADIATION  ######################
    
    # Calculate Longwave Radiation Outwards
    
    rad_long_out_path = os.path.join(paths_0.INPUT_FOLDER,paths_0.FOLDERS[2], 
                                     os.path.basename(reference_img.replace('B5.TIF', 'RadLongOut.TIF')))

    rad_long_out = process_func.longout(lst_4.lse_b10, 
                                        lst_4.lst_sw, 
                                        reference_img, rad_long_out_path)

    # Calculate Longwave Radiation Inwards
    rad_long_in_path = os.path.join(paths_0.INPUT_FOLDER,
                                    paths_0.FOLDERS[2], 
                                    os.path.basename(reference_img.replace('B5.TIF', 'RadLongIn.TIF')))
    
    rad_long_in = process_func.longin(meteo_3.emissivity_atmos, 
                                      lst_4.lst_sw, 
                                      reference_img, 
                                      rad_long_in_path)

    # Calculate Shortwave Radiation Inwards
    rad_short_in = process_func.shortin(const_param_0.SOLAR_CONSTANT, 
                                        meteo_3.zenithAngle, 
                                        meteo_3.inverseSE, 
                                        meteo_3.transmis_atm)
    #pprint(rad_short_in)
    # Calculate Shortwave Radiation Outwards
    rad_short_out_path = os.path.join(paths_0.INPUT_FOLDER,paths_0.FOLDERS[2], 
                                      os.path.basename(reference_img.replace('B5.TIF', 'RadShortOut.TIF')))
    
    rad_short_out = process_func.shortout(albd, 
                                          rad_short_in, 
                                          reference_img, 
                                          rad_short_out_path)

    # Calculate Net Radiation
    net_radiation_path = os.path.join(paths_0.INPUT_FOLDER,
                                      paths_0.FOLDERS[2], 
                                      os.path.basename(reference_img.replace('B5.TIF', 'netRadiation.TIF')))
    
    net_radiation = process_func.netradiation(rad_short_in, 
                                              rad_short_out, 
                                              rad_long_in, 
                                              rad_long_out, 
                                              reference_img, 
                                              net_radiation_path,
                                              albd, 
                                              lst_4.lse_b10)

    ### Paths
    g_path_ssebi = os.path.join(paths_0.INPUT_FOLDER, 
                                paths_0.FOLDERS[5], 
                                os.path.basename(reference_img.replace('B5.TIF', 'g_ssebi.TIF')))
    
    g_flux_ssebi = process_func.soilGFlux_ssebi(lst_4.lst_sw, 
                                                albd, 
                                                lst_4.ndvi, 
                                                net_radiation, 
                                                g_path_ssebi, 
                                                reference_img) #soil heat flux
    g_flux_MJ = process_func.W_to_MJday(g_flux_ssebi)  #convert W/m2 to MJ/s/day
    
    net_radiation_MJ = process_func.W_to_MJday(net_radiation) #convert W/m2 to MJ/s/day
    

    ################################
    ### PENMAN-MONETITH  ##
    ################################

    et0_PM_path = os.path.join(paths_0.INPUT_FOLDER,paths_0.FOLDERS[6], 
                               os.path.basename(reference_img.replace('B5.TIF', 'et0_day_PM.TIF')))
    
    PM_ea_path = os.path.join(paths_0.INPUT_FOLDER,paths_0.FOLDERS[6], 
                              os.path.basename(reference_img.replace('B5.TIF', 'eta_day_PM.TIF')))

    ET0_pm = process_func.ET0(meteo_3.e0,
                              meteo_3.slope_vap_press, 
                              img_prep_2.meteorologyDict[date]['wind_sp'], 
                              meteo_3.es, 
                              g_flux_MJ, 
                              meteo_3.psychro, 
                              net_radiation_MJ, 
                              img_prep_2.meteorologyDict[date]['avg_temp'], 
                              et0_PM_path, 
                              reference_img)
    

    lai_path = os.path.join(paths_0.INPUT_FOLDER,
                            paths_0.FOLDERS[1], 
                            os.path.basename(reference_img.replace('B5.TIF', 'lai.TIF')))
    
    lai_index = process_func.lai(lst_4.ndvi, 
                                 lai_path, 
                                 reference_img)

    kc_path = os.path.join(paths_0.INPUT_FOLDER,
                           paths_0.FOLDERS[1], 
                           os.path.basename(reference_img.replace('B5.TIF', 'kc.TIF')))
    
    kc = process_func.Kc_LAI(lai_index, kc_path, reference_img)

    PM_ea_day = process_func.ea_pm(ET0_pm, 
                                   kc, 
                                   PM_ea_path, 
                                   reference_img)


    ################################
    ### CCI  ##
    ################################

    eti_path = os.path.join(paths_0.INPUT_FOLDER,
                            paths_0.FOLDERS[6], 
                            os.path.basename(reference_img.replace('B5.TIF', 'eti.TIF')))
    
    eti = process_func.ETI(kc, ET0_pm, eti_path, reference_img)

    cci_path = os.path.join(paths_0.INPUT_FOLDER, 
                            paths_0.FOLDERS[6], 
                            os.path.basename(reference_img.replace('B5.TIF', 'cci.TIF')))
    
    cci = process_func.CCi(albd, 
                           eti, 
                           paths_0.HILLSHADE, 
                           cci_path, 
                           reference_img)