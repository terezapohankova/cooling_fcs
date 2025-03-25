
from pprint import pprint
import numpy as np
import process_func
import const_param_0
import img_prep_2


for date in img_prep_2.sensing_date_list:
    pprint('===================================')
    pprint(f'Sensing Date Calculated: {date}')
    pprint('===================================')
    
    

    # from kg m-2 to g cm-2
    #water_vap_cm = ((meteorologyDict[date]['total_col_vat_wap_kg']) / 10)
    
    ta_kelvin = img_prep_2.meteorologyDict[date]['avg_temp'] + 273.15

    zenithAngle = process_func.zenithAngle(img_prep_2.imgDict[date]['B2_L2']['sunElev'])
    inverseSE = process_func.dr(img_prep_2.imgDict[date]['B2_L2']['dES'])
    
    
    ########## METEOROLOGICAL PARAMETERS ######################
    #pprint(f"Calculating Atmosphere Emissiivty for {date}")
    
    emissivity_atmos = process_func.atmemis(ta_kelvin)
    pprint(f' Atmospherical Emissivity : {round(emissivity_atmos,3)}')

    transmis_atm = 0.75+2*(10**-5)*219  #process_func.atm_transmiss(theta_vals['theta1'])
    pprint(f' Atmospherical Transmissivity : {round(transmis_atm,3)} ')

    p = process_func.atmPress(const_param_0.Z)
    pprint(f' Atmospherical Pressure : {round(p,3)} kPa')

    es = process_func.es(img_prep_2.meteorologyDict[date]['avg_temp'])
    pprint(f' Saturated Water Vapour Pressure : {round(es,3)} kPa')

    e0 = process_func.e0(img_prep_2.meteorologyDict[date]['rel_hum'], es)
    pprint(f' Actual Water Vapour Pressure : {round(e0,3)} kPa')


    water_vap_cm = (0.0981 * (10 * e0 * img_prep_2.meteorologyDict[date]['rel_hum']) + 0.1697)/10
   
    pprint(f' Total Atmospherical Content of Water Vapour : {round(water_vap_cm,3)} ')

    slope_vap_press = process_func.slopeVapPress(img_prep_2.meteorologyDict[date]['avg_temp'])
    pprint(f' Slope Water Vapour : {round(slope_vap_press,3)} kPa')

    rho = process_func.densityair(p, 
                                  img_prep_2.meteorologyDict[date]['avg_temp'], 
                                  e0)
    
    pprint(f' Air Density : {round(rho,3)} kg/m^3')

    psychro = process_func.psychroCons(p)
    pprint(f' Psychrometric Constant : {round(psychro,3)} kPa/˚C')

    tw_bulb = process_func.Tw_bulb(img_prep_2.meteorologyDict[date]['avg_temp'], 
                                      img_prep_2.meteorologyDict[date]['rel_hum'])
    pprint(f' Wet Bulb Temperature : {round(tw_bulb,3)} ˚C')
