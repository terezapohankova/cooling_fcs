
from pprint import pprint
import numpy as np
import process_func     # Import custom functions for calculations
import const_param_0    # Import constants and parameters
import img_prep_2       # Import image preprocessing module

# Loop through each sensing date from image preprocessing data
for date in img_prep_2.sensing_date_list:
    pprint('===================================')
    pprint(f'Sensing Date Calculated: {date}')
    pprint('===================================')
    
    

    # Convert air temperature from Celsius to Kelvin
    ta_kelvin = img_prep_2.meteorologyDict[date]['avg_temp'] + 273.15

    # Compute solar zenith angle and inverse square of Earth-Sun distance
    zenithAngle = process_func.zenithAngle(img_prep_2.imgDict[date]['B2_L2']['sunElev'])
    inverseSE = process_func.dr(img_prep_2.imgDict[date]['B2_L2']['dES'])
    
    
    ########## METEOROLOGICAL PARAMETERS ######################
  
    # atmospheric emissivity
    emissivity_atmos = process_func.atmemis(ta_kelvin)
    pprint(f' Atmospherical Emissivity : {round(emissivity_atmos,3)}')

    # atmospheric transmissivity
    transmis_atm = 0.75 + 2 * (10 ** -5) * 219 
    pprint(f' Atmospherical Transmissivity : {round(transmis_atm,3)} ')

    # atmospheric pressure (kPa)
    p = process_func.atmPress(const_param_0.Z)
    pprint(f' Atmospherical Pressure : {round(p,3)} kPa')

    # saturated vapor pressure (kPa)
    es = process_func.es(img_prep_2.meteorologyDict[date]['avg_temp'])
    pprint(f' Saturated Water Vapour Pressure : {round(es,3)} kPa')

    # actual water vapor pressure (kPa)
    e0 = process_func.e0(img_prep_2.meteorologyDict[date]['rel_hum'], es)
    pprint(f' Actual Water Vapour Pressure : {round(e0,3)} kPa')

    # total atmospheric content of water vapor
    water_vap_cm = (0.0981 * (10 * e0 * img_prep_2.meteorologyDict[date]['rel_hum']) + 0.1697)/10
    pprint(f' Total Atmospherical Content of Water Vapour : {round(water_vap_cm,3)} ')

    # slope of vapor pressure curve (kPa/°C)
    slope_vap_press = process_func.slopeVapPress(img_prep_2.meteorologyDict[date]['avg_temp'])
    pprint(f' Slope Water Vapour : {round(slope_vap_press,3)} kPa')

    # air density (kg/m³)
    rho = process_func.densityair(p, 
                                  img_prep_2.meteorologyDict[date]['avg_temp'], 
                                  e0)
    
    pprint(f' Air Density : {round(rho,3)} kg/m^3')

    # psychrometric constant (kPa/°C)
    psychro = process_func.psychroCons(p)
    pprint(f' Psychrometric Constant : {round(psychro,3)} kPa/˚C')

    # wet bulb temperature (°C)
    tw_bulb = process_func.Tw_bulb(img_prep_2.meteorologyDict[date]['avg_temp'], 
                                      img_prep_2.meteorologyDict[date]['rel_hum'])
    pprint(f' Wet Bulb Temperature : {round(tw_bulb,3)} ˚C')
