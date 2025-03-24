Z = 650
MEASURING_HEIGHT = 2
CP = 1004  
B_GAMMA = 1324 # [K]
SOLAR_CONSTANT = 1367 #[W/m2]


####ESUN
# 10.3390/rs12030498
esun = {'B2' : 2067,
        'B3' : 1893,
        'B4' : 1603,
        'B5' : 972.6,
        'B6' : 245,
        'B7' : 79.72}

## Skoković, D., Sobrino, J. A., Jimenez-Munoz, J. C., Soria, G., Julien, Y., Mattar, C., & Cristóbal, J. 
#(2014). Calibration and Validation of land surface temperature for Landsat8-TIRS sensor. Land product validation and evolution.
emissivity= { #(b10, b11)
        "vegetation": (0.987, 0.989),
        "bare_soil": (0.971, 0.977)
    }


## Jimenez-Munoz, J. C., Sobrino, J. A., Skoković, D., Mattar, C., & Cristobal, J. (2014). 
# Land surface temperature retrieval methods from Landsat-8 thermal infrared sensor data. 
# IEEE Geoscience and remote sensing letters, 11(10), 1840-1843.
c_coeffs = {"c0": -0.268,
            "c1": 1.378,
            "c2": 0.183,
            "c3": 54.300,
            "c4": -2.238,
            "c5": -129.200,
            "c6": 16.400}



lheat_vapor = 2.45 #Latent Heat of Vaporisation


#thetaX_dict["a"] for LST single channel
# https://www.mdpi.com/2072-4292/10/3/431
"""theta1_dict = {
    "a": 4.4729730361,
    "b": -0.0000748260,
    "c": 0.0466282124,
    "d": 0.0231691781,
    "e": -0.0000496173,
    "f": -0.0262745276,
    "g": -2.4523205637,
    "h": 0.0000492124,
    "i": -7.2121979375
}

theta2_dict = {
  "a": -30.3702785256,
  "b": 0.0009118768,
  "c": -0.5731956714,
  "d": -0.7844419527,
  "e": 0.0014080695,
  "f": 0.2157797227,
  "g": 106.5509303783,
  "h": -0.0003760208,
  "i": 89.6156888857
}

theta3_dict = {
  "a": -3.7618398628,
  "b": -0.0001417749,
  "c": 0.0911362208,
  "d": 0.5453487543,
  "e": -0.0009095018,
  "f": 0.0418090158,
  "g": -79.9583806096,
  "h": -0.0001047275,
  "i": -14.6595491055
}"""


