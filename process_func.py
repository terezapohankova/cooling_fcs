import os
import fiona
import rasterio
import rasterio.mask
from pprint import pprint
import numpy as np
import tifffile as tf
import rasterio as rio
from affine import Affine
from datetime import datetime
import warnings
from osgeo import gdal
import math
import json

import matplotlib.pyplot as plt
from scipy import stats
import warnings



############################################################################################################################################
#   GENERAL FUNCTIONS
############################################################################################################################################

def getfilepath(input_folder, suffix):
    """_summary_

    Args:
        input_folder (string):          path to chosen folder
        suffix (string):                suitable suffix for filtering (e.g. TIF, JSON)
    Returns:
        list:                           list of filtered paths
    """
   
    pathListFolder = []
    for root, dirs, files in os.walk(input_folder, topdown=False):
        for name in files:
            if name.endswith(suffix):
                if name not in pathListFolder:
                    pathListFolder.append(os.path.join(root, name))
    return pathListFolder

######################################################################

def createmeteodict(csv_file):
    """create dictionary for meteorological data

    Args:
        csv_file (string):          path to CSV file containing meteodata

    Returns:
        dictionary:                 input CSV file converted to dictionary
    """
  
    
    with open(csv_file, mode = 'r') as infile:
        csv_list = [[val.strip() for val in r.split(",")] for r in infile.readlines()] #[['date', 'avg_temp', 'wind_sp', 'relHum', 'max_temp', 'min_temp'],
                                                                                       #['20220518', '14.25', '1.25', '65.52', '19.8', '8.7'],
                                                                                       #['20220612', '22.60', '0.83', '70.21', '28.4', '13.3']]

        (_, *header), *data = csv_list                                              #(('date', 'avg_temp', 'wind_sp', 'relHum', 'max_temp', 'min_temp'),
                                                                                    #['20220518', '14.25', '1.25', '65.52', '19.8', '8.7'],
                                                                                    #['20220612', '22.60', '0.83', '70.21', '28.4', '13.3'])
        csv_dict = {}
        for row in data:
            key, *values = row                                                      # ('20220612', '22.60', '0.83', '70.21', '28.4', '13.3')
            csv_dict[key] = {key : float(value) for key, value in zip(header, values)}     #{'date': [avg_temp': '22.60','max_temp': '28.4','min_temp': '13.3','relHum': '70.21','wind_sp': '0.83']}
    return  csv_dict

######################################################################

def loadjson(jsonFile):
    """load Landsat metadata from JSON file

    Args:
        jsonFile (string):          path to JSON file with metadata

    Returns:
        dictionary:                 converted JSON file 
    """

    with open(jsonFile, 'r') as j:
        data = json.load(j)
    
        return data

######################################################################

def clipimage(maskpath, inputBand, outImgPath, cropping = True, filling = True, inversion = False):

    """satellite image clipping using a pre-prepared GeoPackage mask in the same coordinate system as the images
        https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html
    
    Args:
        maskPath (str):             path to polygon mask
        inputBand (str):            path to image to be clipped
        outImgPath (str):           path to new (cropped) image
        cropping (bool):            whether to crop the raster to the mask extent (default True)
        filling (bool):             whether to set pixels outside the extent to no data (default True)
        inversion (bool):           whether to create inverse mask (default False)
    
    Returns:
        none
    """

    with fiona.open(maskpath, "r") as gpkg:
        shapes = [feature["geometry"] for feature in gpkg]

    with rasterio.open(inputBand) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop = cropping, filled = filling, invert = inversion)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff", # output format GeoTiff
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "lzw",
                    "tiled" : True})

    with rasterio.open(outImgPath, "w", **out_meta) as dest:
        pprint(f"mask: {maskpath}")
        pprint(f"original image: {inputBand}")
        pprint(f"cropped image: {outImgPath}")
        dest.write(out_image)
    
   
    return 

######################################################################

def savetif(img_new, outputPath, image_georef, epsg = 'EPSG:32633'):
    """ Save created image in chosen format to chosen path.

    Args:
        image_georef (string):      path to sample georeferenced image with parameters to be copied to new image
        img_new (numpy.array):      newly created image to be saved
        outputPath (str):           path including new file name to location for saving
        epsg (str):                 EPSG code for SRS (default 'EPSG:32633')

    Returns:
        none
    """    
    step1 = gdal.Open(image_georef, gdal.GA_ReadOnly) 
    GT_input = step1.GetGeoTransform()
    afn = Affine.from_gdal(* GT_input)
    new_dataset = rio.open(outputPath, "w", 
        driver = "GTiff",
        height = img_new.shape[0],
        width = img_new.shape[1],
        count = 1,
        #nodata = -9999, # optinal value for nodata
        dtype = img_new.dtype,
        crs = epsg, # driver for coordinate system code
        transform=afn,
        compress = "lzw")
    new_dataset.write(img_new, 1)
    new_dataset.close()
 
    return



"""def scatterplot_2d(x, y, var1, var2, pathout):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = r_value ** 2
    sigma = np.std(y - x)
    bias = np.sum(np.mean(y - x))
    rmse = np.sqrt(bias ** 2 + sigma ** 2)
    num = len(x)
    mae = np.sum(np.abs(y - x)) / num
    
    print('r2: ' + str(r2))
    print('sigma: ' + str(sigma))
    print('bias: ' + str(bias))
    print('rmse: ' + str(rmse))
    print('mae: ' + str(mae))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # set figure,axes and size
    ax.set_title(var1+' VS '+var2, size=18)  # title
    plt.xlabel(var1, fontsize=15)  # label x
    plt.ylabel(var2, fontsize=15)  # label y
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)

    cmap = plt.cm.jet
    plt.hist2d(x, y, bins=500, cmap=cmap, cmin=1)
    cbar = plt.colorbar()
    cbar.set_label('Count')

    # Calculate the linear regression line
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept

    # Plot the linear regression line
    plt.plot(x_fit, y_fit, color='red', linewidth=2,linestyle='dashed')

    # set limits
    max_x_val, max_y_val = float(np.percentile(x, 99)), float(np.percentile(y, 99))
    min_x_val, min_y_val = float(np.percentile(x, 1)), float(np.percentile(y, 1))
    min_val = np.min([min_x_val,min_y_val])
    max_val = np.max([max_x_val, max_y_val])
    plt.xlim(left=min_x_val)
    plt.xlim(right=max_x_val)
    plt.ylim(top=max_y_val)
    plt.ylim(bottom=min_y_val)

    t = 'RMSE = %.2f\nR^2 = %.2f\nsigma = %.2f\nbias = %.2f\nMAE = %.2f  ' % (rmse, r2, sigma, bias, mae)
    plt.text(x=0.55, y=0.35, s=t,
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes, size=15, bbox=dict(fc='white'))

    #plt.savefig(pathout + '.png', dpi=150)
    #plt.show()
    return"""


def W_to_MJday(variable):
    """Convert variable in W/m^2 to MJ/day

    Args:
        variable (numpy.array; float):            input variable [W/m^2]

    Returns:
        numpy.array; float :                      output variable [MJ/day]
    """
    return (variable * 0.0864)
############################################################################################################################################
#   ALBEDO                                                                                                                                 #
# via https://www.scielo.br/j/rbeaa/a/sX6cJjNXWMfHQ5p4h33B8Zz/?lang=en&format=pdf                                                          #
############################################################################################################################################
                                                                                                                                    
def dr(sunDist):

    """Correction of the eccentricity of the terrestrial orbit

    Args:
        sunDist (float):          distance Earth-Sun  extracted from image metadata (EARTH_SUN_DISTANCE) [AU]

    Returns:
        dr_num (float):           value of corrected eccentricity of terrestrial orbit  [AU] 
    """

    dr_num = 1 / (sunDist ** 2)
     
    return dr_num

############################################################################################################################################

def zenithAngle(sunElev):

    """ Calculation of sun zenith angle 
    
    Args:
        sunElev (float):          Sun Elevation angle extracted from image metadata (SUN_ELEVATION) [rad]

    Returns:
        zenithan (float):         Zenith angle [rad]
    """
    zenithan = ((90 - sunElev))
    return zenithan

############################################################################################################################################

def lb_band(offset, gain, band_path, outputPath):
   
    """calculate pixel radiance

    Args:
        offset (float):     Additive Radiance Term from the Metadata (RADIANCE_ADD_BAND_x)
        gain (float):       Multiplicative Rescaling Factor from the metadata (RADIANCE_MULT_BAND_x)
        band_path (str):    path to input Landsat band

    Returns:
        Lb (numpy.array):   pixel radiance for a band [W/(m^2 * srad * μm)]
    """
    band = np.array(tf.imread(band_path))
    Lb = offset + gain * band
    
    savetif(Lb, outputPath, band_path)
    return Lb

############################################################################################################################################

def reflectivity_band(lb, esun_band, dr, band_path, zenithAngle, outputPath):
   
    """
    # Band Reflectance [W/m2]

    AddRef = Reflectance Additive Term from the Metadata (REFLECTANCE_ADD_BAND_X)
    MultRef =  Reflectance Multiplicative Term from the Metadata (REFLECTANCE_MULT_BAND_X)
    band = Landsat band
    dr = Correction of the eccentricity of the terrestrial orbit
    zenithAngle = Sun zenith angle in radians
    """
    band = np.array(tf.imread(band_path))
    
    rb = (3.14 * lb) / (esun_band * (math.cos(math.radians(zenithAngle)) * (dr)))
    
    savetif(rb, outputPath, band_path)
    return rb

######################################################################

def pb(esun_band, esun_sum):
    
    return esun_band / esun_sum


def albedo_toa_band(pbx,rbx):
    """planetary albedo (without atmospheric correction)

    Args:
        pbx (float):          band weight derived from its influence on albedo
        rbx (numpy.array):    band reflectivity

    Returns:
        numpy.array:          uncorrected planetary albedo [unitless]
    """
  

    toaPlanet = (pbx * rbx)   
    return toaPlanet


######################################################################

def albedo(toaplanet, atm_trans, outputPath, band_path):
    """ corrected albedo
    
    Args:

        toaplanet (numpy.array):  planetary top of atmosphere radiance  
        atm_trans /float):        Atmospheric transmittance in the solar radiation domain
    
    Returns:
        numpy.array:              corrected albedo
    
    """
    albedo = (toaplanet - 0.03) / (atm_trans ** 2)
    albedo = np.where(albedo < 0, np.nan, albedo)
    
    savetif(albedo, outputPath, band_path)
    return albedo


############################################################################################################################################
#   ATMOSHEPRIC FUNCTIONS
############################################################################################################################################

def atm_press(elevation):

    P0 = 1013.25  # Sea level standard atmospheric pressure (hPa)
    T0 = 288.15   # Standard temperature at sea level (K)
    L = 0.0065    # Temperature lapse rate (K/m)
    g = 9.80665   # Gravitational acceleration (m/s²)
    M = 0.0289644 # Molar mass of Earth's air (kg/mol)
    R = 8.3144598 # Universal gas constant (J/(mol*K))

    # Barometric formula
    pressure = P0 * (1 - (L * elevation) / T0) ** ((g * M) / (R * L))

    return round(pressure, 3)  # Pressure in hPa


def e0(Ta):
    """Partial Water Vapour Pressure (actual vapour pressure)
        # It is a measure of the tendency of a substance to evaporate or transition from its condensed phase to the vapor phase.
        # Tetens Formula via https://www.omnicalculator.com/chemistry/vapour-pressure-of-water
    
    Args:
        Ta (float):     average daily air temperature [˚C]

    Returns:
        float:          value of actual vapour pressure [kPa]
    """
    
    e0 = 0.6108 * math.exp(((17.27 * Ta) / (Ta + 237.3)))
    return e0

######################################################################

def es(Ta):
    """ Saturated Vapour Pressure [kPa]
        # via https://www.fao.org/3/x0490e/x0490e07.htm#atmospheric%20pressure%20(p)
    Args: 
        Ta (float):     average daily air temperature [˚C]

    Returns:
        float:          value of saturated vapour pressure [kPa]
    """
    
    es = (6.1078 * 10**(7.5 * Ta /(Ta + 237.3))) / 10
    return es

######################################################################

def atmPress(Z):
    """Atmospheric Pressure [kPa]
       # via https://designbuilder.co.uk/helpv3.4/Content/Calculation_of_Air_Density.htm

    Args:
        Z (float):  Elevation above sea level [m]

    Returns:
        float:      Atmospheric Pressure [kPa] 
    """
    
    P = (101325 * ((1.0 - (Z * 0.0000225577)) ** 5.2559)) / 1000
    return P

######################################################################

def psychroCons(P):
    """
    # Psychrometric constant [kPa/˚C]
    # via https://www.fao.org/3/x0490e/x0490e07.htm#psychrometric%20constant%20(g)

    P = Atmospheric Pressure [kPa]
    """
    y =  0.000665 * P
    return y

######################################################################

def densityair(P, Ta, RH, R = 278.05):
    """
    Density of Air [kg/m3]
    # via https://designbuilder.co.uk/helpv3.4/Content/Calculation_of_Air_Density.htm
    
    R  = Gas Constant (287.05 J/kg-K)
    Ta = Air Temperature in K (T [˚C] + 273.15)
    P = Standard Pressure [kPa]
    """
    P = P * 1000

    #saturated vapour pressure in Pa
    p1 = 6.1078 * (10**(7.5 * Ta / (Ta + 237.3)))
    
    # the water vapor pressure in Pa
    pv = p1 * RH

    #pressure of dry air
    pd = P - pv
    air_density = (pd / (R * (Ta + 273.15))) + (pv / (461.495 * (Ta + 273.15)))
  
    return air_density

######################################################################

def	atmemis(Ta_K):
	#(Brutsaert 1982).
    """
    # Emmisivity of the Atmosphere
    # via https://www.nature.com/articles/s41598-023-40499-6

    Ta = Air Temperature [K]
    
    """
    
    atmEmis = 0.0000092 * ((Ta_K) **  2)
    return atmEmis

######################################################################

def z0m(vegHeight, outputPath, reference_img):
    """
    # Roughness length governing momentum transfer [m]
    # https://posmet.ufv.br/wp-content/uploads/2017/04/MET-479-Waters-et-al-SEBAL.pdf
    vegHeigth = height of vegetation
    """    

    z0m = 0.123 * 0.3
    #z0m[z0m <=0] = 0.1

    #savetif(z0m, outputPath, reference_img)
    return z0m

def wind_speed_blending(wind_ref_h, blending_height, momentum_z0m, ref_height):
    wind_bl_hg = wind_ref_h * np.log(blending_height / momentum_z0m) / (np.log(ref_height / momentum_z0m))
    #wind_bl_hg = np.where(wind_bl_hg < 0, np.nanmean(wind_bl_hg), wind_bl_hg)
    
    #savetif(wind_bl_hg, outputpath, reference_img)
    return wind_bl_hg

def u_fric_vel(u_bl_heigh, blend_height, momentum_z0m, k = 0.41):
    u_ast = (k * u_bl_heigh) / np.log(blend_height / momentum_z0m)
    
    return u_ast
    
def rah(z1, z2, fric_vel, k = 0.41):
    r_ah = (np.log(z2 / z1) ) / (fric_vel * k)
    #r_ah[np.isnan(r_ah)] = np.nanmean(r_ah)
    
    #savetif(r_ah, outputpath, reference_img)
    return r_ah


######################################################################
def slopeVapPress(Ta):
    slopeVapPress = round((4098 * 0.6108 * np.exp ((17.27 * Ta)/(Ta+237.3)) / (Ta + 237.3) ** 2),3)
    return slopeVapPress

######################################################################

# Stull formula; relative humidities between 5% and 99% and temperatures between -20°C and 50°C. 
def Tw_bulb(Ta, RH):
    tw = (Ta * np.arctan(0.151977 * (np.sqrt(RH + 8.31659)))) + \
    + ((0.00391838 * np.sqrt(RH**3)) * np.arctan(0.023101*RH)) - \
    (np.arctan(RH - 1.676331)) + np.arctan(Ta+RH) - 4.686035

    return tw

def tr(air_temp_k):
    return 1 - (373.15 / air_temp_k)

def e_aster(Po, Tr):
    return Po * np.exp((13.3185*Tr) - (1.976*(Tr**2)) - (0.6445*(Tr**3)) - (0.1299*(Tr**4)))

def delta_pt(e_ast, air_temp_k, Tr):
    return ((373.15 * e_ast) / (air_temp_k**2)) * (13.3185 - (3.952*Tr) - (1.9335*(Tr**2)) - (0.5196*(Tr**3)))

######################################################################

def ET0(e0, SatVapCurve, WindSp, es, G, psych, Rn, Ta, outputPath, band_path):
    # https://edis.ifas.ufl.edu/publication/AE459
    #VPD = es-e0
    ET0_daily = 0.408 * SatVapCurve * (Rn - G) + (900 * psych * WindSp * (es - e0)) /  (Ta+273.15) / (SatVapCurve + psych * (1 + 0.34 * WindSp))

    savetif(ET0_daily, outputPath, band_path)
    #savetif(ET_rad, outputPath, band_path)
    return ET0_daily

def ea_pm(et0, kc, outputPath, reference_img):

    ea_pm  =  et0 * kc
     

    savetif(ea_pm, outputPath, reference_img)
    return

############################################################################################################################################
#   RADIATION
############################################################################################################################################



def bt(K1, K2, L_sen, outputPath, thermal_band, reference_band):
    """
    # Top of Atmosphere Brightness Temperature [˚C/K]
    # via Landsat 8 Data Users Handbook (https://www.usgs.gov/media/files/landsat-8-data-users-handbook)

    K1 = Band-specific thermal conversion constant from the metadata (K1_CONSTANT_BAND_x, where x is the thermal band number)
    K2 = Band-specific thermal conversion constant from the metadata (K2_CONSTANT_BAND_x, where x is the thermal band number)
    RadAddBand = Radiation Additive Term from the Metadata (RADIANCE_ADD_BAND_X)
    RadMultBand =  Radiation Multiplicative Term from the Metadata (RADIANCE_MULT_BAND_X)
    band = Landsat Thermal Band
    TOARad = Top of Atmosphere Radiation
    """
    
    thermal_band = np.array(tf.imread(thermal_band))
    #TOARad = RadMultBand * thermal_band + RadAddBand ## calibrated radiance TOA W/(m^2 sr)

    #BT = (K2 / np.log(K1/TOARad + 1)) #- 273.15      ## brightness temeprature in ˚C
    #BT[BT == -273.15] = 0

    BT = (K2 / np.log(K1 / L_sen + 1))
    savetif(BT, outputPath, reference_band)
    return BT



# sensor radiance
def sensor_radiance(gain, thermal_band, offset, outputPath, reference_band):
    warnings.filterwarnings('ignore')

    thermal_band = np.array(tf.imread(thermal_band))
    #Lsens = K1 / (np.exp(K2 / bt) - 1)
    Lsens = gain * thermal_band + offset
    savetif(Lsens, outputPath, reference_band)
    return Lsens


def gamma(bt, B_GAMMA, sensor_radiance):
    # https://sci-hub.se/10.1109/lgrs.2014.2312032
    gamma_calc = (bt ** 2) / (B_GAMMA * sensor_radiance)
    return gamma_calc

def delta(bt, B_GAMMA, gamma_c, sens_radiance):
    #https://sci-hub.se/10.1109/lgrs.2014.2312032
    delta_calc = bt - ((bt ** 2)/B_GAMMA) 

    return delta_calc
######################################################################

def longout(emisSurf, LST, reference_band_path, outputPath):
    """
    # Outgoing Longwave Radiation [W/m2]
    # via Stephan-Boltzmann law https://doi.org/10.1016/j.jrmge.2016.10.004

    emisSurf = emissivity of surface [-]
    LST = Land Surface Temprature [˚C]
    """
    #LST = LST + 273.15
    longOut = emisSurf * 5.6703 * 10.0 ** (-8.0) * LST ** 4

    savetif(longOut, outputPath, reference_band_path)
    return longOut

######################################################################

def longin(emisAtm, LST, reference_band_path, outputPath):
    """
    # Incoming Longwave Radiation [W/m2]
    # via Stephan-Boltzmann law https://doi.org/10.1016/j.jrmge.2016.10.004

    emis = emissivity of atm [-]
    LST = Land Surface Temprature [˚C]
    """
    #LST = LST + 273.15
    longIn = emisAtm * 5.6703 * 10.0 ** (-8.0) * LST ** 4
    savetif(longIn, outputPath, reference_band_path)
    return longIn

######################################################################

def t(Z):
    return 0.75 + (2 * (10**-5) * Z)


def getDOY(date): #get day of the year
    """ Day of the Year

    Args:
        date (str): Date

    Returns:
        int: Day of the Year
    """    
    convertdate = datetime.strptime(str(date), '%Y%m%d')
    DOY = int(convertdate.strftime('%j'))

    return DOY


def ecc_corr(doy):
    return 1 + 0.033 * math.cos(((2 * math.pi * doy) / 365)) 

def shortin(solar_cons, solar_zenith_angle, inverte_SE, atm_trans):
    # https://posmet.ufv.br/wp-content/uploads/2017/04/MET-479-Waters-et-al-SEBAL.pdf

    #pprint("cos")
    #pprint((math.cos(math.radians(solar_zenith_angle))))
    
    return solar_cons * math.cos(math.radians(solar_zenith_angle)) * inverte_SE * atm_trans

    #return solar_cons * solar_zenith_angle * inverte_SE * atm_trans
    

def atm_transmiss(theta1):

    ## accroding to Jimenez-Munoz, J. C., Sobrino, J. A., Skokovic, D., Mattar, C., & Cristobal, J. (2014). L
    #and Surface Temperature Retrieval Methods From Landsat-8 Thermal Infrared Sensor Data. 
    #IEEE Geoscience and Remote Sensing Letters, 11(10), 1840–1843. doi:10.1109/lgrs.2014.2312032 
    
    
    return  1 / theta1

######################################################################
def shortout(albedo, shortin, reference_band_path, outputPath):
    """
    # Outgoing Shortwave Radiation [W/m2]
    # via https://www.posmet.ufv.br/wp-content/uploads/2016/09/MET-479-Waters-et-al-SEBAL.pdf

    albedo = Albedo [-]
    shortin = Shortwave Incoming Radiation [W/m2]
    """
    shortOut =  albedo * shortin
    savetif(shortOut, outputPath, reference_band_path)  
    return shortOut

######################################################################

def netradiation(shortIn, shortOut, longIn, longOut, reference_band_path, outputPath, albedo, lse):
    """
    # Net Energy Bdget [W/m2]
    # via https://www.redalyc.org/journal/2736/273652409002/html/#redalyc_273652409002_ref4

    shortIn = Incoming Shortwave Radiation [W/m2]
    shortOut = Outgoing Shortwave Radiation [W/m2]
    longIn = Incoming Longwave Radiation [W/m2]
    longOut = Outgoing Longwave Radiation [W/m2]
    """
    Rn = (shortIn - shortOut) + (longIn - longOut)
    short_diff = shortIn - shortOut
    long_diff = longIn - longOut

    #Rn = short_diff + long_diff
    #Rn = np.where(Rn < 0, Rn * (-1), Rn)
    Rn = (1 - albedo) * shortIn + longIn - longOut - (1 - lse) * longIn
   
    savetif(Rn, outputPath, reference_band_path)
    return Rn

############################################################################################################################################
#   SURFACE
############################################################################################################################################

def emis(emis_vege, Pv, emis_soil, output_path, reference_img): #surface emis
  
    surface_emis = emis_vege * Pv + emis_soil * (1 - Pv)
    savetif(surface_emis, output_path, reference_img)

    return surface_emis



######################################################################

def LST(BT, emis, band_path, outputPath):
    """
    # Land Surface Temperature [˚C]
    # via Ugur Avdan and Gordana Jovanovska. “Automated
            Mapping of Land Surface Temperature Using
            LANDSAT 8 Satellite Data”, Journal of Sensors,
            Vol. 2016, Article ID 1480307, 2016.

    BT - Brightness temperature [K]
    emis - emissivity [-]
    band - Landsat band [-]
    """
    band = np.array(tf.imread(band_path))
    
    
    LST = (BT / (1 + ((0.0015 * BT)/1.4488) * np.log(emis)))
    
    #savetif(LST, outputPath, band_path)
    return LST

######################################################################

def bowenIndex(H, LE, outputPath, band_path):
    """
    # Bowen Index
    # via https://daac.ornl.gov/FIFE/Datasets/Surface_Flux/Bowen_Ratio_USGS.html
    H = Sensible Heat Flux [W/m2]
    LE =
     Latent Heat Flux [W/m2]
    """
    #warnings.filterwarnings('ignore')
    BI = H / LE
    #BI[BI < 0] = np.nan
    BI = np.where(BI > 5, np.nan, BI)
    savetif(BI, outputPath, band_path)
    return BI

#########################################################
################### S-SEBI functions ###################
#########################################################

def ef_ssebi(albedo, lst, output_path, referecnce_band):
    
    # Initialize arrays for LST max and min
    lst_max = np.full(albedo.shape, np.nan)
    lst_min = np.full(albedo.shape, np.nan)

    # Determine LSTmax and LSTmin for different albedo bins
    albedo_bins = np.linspace(np.min(albedo), np.max(albedo), 100)

    for i in range(len(albedo_bins) - 1):
        mask = (albedo >= albedo_bins[i]) & (albedo < albedo_bins[i + 1])
        if np.any(mask):
            lst_max[mask] = np.nanmax(lst[mask])
            lst_min[mask] = np.nanmin(lst[mask])

    # Compute normalized LST
    lst_max[np.isnan(lst_max)] = np.nanmax(lst)  # replace NaNs with global max
    lst_min[np.isnan(lst_min)] = np.nanmin(lst)  # replace NaNs with global min

    t_prime = (lst - lst_min) / (lst_max - lst_min)
    t_prime = np.clip(t_prime, 0, 1)  # ensure t_prime is within [0, 1]

    # Calculate evaporative fraction
    ef = 1 - t_prime
    savetif(ef, output_path, referecnce_band)

    return ef

######################################################################

def le_ssebi(ef, rn, g, outputPath, band_path):

    # Latent HEat Flux [W/m2]
    # via Baasriansen, 2000 (BASTIAANSSEN, W. G. M. SEBAL - based sensible and latent heat fluxes in the irrigated Gediz Basin, Turkey. Journal of Hydrology, v.229, p.87-100, 2000.)

    le = ef * (rn - g)

    savetif(le, outputPath, band_path)
    return le
######################################################################

def soilGFlux_ssebi(LST_C, albedo, ndvi, Rn, outputPath, reference_path):
    """
    # Soil/Ground Heat Flux [W/m2]
    # via BASTIAANSSEN, 2000 (BASTIAANSSEN, W. G. M. SEBAL - based sensible and latent heat fluxes in the irrigated Gediz Basin, Turkey. Journal of Hydrology, v.229, p.87-100, 2000.)

    LST = Land Surface Temperature [°C]
    albedo = Albedo [-]
    ndvi = Normal Differential Vegetation Index [-]
    Rn - Net ENergy Budget [W/m-2]
    """

    G = LST_C / albedo * (0.0038 * albedo + 0.0074 * albedo ** 2) * (1 - 0.98 * ndvi ** 4) * Rn
    

    G = np.where(ndvi < 0, Rn * 0.5, G)  #assume water
    G = np.where((LST_C < 4) & (albedo > 0.45), Rn * 0.5, G) #assume snow

    

    savetif(G, outputPath, reference_path)
    return G

######################################################################

def h_ssebi(ef, rn, g, outputPath, band_path):
       
    h = (1 - ef) * (rn - g)

    savetif(h, outputPath, band_path)

    return h

def et_a_day_ssebi(le, rn, lambda_v, output_path, reference_img):
    #&eta =  (((rn - g) * ef)) #/ lambda_v) * 86400
    #eta = (le * 86400) / lambda_v

    eta = le * (rn * 86400) / (lambda_v * rn)
    savetif(eta, output_path, reference_img)
    return eta

#########################################################
################### SSeBop functions ###################
#########################################################

# https://hess.copernicus.org/preprints/11/723/2014/hessd-11-723-2014.pdf
def etf_ssebop(lst, th, tc, output_path, reference_img, ):
    #th = np.nanmax(lst)

    etf = (th - lst) / (th - tc)

    savetif(etf, output_path, reference_img)
    return etf


def le_ssebop(ef, rn, g, outputPath, band_path):

    le = ef * (rn - g)

    savetif(le, outputPath, band_path)
    return le

def h_ssebop(ef, rn, g, outputPath, band_path):
       
    h = (1 - ef) * (rn - g)

    savetif(h, outputPath, band_path)

    return h

def et_a_day_sssebop(le, rn, lambda_v, output_path, reference_img):
    #&eta =  (((rn - g) * ef)) #/ lambda_v) * 86400
    #eta = (le * 86400) / lambda_v

    eta = le * (rn * 86400) / (lambda_v * rn)
    savetif(eta, output_path, reference_img)
    return eta



# https://hess.copernicus.org/preprints/11/723/2014/hessd-11-723-2014.pdf
def eta_ssebop(etf, k, et0,outputpath, reference_img):
    eta_ssebop = (etf *  et0)

    savetif(eta_ssebop, outputpath, reference_img)
    return eta_ssebop


def identify_cold_hot_pixels(lst, ndvi, albedo):
   
    # Cold pixel: High NDVI, Low LST, Low Albedo
    try:
        cold_mask = ndvi > 0.5 \
                    & (lst < np.nanpercentile(lst, 10)) \
                    & (albedo < 0.2)
        
        cold_pixels = np.where(cold_mask)

    except:
        
        cold_mask = (ndvi > np.nanpercentile(ndvi, 90)) \
                    & (lst < np.nanpercentile(lst, 10)) \
                    & (albedo < 0.2)
        
        cold_pixels = np.where(cold_mask)
    
    
    # Hot pixel: Low NDVI, High LST, High Albedo
    hot_mask = (ndvi < 0.2) \
                    & (lst > np.nanpercentile(lst, 90)) \
                    & (albedo > 0.3)
    
    hot_pixels = np.where(hot_mask)
    
    # Extract cold and hot pixel values
    cold_pixel_values = lst[cold_pixels]
    hot_pixel_values = lst[hot_pixels]
    
    # Calculate representative cold and hot temperature thresholds
    t_cold = np.nanmean(cold_pixel_values)
    t_hot = np.nanmean(hot_pixel_values)
    
    return t_cold, t_hot

######################################################################
######################################################################
######################################################################
######################################################################

def ETI(Kc, ET0, outputPath, reference_img):
       #nanmax -> ignore nan values
    ETI = (Kc * ET0) /  np.nanmax(ET0)
    ETI = np.where(ETI > 1, 1, ETI)
    savetif(ETI, outputPath, reference_img)
    return ETI
######################################################################

def CCi(albedo, ETI, hillshade, outputPath, band_path):
    shade = np.array(tf.imread(hillshade))
    #ETI_array = np.array(tf.imread(eti))
    hillshade = np.array(tf.imread(hillshade))
    normalization_hillshade = hillshade / 255
    
    CCi = (0.6 * normalization_hillshade) + (0.2 * albedo) + (0.2 * ETI)   
    CCi[CCi < 0] = np.nanmean(CCi)
    savetif(CCi, outputPath, band_path)
    return CCi

######################################################################

# https://github.com/natcap/invest.users-guide/raw/main/data-sources/kc_calculator.xlsx - from invest 
def Kc_LAI(LAI, outputPath, reference_img):
    np.seterr(all = "ignore")
    
    Kc_LAI = 1.1 * (1- np.exp(-1.5*(LAI)))

    savetif(Kc_LAI, outputPath, reference_img)
    return Kc_LAI

############################################################################################################################################
#   VEGETATION
############################################################################################################################################
def msavi(green, red, outputPath):
    """
    # Modified Soil Adjusted Vegetation Index
    # via http://www.jbrom.smoothcollie.eu/?page_id=147
    
    green = green landsat band
    red = red landsat band

    """
    green = np.array(tf.imread(green))
    red = np.array(tf.imread(red))

    msavi = 0.5 * ((2 * red + 1) - ((2 * red + 1) ** 2.0 - 8*(red - green)) ** 0.5)
    msavi[msavi == np.inf] = 0               	
    msavi[msavi == -np.inf] = 0

    #savetif(msavi, outputPath)
    
    return msavi

############################################################################################################################################

def ndvi(nir_path, red_path, outputPath):
    """
    # Normalized Differential Vegetation Index
    # NDVI = (NIR - RED) / (NIR + RED)
    # Unitless
    # Range from -1 to +1
    """
    nir = np.array(tf.imread(nir_path))
    red = np.array(tf.imread(red_path))
    zero_except = np.seterr(all = "ignore")
    NDVI = (nir - red) / (nir + red)
    NDVI[NDVI > 1] = np.mean(NDVI)
    NDVI[NDVI < -1] = np.mean(NDVI)


    savetif(NDVI, outputPath, nir_path)

    return NDVI

############################################################################################################################################

def savi(red, nir, outputPath, reference_img):
    nir = np.array(tf.imread(nir))
    red = np.array(tf.imread(red))

    #SAVI = ((nir - red) / (nir + red + 0.5)) * (1 + 0.5)
    SAVI =((1 + 0.5)*(nir-red)) / (nir + red + 0.5)
    SAVI[SAVI > 0.68] = 0.68
    
    savetif(SAVI, outputPath, reference_img)
    return SAVI

# https://www.sciencedirect.com/science/article/pii/S1470160X22000243
def lai(ndvi, outputPath, reference_img):
    
    LAI = (ndvi - np.nanmin(ndvi)) / 0.6
    
    savetif(LAI, outputPath, reference_img)
    return LAI

############################################################################################################################################

######################################################################

# BROM sebcs
def	vegHeight(h_min, h_max, msavi, outputPath):
    """
    Heigth of vegetation cover (m) derived from MSAVI index according to Gao et al. (2011).

    # via http://www.jbrom.smoothcollie.eu/?page_id=147
    """
    #msavi = np.array(tf.imread(msavi))
    minmsavi = np.min(msavi)
    maxmsavi = np.max(msavi)

    h_eff = h_min + (msavi - minmsavi) / (minmsavi - maxmsavi) * (h_min - h_max)
    
    h_eff[h_eff < h_min] = h_min
    h_eff[h_eff > h_max] = h_max
    
    #savetif(h_eff, outputPath)
    return h_eff

######################################################################
#vegetation fraction (proportion of vegetation)
def pv(NDVI, outputPath, referencePath):
    """
    # Fraction of Vegetation
    # via https://www.scirp.org/journal/paperinformation?paperid=112476#ref40

    NDVI = Normal Differential Vegetation Index
    """
    zero_except = np.seterr(all = "ignore") 
    
    PV = ((NDVI - np.nanmin(NDVI)) / (np.nanmax(NDVI) - np.nanmin(NDVI))) ** 2
    
 
    """PV = np.divide(np.power(NDVI, 2), 0.3)
       
    PV[PV > 1] = 0.99
    PV[PV < 0] = 0.1"""

    savetif(PV, outputPath, referencePath)

    return PV

######################################################################

def evapoFraction(Ta_max, Ta, LST, reference_band, outputPath):
    """
    # Fraction of Evapotranspiration
    # via https://www.sciencedirect.com/science/article/pii/S0309170811000145?via%3Dihub

    Ta_max = Maximum Air Temperature [˚C]
    Ta = Air Temperature [˚C]
    LST = Land Surface Temperature [˚C]
    outputPath = path to output directory
    """
    
    EF = (Ta_max/LST) / (Ta_max / Ta)
    savetif(EF, outputPath, reference_band)
    return EF

######################################################################

def tpw(Z):
    """
    # Total Precipitable Water [kg/m2]
    # via https://www.scielo.br/j/rbeaa/a/sX6cJjNXWMfHQ5p4h33B8Zz/?lang=en&format=pdf

    RU = Relative Humidity [%]
    P = Atmospheric Pressure [kPa]  
    """

    tpw = 0.75 +(0.00002 * Z)
    return tpw


def albedoLiang(b2, b4, b5, b6, b7, reference_band_path, outputPath):
    #b2 = np.array(tf.imread(b2_path))
    #b4 = np.array(tf.imread(b4_path))
    #b5 = np.array(tf.imread(b5_path))
    #b6 = np.array(tf.imread(b6_path))
    #b7 = np.array(tf.imread(b7_path))

    
    a = (((0.356 * b2) + (0.130 * b4) + (0.373 * b5) + (0.085 * b6) + (0.072 * b7)) - 0.018) / 1.016
    
    savetif(a, outputPath, reference_band_path)   
    return a