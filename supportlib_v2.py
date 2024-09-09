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
from osgeo import gdal, osr
import math
import json

import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import stats
import warnings



############################################################################################################################################
#   GENERAL FUNCTIONS
############################################################################################################################################

def getfilepath(input_folder, suffix):
    
    """
    # Get a filtered list of paths to files containing a certain suffix 

    input_folder = folder contaning the input files
    suffix = file suffix to be filtered (eg. .TIF, .JSON)
    """

    pathListFolder = []
    for root, dirs, files in os.walk(input_folder, topdown=False):
        for name in files:
            if name.endswith(suffix):
                if name not in pathListFolder:
                    pathListFolder.append(os.path.join(root, name))
    return pathListFolder


def get_band_filepath(input_folder, suffix):
    
    """
    # Get a filtered list of paths to files containing a certain suffix 

    input_folder = folder contaning the input files
    suffix = file suffix to be filtered (eg. .TIF, .JSON)
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

    """
    # Create a dictionary of meteorology data
    # eg. {'date': [avg_temp': '22.60','max_temp': '28.4','min_temp': '13.3','relHum': '70.21','wind_sp': '0.83']}

    csv_file = file in CSV format containing meterological data with date in original format (YYYYMMDD)
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

def load_json(jsonFile):
    with open(jsonFile, 'r') as j:
        data = json.load(j)
    return data

######################################################################

def clipimage(maskPath, inputBand, outImgPath, cropping = True, filling = True, inversion = False):

    """
    # Image clipping using a pre-prepared GeoPackage mask in the same coordinate system as the images

    maskPath = path to polygon mask
    inputBand = image to be clipped
    outImgPath = path to new cropped image

    """

    with fiona.open(maskPath, "r") as gpkg:
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
        pprint(f"maska: {maskPath}")
        pprint(f"vstupni data: {inputBand}")
        pprint(f"vystupni cesta: {outImgPath}")
        dest.write(out_image)
    
   
    return 

######################################################################

def savetif(img_new, outputPath, image_georef, epsg = 'EPSG:32633'):
    """ Save created image in chosen format to chosen path.

    Args:
        image_georef (string):      path to sample georeferenced image with parameters to be copied to new image
        img_new (numpy.array):      newly created image to be saved
        outputPath (str):           path including new file name to location for saving
        epsg (str):                 EPSG code for SRS (e.g. 'EPSG:32632')
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



def scatterplot_2d(x, y, var1, var2, pathout):
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
    return


def W_to_MJday(variable):
    return (variable * 0.0864)
############################################################################################################################################
#   ALBEDO                                                                                                                                 #
# via https://www.scielo.br/j/rbeaa/a/sX6cJjNXWMfHQ5p4h33B8Zz/?lang=en&format=pdf                                                          #
############################################################################################################################################
                                                                                                                                    
def dr(sunDist):

    """
    # Correction of the eccentricity of the terrestrial orbit

    sunDist = distance Earth-Sun in AU extracted from image metadata (EARTH_SUN_DISTANCE)
    """

    dr_num = 1 / (sunDist ** 2)
     
    return dr_num

############################################################################################################################################

def zenithAngle(sunElev):

    """
    # Calculation of sun zenith angle in radians

    sunElev = Sun Elevation angle extracted from image metadata (SUN_ELEVATION)
    """
    return ((90 - sunElev))# * math.pi) / 180

############################################################################################################################################

def lb_band(offset, gain, band_path, outputPath):
   
    """
    # Pixel Radiance

    AddRad = Additive Radiance Term from the Metadata (RADIANCE_ADD_BAND_X)
    MultRef =  Radiance Multiplicative Term from the Metadata (RADIANCE_MULT_BAND_X)
    band = Landsat band
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

"""def kb(rb, lb, dr, zenithAngle, outputPath, band_path):
    
    # Solar Constant [W/m2]

    rb = Band reflectance
    lb = Radiance of each pixel [W/m2]   
    dr = correction of the eccentricity of the terrestrial orbit

    
    
    #rb_band = np.array(tf.imread(rb))
    #lb_band = np.array(tf.imread(lb))
    
    kb = (math.pi * lb) / (rb * math.cos(math.radians(zenithAngle)) * dr)
    kb[kb < 0] = 0

    savetif(kb, outputPath, band_path) 
    return kb"""

######################################################################

def pb(esun_band, esun_sum):
    return esun_band / esun_sum


def albedo_toa_band(pbx,rbx):
    
    """
    # pPanetary albedo (without atmospheric correction)
    # Unitless or %
    # Range from 0 to 1 (ot 0 % to 100 %)

    toaplanet = Planetary Top Of Atmosphere Radiance  
    pbx = weight of spectral band
    rbx = and reflectance
    """

    toaPlanet = (pbx * rbx)   
    return toaPlanet


######################################################################

def albedo(toaplanet, atm_trans, outputPath, band_path):
    """
    # Albedo 
    # Unitless or %
    # Range from 0 to 1 (ot 0 % to 100 %)

    toaplanet = Planetary Top Of Atmosphere Radiance  
    Toc = Atmospheric transmittance in the solar radiation domain
    """
    albedo = (toaplanet - 0.03) / (atm_trans ** 2)
    albedo = np.where(albedo < 0, np.nan, albedo)
    
    savetif(albedo, outputPath, band_path)
    return albedo


############################################################################################################################################
#   ATMOSHEPRIC FUNCTIONS
############################################################################################################################################
def e0(T):
    """
    # Partial Water Vapour Pressure [kPa] /  actual vapour pressure
    # It is a measure of the tendency of a substance to evaporate or transition from its condensed phase to the vapor phase.
    # Tetens Formula via https://www.omnicalculator.com/chemistry/vapour-pressure-of-water

    T = Air Temperature [˚C]
    """
    e0 = 0.6108 * math.exp(((17.27 * T) / (T + 237.3)))
    return e0

######################################################################

def es(Ta):
    """
    # Saturated Vapour Pressure [kPa]
    # via https://www.fao.org/3/x0490e/x0490e07.htm#atmospheric%20pressure%20(p)
    
    Ta = Average Air Temperature (˚C)
    """
    
    es = (6.1078 * 10**(7.5 * Ta /(Ta + 237.3))) / 10
    return es

######################################################################

def atmPress(Z):
    """
    # Atmospheric Pressure [kPa]
    # via https://designbuilder.co.uk/helpv3.4/Content/Calculation_of_Air_Density.htm

    Z = Elevation above sea level [m]
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
    #ET0_daily = ((0.408 * SatVapCurve * (Rn - G) + psych * (900 / (Ta + 273.15)) 
                  #* WindSp * VPD) / (SatVapCurve + psych * (1 + 0.34 * WindSp))) 
    
    DT = SatVapCurve / (SatVapCurve + psych * 1 + 0.34 * WindSp)
    PT = psych / (SatVapCurve + psych * (1 + 0.34*WindSp))
    TT = (900 / Ta + 273) * WindSp
    ET_wind = PT * TT * (es - e0)

    Rng = Rn * 0.408
    ET_rad = DT * Rng
    ET0_daily = (ET_wind + ET_rad) 
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

def emis(ndvi, Pv, output_path, reference_band_path): #surface emis
    """
    # Surface Emmisivity
    # via https://www.scirp.org/journal/paperinformation?paperid=112476#return61

    red = Landsat Red Band
    ndvi = Normal Differential Vegetation Index
    Pv = Fraction of Vegetation
    """
        
    E = np.where(ndvi < 0, 0.991,  # Water
                  np.where(ndvi > 0.5, 0.993,  # Dense vegetation
                           np.where(ndvi <= 0.2, 0.996,  # Bare soil
                                    (0.973 * Pv + (1 - Pv) *  0.973 + 0.005))))  # Sparse vegetation

    #savetif(E, output_path, reference_band_path)
    return E

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
def etf_ssebop(lst, Rn, rho, cp, output_path, reference_img, rah = 110):
    th = np.nanmax(lst)

    etf = (th - lst) / ((Rn * rah) / (rho * cp))

    savetif(etf, output_path, reference_img)
    return etf

# https://hess.copernicus.org/preprints/11/723/2014/hessd-11-723-2014.pdf
def eta_ssebop(etf, k, et0,outputpath, reference_img):
    eta_ssebop = (etf * (k * et0)) 

    savetif(eta_ssebop, outputpath, reference_img)
    return eta_ssebop


#########################################################
################### SEBAL functions ###################
#########################################################

def u_fric_vel_corr(u_bl_heigh, blend_height, momentum_z0m, psi_m_200, k = 0.41):
    u_ast = (k * u_bl_heigh) / (np.log(blend_height / momentum_z0m)) - psi_m_200
    return u_ast

def rah_corr(z1, z2, fric_vel_corr, psi_h_2, psi_h_0_1, k = 0.41):
    r_ah = (np.log(z2 / z1) )- psi_h_2 + psi_h_0_1 / (fric_vel_corr * k)
    return r_ah

def select_cold_pixel(albedo, LAI, LST):
    """Select the cold pixel based on albedo, LAI, and LST, or use average LST if none found."""
    cold_mask = (albedo >= 0.22) & (albedo <= 0.24) & (LAI >= 4) & (LAI <= 6)
    cold_candidates = np.where(cold_mask)

    if len(cold_candidates[0]) > 0:
        min_LST_index = np.argmin(LST[cold_candidates])
        cold_pixel_idx = (cold_candidates[0][min_LST_index], cold_candidates[1][min_LST_index])
    else:
        warnings.warn("No cold pixel found with the specified conditions. Using the pixel closest to average LST.")
        average_LST = np.nanmean(LST)
        cold_pixel_idx = np.unravel_index(np.argmin(np.abs(LST - average_LST)), LST.shape)

    return cold_pixel_idx

def select_hot_pixel(LAI, LST):
    """Select the hot pixel based on LAI and LST, or use average LST if none found."""
    hot_mask = (LAI >= 0) & (LAI <= 0.4)
    hot_candidates = np.where(hot_mask)

    if len(hot_candidates[0]) > 0:
        max_LST_index = np.argmax(LST[hot_candidates])
        hot_pixel_idx = (hot_candidates[0][max_LST_index], hot_candidates[1][max_LST_index])
    else:
        warnings.warn("No hot pixel found with the specified conditions. Using the pixel closest to average LST.")
        average_LST = np.nanmean(LST)
        hot_pixel_idx = np.unravel_index(np.argmin(np.abs(LST - average_LST)), LST.shape)

    return hot_pixel_idx

def calculate_dT_hot(net_radiation, g_flux, ra, hot_pixel):
    """Calculate dT for the hot pixel, using average ra if the hot pixel's ra value is NaN."""
    
    try:
        # Ensure ra is a numpy array
        ra = np.asarray(ra)
        
        # Attempt to access the ra value at the hot_pixel index
        ra_hot = ra[hot_pixel]
        
        # Check if ra at the hot_pixel index is NaN
        if np.isnan(ra_hot):
            # Calculate the average of ra over all valid (non-NaN) values
            ra_hot = np.nanmean(ra)
            warnings.warn("Warning: ra at the hot pixel is NaN. Using average ra value.")
    
    except IndexError:
        # If indexing fails, fall back to using the average of ra
        ra_hot = np.nanmean(ra)
        warnings.warn("IndexError: Using average ra value instead.")
    
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
    # Calculate dT_hot
    dT_hot = (net_radiation[hot_pixel] - g_flux[hot_pixel]) * ra_hot / (1.225 * 1004)
    
    return dT_hot

def calculate_dT_cold():
    """Calculate dT for the cold pixel."""
    # Assume dT_cold is 0 for well-watered conditions
    dT_cold = 0.0
    return dT_cold

def calculate_dt_image(dT_hot, dT_cold, lst, cold_pix, hot_pix):
    """Calculate the dT across the entire image using the hot and cold pixel values."""
    # Calculate coefficients a and b using the SEBAL model equations
    a = (dT_hot - dT_cold) / (lst[hot_pix] - lst[cold_pix])
    b = dT_hot - a * lst[hot_pix]
    
    # Apply the linear relationship dT = a * LST + b across the entire image
    dT = a * lst + b
    
    return dT

def calculate_MO(rho, cp, T, H, u_star, kappa=0.4, g=9.81,
                 lower_bound=-1e6, upper_bound=1e6):
    """Calculate the Monin-Obukhov length."""
    mo = -((u_star**3 * T) / (kappa * g * H * (rho / cp)))

    # Identify and replace extreme values
    extreme_mask = (mo < lower_bound) | (mo > upper_bound) | np.isnan(mo)
    if np.any(extreme_mask):
        # Calculate the mean of non-extreme, valid values
        mo_mean = np.nanmean(mo[~extreme_mask])
        # Replace extreme values with the mean
        mo = np.where(extreme_mask, mo_mean, mo)

    return mo

def calculate_x(factor, MO):
    x = (1 - 16 * (factor / MO)) ** 0.25
    return x
    
def stability_correction(x, MO, factor):
    """Calculate stability-corrected aerodynamic resistance."""
    
    # Initialize the output variables
    psi_m_200 = 0
    psi_h_200 = 0
    psi_h_2 = 0
    psi_h_0_1 = 0
    
    if np.any(MO < 0):
        # Unstable conditions
        if factor == 200:
            term1 = 2 * np.log((1 + x) / 2)
            term2 = np.log((1 + x**2) / 2)
            term3 = -2 * np.arctan(x)
            term4 = 0.5 * np.pi
            psi_m_200 = term1 + term2 + term3 + term4
            return psi_m_200
        if factor == 2:
            psi_h_2 = 2 * np.log((1 + x**2) / 2)  # Ensure term2 is defined
            return psi_h_2
        
    elif np.any(MO > 0):
        # Stable conditions
        if factor == 200 or factor == 2:
            psi_m_200 = -5 * (200 / MO)
            psi_h_2 = -5 * (2 / MO)
            return psi_m_200, psi_h_2
        elif factor == 0.1:
            psi_h_0_1 = -5 * (0.1 / MO)
            return psi_h_0_1
    
    else:
        # Neutral conditions, return zeros
        return psi_m_200, psi_m_200, psi_h_2, psi_h_0_1


def h_incorr_sebal(net_radiation, g_flux, LST, albedo, 
                          LAI, ra, rho, cp, u, z, z0, T, 
                          outputPath, reference_img):
    """Iterate to calculate H with Monin-Obukhov stability correction."""

    cold_pixel = select_cold_pixel(albedo, LAI, LST)
    hot_pixel = select_hot_pixel(LAI, LST)

    dT_cold = calculate_dT_cold()
    dT_hot = calculate_dT_hot(net_radiation, g_flux, ra, hot_pixel)

    H_prev = None
    H_current = np.zeros(LST.shape)

    tolerance = 0.01
    iteration_count = 0

    while H_prev is None or np.max(np.abs(H_current - H_prev)) > tolerance:
        iteration_count += 1
        H_prev = H_current.copy()

        dT = calculate_dt_image(dT_hot, dT_cold, LST, cold_pixel, hot_pixel)
        
        # Calculate the friction velocity
        u_star = u_fric_vel(u, z, z0)

        # Calculate Monin-Obukhov length
        L = calculate_MO(rho, cp, T, H_current, u_star)

        H_current = (rho * cp * dT)  / ra

        print(f"Iteration {iteration_count}: Max change in H = {np.max(np.abs(H_current - H_prev))}")

    # Save the final H when optimal
    savetif(H_current, outputPath, reference_img)

    return H_current

def le_sebal(net_radiantion, g_flux, h_flux, ouputpath, reference_img):
    le = net_radiantion - g_flux - h_flux
    savetif(le, ouputpath, reference_img)
    return le
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
    #CCi[CCi < 0] = 0.5556
    savetif(CCi, outputPath, band_path)
    return CCi

######################################################################

def Kc(red, nir, outputPath):
    red = np.array(tf.imread(red))
    nir = np.array(tf.imread(nir))

    np.seterr(all = "ignore")
    RVI = nir / red

    Kc = 1.1 * (1- np.exp(-1.5 * RVI))

    #savetif(Kc, outputPath)
    return Kc

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

############################################################################################################################################
# https://www.sciencedirect.com/science/article/pii/S1470160X22000243
def lai(ndvi, outputPath, reference_img):
    
    LAI = (ndvi - np.nanmin(ndvi)) / 0.6
    
    savetif(LAI, outputPath, reference_img)
    return LAI

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