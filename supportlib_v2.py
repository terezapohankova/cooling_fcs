import os
import fiona
import rasterio
import rasterio.mask
from pprint import pprint
import numpy as np
#from sympy import Le, ln
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


#import earthpy.spatial as es
"""import fiona

with fiona.open(r'/home/tereza/Documents/SNIMKY/cernovice_2022/cernovice_bb.gpkg') as layer:
    for feature in layer:
        pprint(feature['geometry'])"""



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
    
    """pprint(f"maska: {maskPath}")
    pprint(f"vstupni data: {inputBand}")
    pprint(f"vystupni cesta: {outImgPath}")

    command = f"gdalwarp -cutline {maskPath} -crop_to_cutline -dstnodata NoData {inputBand} {outImgPath}"
    os.system(command)"""

   
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

def kb(rb, lb, dr, zenithAngle, outputPath, band_path):
    """
    # Solar Constant [W/m2]

    rb = Band reflectance
    lb = Radiance of each pixel [W/m2]   
    dr = correction of the eccentricity of the terrestrial orbit

    """
    
    #rb_band = np.array(tf.imread(rb))
    #lb_band = np.array(tf.imread(lb))
    
    kb = (math.pi * lb) / (rb * math.cos(math.radians(zenithAngle)) * dr)
    kb[kb < 0] = 0

    savetif(kb, outputPath, band_path) 
    return kb

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

    #albedo[albedo < 0] = 0.1
    
    #albedo[albedo > 1] = 0.99

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

def es(Tamax, Tamin, Ta):
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
    y =  0.00065 * P
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
    p1 = 6.1078 * (10**(7.5 * Ta /(Ta + 237.3)))
    
    # the water vapor pressure in Pa
    pv = p1 * RH

    #pressure of dry air
    pd = P - pv
    air_density = (pd / (R * (Ta + 273.15))) + (pv / (461.495  * (Ta + 273.15)))
  
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

def ra(airDensity, LST, Ta, eo, es, psychro, Rn, reference_band, OutputPath, cp = 1013):
    
    # Aerodynamic Resistance
    # via https://www.posmet.ufv.br/wp-content/uploads/2016/09/MET-479-Waters-et-al-SEBAL.pdf

    """airDensity = Density of Air [kg/m3]
    LST = Land Surface Temperature [˚C]
    Ta = Air Temperature [˚C]
    eo = Partial Water Vapour Pressure [kPa]
    es = Saturated vapour pressure [kPa]
    psychro = Psychrometric constant [kPa/˚C]
    Rn = Net Energy Budget [W/m2]
    cp = Specific heat at constant pressure """

    
    ra = (airDensity * cp * ((LST - Ta) + ((eo - es) / psychro))) / Rn
    #savetif(ra, OutputPath, reference_band)
    return ra 

"""def ra_2(h, u, reference_band, outputPath, zh = 2, zm = 2 ,k = 0.41):

    
    h = average crop height [m]
    d = effective crop height [m]
    zom = the roughness length governing momentum transfer [m]
    zoh = length governing transfer of heat and vapour [m]
    u = wind speed [m/s]
    zm, zh -> # standardized height for wind speed, temperature and humidity
    
    #h = np.array(tf.imread(h))
    d = (2/3) * h
    zom = 0.123 * h
    zoh = 0.1 * zom
    ra_1 = np.log((zm - d)/ zom) * np.log((zh - d)/ zoh)
    ra_2 = (k **2) * u

    ra = ra_1 / ra_2

    savetif(ra, outputPath, reference_band)
    return ra"""

def z0m(vegHeight, outputPath):

    """
    # Roughness length governing momentum transfer [m]

    vegHeigth = height of vegetation
    """

    
    z0m = 0.123 * vegHeight

    #savetif(z0m, outputPath, vegHeight)
    return z0m

def u_fric_vel_measure(wind_sp, measure_height, momentum_z0m ,k = 0.41):  
    u = (k * wind_sp) / np.log(measure_height / momentum_z0m)
    return u 

def wind_speed_blending(blending_height, momentum_z0m, friction_vel_measure, k = 0.41):
    u = friction_vel_measure * ((np.log(blending_height/momentum_z0m)) / k)
    return u

def u_fric_vel_pix(blend_height, wind_speed_blend, momentum_z0m, k = 0.41):

    u = (k * wind_speed_blend) / (np.log(blend_height / momentum_z0m))
    return u
    
def rah(z1, z2, fric_vel_pix, k = 0.41):
    r_ah = (np.log(z2 / z1) ) / (fric_vel_pix / k)
    return r_ah
######################################################################
def slopeVapPress(Ta):
    slopeVapPress = round((4098 * 0.6108 * 2.7183 ** ((17.27 * Ta)/(Ta+237.3)) / (Ta + 237.3) ** 2),3)
    return slopeVapPress


############################################################################################################################################
#   FLUXES
############################################################################################################################################

def soilGFlux(LST_C, albedo, ndvi, Rn, outputPath, reference_path):
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

def sensHFlux(airDens, LST_K, ra, Ta_K, outputPath, reference_img, cp = 0.001013):
    
    # Sensible Heat Flux
    # via

    """LST = Land Surface Temperature [˚C]
    ra = Air Resistence [s/m]
    Ta = Air Temperature [˚C]
    cp = Specific heat at constant pressure [MJ/kg/°C]
    airDens = Density of Air [kg/m3]
    LST_K = Land Surface Temperature [K]
    Ta_K = Air Temperature [K]"""
    
    

    H = (airDens * cp * (LST_K - Ta_K)) / ra
    savetif(H, outputPath, reference_img)
    return H

def H(LE, Rn, G, outputPath, band_path):
    # gradient method

    #LE = np.array(tf.imread(LE))
    #Rn = np.array(tf.imread(Rn))
    #G = np.array(tf.imread(G))

    h = Rn - G - LE
    savetif(h, outputPath, band_path)

    return h

######################################################################

def tr(air_temp_k):
    return 1 - 373.15 / air_temp_k

def e_aster(Po, Tr):
    return Po * np.exp((13.3185*Tr) - (1.976*(Tr**2)) - (0.6445*(Tr**3)) - (0.1299*(Tr**4)))

def delta_pt(e_ast, air_temp_k, Tr):
    return ((373.15 * e_ast) / (air_temp_k**2)) * (13.3185 - (3.952*Tr) - (1.9335*(Tr**2)) - (0.5196*(Tr**3)))


def le(phi, Rn, G, delta_pt, psychro, outputPath, band_path):
    
    # Latent HEat Flux [W/m2]
    # via Baasriansen, 2000 (BASTIAANSSEN, W. G. M. SEBAL - based sensible and latent heat fluxes in the irrigated Gediz Basin, Turkey. Journal of Hydrology, v.229, p.87-100, 2000.)

    LE = phi * (Rn - G * (delta_pt / (delta_pt + psychro)))
    savetif(LE, outputPath, band_path)
    return LE

######################################################################

def ET0(e0, SatVapCurve, WindSp, es, G, psych, Rn, Ta, outputPath, band_path):
    VPD = e0-es
    ET0_daily = ((0.408 * SatVapCurve * (Rn - G) + psych * (900 / (Ta + 273.15)) 
                 * WindSp * VPD)/(SatVapCurve + psych * (1 + 0.34 * WindSp))) 
    
    savetif(ET0_daily, outputPath, band_path)
    return ET0_daily


def ef(lst, ndvi, output_path, refernce_img):

    ndvi = np.where(ndvi > 0.05, ndvi, np.nan)
    Thot = np.nanmin(ndvi) + np.nanmax(lst)
    Tcold = np.nanmax(ndvi) + np.nanmin(lst)
    evapofract = (Thot - lst) / (Thot - Tcold)

    evapofract = np.where(evapofract == np.nan, np.nanmean(evapofract), evapofract )

    savetif(evapofract, output_path, refernce_img)

    return evapofract



def phi_max(ndvi, lst, num_intervals, reference_band, outputPath, phi_min_global=0):
    
    np.seterr(all="ignore")
    global_phi_max = np.nanmax(ndvi)
    global_phi_max_values = np.zeros_like(ndvi)

    # Segment NDVI into intervals and determine \phi_{\max} for each interval
    ndvi_intervals = np.linspace(np.nanmin(ndvi), np.nanmax(ndvi), num=num_intervals)

    for i in range(len(ndvi_intervals) - 1):
        # Create a mask to identify pixels within the current NDVI interval
        mask = (ndvi >= ndvi_intervals[i]) & (ndvi < ndvi_intervals[i + 1])

        if np.any(mask):
            # Find the pixel with the lowest temperature within this NDVI interval
            masked_lst = lst[mask]
            min_index = np.argmin(masked_lst)
            lowest_temp_pixel = np.where(mask)[0][min_index], np.where(mask)[1][min_index]

            interval_phi_max = ndvi[lowest_temp_pixel]
            global_phi_max_values[mask] = interval_phi_max
        else:
            # Handle case where no pixels in the interval
            print(f"No pixels in NDVI interval {i}")  # Or handle differently

    # Interpolate \phi_{\max} for each pixel
    phi_max = phi_min_global + (ndvi / global_phi_max) * (global_phi_max_values - phi_min_global)
    #global_phi[np.isnan(global_phi)] = 0
    
    if np.any(phi_max):
        pprint('phi_max')
        savetif(phi_max, r'/home/tereza/Documents/data/LANDSAT/RESULTS/vegIndices/phi_max.TIF', 
                r'/home/tereza/Documents/data/LANDSAT/RESULTS/vegIndices/clipped_LC09_L2SP_190025_20230926_20230928_02_T1_SR_ndvi.TIF')
    else:
        pprint('None')
    return phi_max

import numpy as np

def phi_min(ndvi, phi_min_global, num_intervals, phi_max):
    # Determine NDVI interval boundaries
    ndvi_max = np.nanmax(ndvi)
    
    # Segment NDVI into intervals and determine \phi_{\max} for each interval
    ndvi_intervals = np.linspace(np.nanmin(ndvi), np.nanmax(ndvi), num=num_intervals)
    
    for i in range(len(ndvi_intervals) - 1):
        # Create a mask to identify pixels within the current NDVI interval
        mask = (ndvi >= ndvi_intervals[i]) & (ndvi < ndvi_intervals[i + 1])
        if np.any(mask):

            phi_min_intervals = phi_min_global + (ndvi / ndvi_max) * (phi_max - phi_min_global)
    
    if np.any(phi_min_intervals):
        savetif(phi_min_intervals, r'/home/tereza/Documents/data/LANDSAT/RESULTS/vegIndices/phi_min_intervals.TIF', 
                r'/home/tereza/Documents/data/LANDSAT/RESULTS/vegIndices/clipped_LC09_L2SP_190025_20230926_20230928_02_T1_SR_ndvi.TIF')
    else:
        pprint('None')
  
    return phi_min_intervals

def interpolate_phi(lst, ndvi, phi_min, phi_max):
    # Normalize LST and NDVI (adjust normalization method as needed)
    lst_norm = (lst - np.nanmin(lst)) / (np.nanmax(lst) - np.nanmin(lst))
    
    ndvi_norm = (ndvi - np.nanmin(ndvi)) / (np.nanmax(ndvi) - np.nanmin(ndvi))
    pprint(np.nanmean(ndvi_norm))
    
    # Assign weights (adjust weights based on desired relationship)
    weight_lst = 1 - lst_norm
    weight_ndvi = ndvi_norm

    # Calculate weighted average
    phi = weight_lst * phi_min + weight_ndvi * phi_max

    if np.any(phi):
        savetif(phi, r'/home/tereza/Documents/data/LANDSAT/RESULTS/vegIndices/phi.TIF', 
                r'/home/tereza/Documents/data/LANDSAT/RESULTS/vegIndices/clipped_LC09_L2SP_190025_20230926_20230928_02_T1_SR_ndvi.TIF')
    else:
        pprint('None')
  

    return phi


def ea(et0, ef, ndvi, outputPath, reference_img):

    ef_correct =  (0.35*(ndvi/0.7)+0.65) * ef
    eactual = (et0 / 86400) * ef_correct
    
    
    
    savetif(eactual, outputPath, reference_img)
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

    pprint("cos")
    pprint((math.cos(math.radians(solar_zenith_angle))))
    
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
    LE = Latent Heat Flux [W/m2]
    """
    BI = H / LE
    BI[BI < 0] = np.nan
    BI[BI > 3] = np.nanmean(BI)
    savetif(BI, outputPath, band_path)
    return BI
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

def Kc_LAI(LAI, outputPath, reference_img):
    np.seterr(all = "ignore")
    Kc_LAI = (1 - np.exp(-0.7 * LAI))
    Kc_LAI = np.where(LAI <= 3, Kc_LAI, (LAI/3))
    
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

def lai(savi, outputPath, reference_img):
    LAI = np.where(savi > 0, np.log((0.61 - savi) / 0.51) / 0.91 * (-1), 0)
    LAI = np.where(savi >= 0.61, 1, LAI)
    
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