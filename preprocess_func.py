import os
import json
import fiona
import rasterio as rio
import rasterio.mask
from pprint import pprint 
from osgeo import gdal
from affine import Affine


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


def get_qa_filepath(input_folder, suffix):
    
    """
    # Get a filtered list of paths to files containing a certain suffix 

    input_folder = folder contaning the input files
    suffix = file suffix to be filtered (eg. .TIF, .JSON)
    """

    pathListFolder = []
    for root, dirs, files in os.walk(input_folder, topdown=False):
        for name in files:
            if 'QA_PIXEL' in name and name.endswith(suffix):
                if name not in pathListFolder:
                    pathListFolder.append(os.path.join(root, name))
    return pathListFolder

######################################################################


"""
Get unique values of pixels in QA_BAND and their frequency, along with their area. 
Use it to differenciate between sensing days ith too many clouds.

pixel_dict = {sensingdate : {pixel_value : number of occurence}}
"""  


def create_qa_dict(path_to_QA):
    for file in path_to_QA:
        dataset = gdal.Open(file)
        band = dataset.GetRasterBand(1)
        #no_data_value = band.GetNoDataValue()  # gather No Data
        get_data = band.ReadAsArray()   # read as array
        
        # get pixel resolution
        gt = dataset.GetGeoTransform()
        pixelSizeX = gt[1]
        pixelSizeY =-gt[5]

        
        pixel_counts = {}
        for value in get_data.flat: # flatten the matrix 
            try:
                pixel_counts[value] = pixel_counts.get(value, 0) + 1 # value - current pixel, 
                pixel_dict = {file.split('_')[5] : pixel_counts} 
            except KeyError: 
                continue
        #pprint(pixel_dict)
        dataset = None
    return pixel_dict

######################################################################
def load_json(jsonFile):
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
        #pprint(f"mask: {maskpath}")
        #pprint(f"original image: {inputBand}")
        #pprint(f"cropped image: {outImgPath}")
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
        nodata = -9999, # optinal value for nodata
        dtype = img_new.dtype,
        crs = epsg, # driver for coordinate system code
        transform=afn)
    new_dataset.write(img_new, 1)
    new_dataset.close()
    return

def cloud_pres(bin_pix_val):
            ## take 4th bit (position -5) and get it in new column (0-no cloud, 1-cloud)
    return [binary_value[-5] if len(binary_value) == 16 else None   
            for binary_value in bin_pix_val]

def calc_area_qa_pixels_m2(area_one_pixel, pixel_frequency):
    return area_one_pixel * pixel_frequency

def calc_area_qa_pixels_percent(width_img, height_img, pixel_frequency):
    return round((pixel_frequency * 100) / (width_img * height_img),2)

def filter_df_cloud_pixels(df, cloud_coverage_percent, cloud_pixels_list):
    return df.loc[(df['pixel_value'].isin(cloud_pixels_list)) & (df['pixel_area_%'] > cloud_coverage_percent)]

def export_df_cloud_csv(csv_name, df, ):
    csv_filename = f"qa_stats_{csv_name}.csv"
    df.to_csv(csv_filename, index = True, header=['pixel_value', 'pixel_frequency', 'sensing_date', 'pixel_area_m2', 'pixel_area_%'])
