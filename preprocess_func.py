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
    Retrieve a list of file paths containing a specified suffix (e.g., .TIF, .JSON).
    
    Args:
        input_folder (str):         Directory containing the input files.
        suffix (str):               File suffix to filter by.
    
    Returns:
        list:                       A list of file paths that end with the specified suffix.
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
    Retrieve a list of paths to QA_PIXEL files containing a specified suffix.

    Args:
        input_folder (str):         Directory containing the input files.
        suffix (str):               File suffix to filter by.
    
    Returns:
        list:                       A list of QA_PIXEL file paths that match the suffix.
    """

    pathListFolder = []
    for root, dirs, files in os.walk(input_folder, topdown=False):
        for name in files:
            if 'QA_PIXEL' in name and name.endswith(suffix):
                if name not in pathListFolder:
                    pathListFolder.append(os.path.join(root, name))
    return pathListFolder

######################################################################


def create_qa_dict(path_to_QA):

    """
    Generate a dictionary of pixel values and their frequencies from QA_PIXEL files.

    Args:
        path_to_QA (list):      List of file paths to QA_PIXEL files.
    
    Returns:
        dict:                   A dictionary mapping sensing dates to pixel value frequency data.
    """


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

    """
    Load a JSON file into a Python dictionary.

    Args:
        jsonFile (str):             Path to the JSON file.
    
    Returns:
        dict:                       Parsed JSON data as a dictionary.
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
        #pprint(f"mask: {maskpath}")
        #pprint(f"original image: {inputBand}")
        #pprint(f"cropped image: {outImgPath}")
        dest.write(out_image)
    
   
    return 
######################################################################

def savetif(img_new, outputPath, image_georef, epsg = 'EPSG:32633'):
    """
    Save a newly created image with georeferencing information.

    Args:
        img_new (numpy.array):          The new image to be saved.
        outputPath (str):               Path to save the new image.
        image_georef (str):             Path to a georeferenced image to copy georeferencing data from.
        epsg (str):                     EPSG code for spatial reference system (default 'EPSG:32633').
    
    Returns:
        None
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

def get_bit_index(bin_pix_val, ix):

    """
    Extract a specific bit from binary pixel values.

    Args:
        bin_pix_val (list):         List of binary pixel values as strings.
        ix (int):                   Index of the bit to extract (from the right).

    Returns:
        list:                       Extracted bits for each pixel value.
    """
     

    return [binary_value[ix] if len(binary_value) >= 16 else None 
            for binary_value in bin_pix_val]

def get_combined_index(bin_pix_val, ix1, ix2):

    """
    Combine two bits from binary pixel values.

    Args:
        bin_pix_val (list):             List of binary pixel values as strings.
        ix1 (int):                      Index of the first bit to extract.
        ix2 (int):                      Index of the second bit to extract.
    
    Returns:
        list:                           Combined bits for each pixel value.
    """

    return [binary_value[ix1] + binary_value[ix2] 
            if len(binary_value) >= 16 else None 
            for binary_value in bin_pix_val]

def calc_area_qa_pixels_m2(area_one_pixel, pixel_frequency):
    """
        Calculate the total area of pixels (in square meters).

        Args:
            area_one_pixel (float):     Area of one pixel in square meters.
            pixel_frequency (int):      Frequency of the pixel value.
        
        Returns:
            float:                      Total area covered by the pixel value.
        """

    return area_one_pixel * pixel_frequency

def calc_area_qa_pixels_percent(width_img, height_img, pixel_frequency):
    """
    Calculate the percentage of the image area covered by a pixel value.

    Args:
        width_img (int):                Width of the image in pixels.
        height_img (int):               Height of the image in pixels.
        pixel_frequency (int):          Frequency of the pixel value.
    
    Returns:
        float:                          Percentage area covered by the pixel value.
    """

    return round((pixel_frequency * 100) / (width_img * height_img),2)

def filter_df_cloud_pixels(df, cloud_coverage_percent, cloud_pixels_list):

    """
    Filter a DataFrame to include only cloud pixels above a given coverage percentage.

    Args:
        df (pandas.DataFrame):              DataFrame with pixel data.
        cloud_coverage_percent (float):     Minimum cloud coverage percentage.
        cloud_pixels_list (list):           List of pixel values representing clouds.
    
    Returns:
        pandas.DataFrame:                   Filtered DataFrame with cloud pixels.
    """

    return df.loc[(df['pixel_value'].isin(cloud_pixels_list)) & (df['pixel_area_%'] > cloud_coverage_percent)]

def export_df_cloud_csv(csv_name, df):
    """
    Append a DataFrame to a CSV file.

    Args:
        csv_name (str):                     Name of the output CSV file.
        df (pandas.DataFrame):              DataFrame to be exported.
    
    Returns:
        None
    """
    file_exists = os.path.exists(csv_name + '.csv')
    df.to_csv(csv_name + '.csv', mode='a', header=not file_exists , index=False)
