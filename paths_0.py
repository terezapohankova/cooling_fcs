import os
import process_func
import img_prep_2
#from lst_4 import reference_img




INPUT_FOLDER = r'/home/tereza/Documents/data/LANDSAT/RESULTS'
#INPUT_FOLDER = input(f"Enter the path for the input directory: ").strip() 

INPUT_DATA = os.path.join(INPUT_FOLDER, 'clipped_bands')
JSON_MTL_PATH = process_func.getfilepath(INPUT_DATA, 'MTL.json') #['root/snimky_L9_testovaci/LC09_L2SP_190025_20220518_20220520_02_T1/LC09_L2SP_190025_20220518_20220520_02_T1_MTL.json']

AUX_DATA = r'aux_data'
METEOROLOGY = os.path.join(AUX_DATA, 'weather_2024_olomouc.csv') 
HILLSHADE = os.path.join(AUX_DATA, 'hillshade_olomouc_32633.tif')


FOLDERS = ['lst', 'vegIndices', 'radiation', 'preprocess', 'albedo', 'fluxes', 'et', 'bowen']

def generate_image_path(date, folder_key, suffix):
    # Ensure img_prep_2 is imported here inside the function to avoid circular imports
    reference_img = img_prep_2.imgDict[date]['B5_L2']['clipped_path']
    return os.path.join(
        INPUT_FOLDER,  # The root input folder
        FOLDERS[folder_key],  # Folder from paths0.FOLDERS
        os.path.basename(reference_img.replace('B5.TIF', suffix))  # Replace 'B5.TIF' with the suffix
    )

