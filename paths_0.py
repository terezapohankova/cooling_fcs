import os
import process_func


INPUT_FOLDER = r'/home/tereza/Documents/data/LANDSAT/RESULTS'
#INPUT_FOLDER = input(f"Enter the path for the input directory: ").strip() 

INPUT_DATA = os.path.join(INPUT_FOLDER, 'clipped_bands')
JSON_MTL_PATH = process_func.getfilepath(INPUT_DATA, 'MTL.json') #['root/snimky_L9_testovaci/LC09_L2SP_190025_20220518_20220520_02_T1/LC09_L2SP_190025_20220518_20220520_02_T1_MTL.json']

AUX_DATA = r'aux_data'
METEOROLOGY = os.path.join(AUX_DATA, 'weather_2024_olomouc.csv') 
HILLSHADE = os.path.join(AUX_DATA, 'hillshade_olomouc_32633.tif')


FOLDERS = ['lst', 'vegIndices', 'radiation', 'preprocess', 'albedo', 'fluxes', 'et', 'bowen']
