import os
import glob
import xarray as xr
import re



def read_files_modis(folder_path, type_modis):
    """
    Reads all NetCDF files in the given folder that of level 1 o level2  
    Args:
        folder_path (str): Path to the folder containing MYD06_L2 NetCDF files.
        
    Returns:
        xarray.Dataset: Concatenated dataset along the time dimension.
    """
    # Find all MYD06_L2 NetCDF files
    if type_modis=="level_1":
        file_pattern = os.path.join(folder_path, '*MYD021*.nc')
    elif type_modis=="level_2":
        file_pattern = os.path.join(folder_path, '*MYD06_L2*.nc')
    files = sorted(glob.glob(file_pattern))

    if not files:
        raise FileNotFoundError("No NetCDF files found in the folder.")


    return files


def extract_key_from_filename(filename):
    """Extract the 'Ayyyydoy.hhmm' timestamp from MODIS filename."""
    match = re.search(r'A\d{7}\.\d{4}', filename)
    return match.group(0) if match else None


def match_level1_level2(files_level1, files_level2):
    level1_dict = {extract_key_from_filename(f): f for f in files_level1}
    level2_dict = {extract_key_from_filename(f): f for f in files_level2}

    all_keys = sorted(set(level1_dict.keys()) | set(level2_dict.keys()))

    matched = []
    missing = []

    for i, key in enumerate(all_keys):
        f1 = level1_dict.get(key)
        f2 = level2_dict.get(key)

        if f1 and f2:
            matched.append((i, key, f1, f2))
        else:
            missing.append((i, key, f1, f2))  # one of them is None

    return matched, missing


def obtain_files_match(folder_path_modis_data):
    # folder_path_modis_data = "/work/bb1143/b381362/MODIS_data/"
    type_modis = "level_2"
    files_level2 = read_files_modis(folder_path_modis_data, type_modis)
    
    type_modis = "level_1"
    files_level1 = read_files_modis(folder_path_modis_data, type_modis)
    
    print(len(files_level1), len(files_level2))
    
    matched, missing = match_level1_level2(files_level1, files_level2)

    # View matched 
    for i, key, f1, f2 in matched:
        print(f"✅ {key} → L1: {os.path.basename(f1)} | L2: {os.path.basename(f2)}")
    # View missing 
    for i, key, f1, f2 in missing:
        print(f"❌ Missing match for {key} → L1: {'Yes' if f1 else 'No'} | L2: {'Yes' if f2 else 'No'}")

    return matched





