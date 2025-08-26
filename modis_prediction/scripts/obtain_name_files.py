
import glob
import os
import re
import argparse


def obtain_name_files(cloud_dir, geoloc_dir, level_modis="level1"):
    # Define directories
    # cloud_dir = "/work/bb1143/b381362/MODIS_data/cloud/"
    # geoloc_dir = "/work/bb1143/b381362/MODIS_data/geoloc/"
    
    # Define file patterns
    if level_modis=="level1":
        cloud_pattern = os.path.join(cloud_dir, "MYD021KM.*.hdf")
        name_files_cloud = "matching_cloud_files_level1.txt"
        name_files_geol = "matching_geoloc_files_level1.txt"
            
    if level_modis=="level2":
        cloud_pattern = os.path.join(cloud_dir, "MYD06_L2.*.hdf")
        name_files_cloud = "matching_cloud_files_level2.txt"
        name_files_geol = "matching_geoloc_files_level2.txt"
    
    geoloc_pattern = os.path.join(geoloc_dir, "MYD03.*.hdf")
    
    # Obtain full file paths
    cloud_files = glob.glob(cloud_pattern)
    geoloc_files = glob.glob(geoloc_pattern)
    
    # Optionally sort the lists if order matters
    cloud_files.sort()
    geoloc_files.sort()
    
    # Regular expression to extract timestamp (e.g., "A2013122.1140")
    timestamp_pattern = re.compile(r"A[0-9]{7}\.[0-9]{4}")
    
    # Build dictionaries mapping timestamp to the filename (base name only)
    cloud_dict = {}
    for f in cloud_files:
        basename = os.path.basename(f)
        match = timestamp_pattern.search(basename)
        if match:
            timestamp = match.group(0)
            cloud_dict[timestamp] = basename
    
    geoloc_dict = {}
    for f in geoloc_files:
        basename = os.path.basename(f)
        match = timestamp_pattern.search(basename)
        if match:
            timestamp = match.group(0)
            geoloc_dict[timestamp] = basename
    
    # Find the common timestamps between cloud and geoloc files
    common_timestamps = set(cloud_dict.keys()).intersection(set(geoloc_dict.keys()))
    
    matching_cloud_files = []
    matching_geoloc_files = []
    
    # Sort the common timestamps to ensure matching order in both lists
    for ts in sorted(common_timestamps):
        matching_cloud_files.append(cloud_dict[ts])
        matching_geoloc_files.append(geoloc_dict[ts])
    
    # Write the matching filenames to text files
    
    # with open("matching_cloud_files_level1.txt", "w") as cf:
    with open(name_files_cloud, "w") as cf:
        for f in matching_cloud_files:
            cf.write(f + "\n")



    # with open("matching_geoloc_files_level1.txt", "w") as gf:
    with open(name_files_geol, "w") as gf:
        for f in matching_geoloc_files:
            gf.write(f + "\n")
    
    print(f"Matching cloud files written to {matching_cloud_files}")
    print(f"Matching geoloc files written to {name_files_geol}")


                                         
def main():
    parser = argparse.ArgumentParser(description='compare methods')
    arg = parser.add_argument
    arg('--cloud_dir', type=str, default="/work")
    arg('--geoloc_dir', type=str, default="/work")
    arg('--level_modis', type=str, default="level1")

    args = parser.parse_args()
    cloud_dir = args.cloud_dir
    geoloc_dir = args.geoloc_dir
    level_modis = args.level_modis
     
    obtain_name_files(cloud_dir, geoloc_dir, level_modis=level_modis)



if __name__ == '__main__':
    main()
