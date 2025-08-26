#!/bin/bash

# Set environment
source ~/.bashrc
module load ncl 
module load python3
# output="/work/bb1143/b381362/MODIS_data/"
output="/scratch/b/b381362/MODIS_data/"


# # =========================================================================================
# # --------------------------------------- STEP 1 ------------------------------------------
# # -----------------------------------------------------------------------------------------
# # =========================================================================================
# # cloud_dir="/work/bb1143/b381362/MODIS_data/cloud/"
# # geoloc_dir="/work/bb1143/b381362/MODIS_data/geoloc/"

# cloud_dir="/scratch/b/b381362/MODIS_data/cloud/"
# geoloc_dir="/scratch/b/b381362/MODIS_data/geoloc/"

# # level_modis="level1"

# for level_modis in "level1" "level2"; do
#     python scripts/obtain_name_files.py --cloud_dir $cloud_dir \
#                                   --geoloc_dir $geoloc_dir \
#                                   --level_modis $level_modis
                                    
# done




# =========================================================================================
# --------------------------------------- STEP 2 ------------------------------------------
# -----------------------------------------------------------------------------------------
# =========================================================================================
# to run  preprocess_modis_level1, 2 first run the obtain_files_names.py to get the matching_cloud_files_level{matching_geoloc_files_level2}.txt matching_geoloc_files_level{num_level},txt and also change the paths of diri1, diri2, output

# ncl preprocess_modis_level1.ncl

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=level1 
#SBATCH --partition=compute
#SBATCH --account=bb1036
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=2 
##SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=08:00:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications
#SBATCH --output="${output}/log_preprocessing_level1_%j.txt"

ncl preprocess_modis_level1.ncl

EOF

# ncl preprocess_modis_level2.ncl
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=level2 
#SBATCH --partition=compute
#SBATCH --account=bb1036
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=2 
## SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=08:00:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications
#SBATCH --output="${output}/log_preprocessing_level2_%j.txt"

ncl preprocess_modis_level2.ncl

EOF