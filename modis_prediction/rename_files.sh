#!/bin/bash


all_data="all_features"
# all_data="most_importances" 

fold_num=0
# fold_num=1


base_path="/scratch/b/b381362/more_data_last_plots_60N_60S/global_${all_data}_bb1143_both_fold${fold_num}/hist"

# df_level2_modis_2305_fold_0.nc
# /scratch/b/b381362/more_data_last_plots_60N_60S/global_all_features_bb1143_both_fold0/hist



for file in "$base_path"/df_level2_modis_*.nc; do
# for file in "$base_path"/df_level2_modis_2305_fold_0.nc; do
  filename=$(basename "$file")  # just the filename
  newname="df_level2_modis_A2010151_${filename#df_level2_modis_}"
  mv "$file" "$base_path/$newname"
done


for file in "$base_path"/df_level1_ml_predictions_*.nc; do
  filename=$(basename "$file")  # just the filename
  newname="df_level1_ml_predictions_A2010151_${filename#df_level1_ml_predictions_}"
  mv "$file" "$base_path/$newname"
done


# # change the day
# base_path="/scratch/b/b381362/more_data_last_plots_60N_60S/global_${all_data}_bb1143_both_fold${fold_num}/hist"

# for file in "$base_path"/df_level2_modis_A2010151_*.nc; do
#   newname="${file/A2010151/A2013122}"
#   mv "$file" "$newname"
# done


# for file in "$base_path"/df_level1_ml_predictions_*.nc; do
#   newname="${file/A2010151/A2013122}"
#   mv "$file" "$newname"
# done
