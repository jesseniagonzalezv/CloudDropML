#!/bin/bash

# Set environment
source ~/.bashrc
module load pytorch 


main_folder="/work/bb1143/test/version_final"

#============================================
# ------------ Option 1 ----------------------
# type_plot="model_metrics"  
# ------------ Option 2 ----------------------
type_plot="join_NN_feature_importance"
all_data="all_features"
# all_data="most_importances"
# ------------ Option 3 ----------------------
type_plot="heatmap_along_methods"
all_data="all_features"
# all_data="most_importances"
# ------------ Option 4 ----------------------
# type_plot="feature_metrics"


#============================================

if [ "$type_plot" = "model_metrics" ]; then
#==================================================================================After training all models with all features and most importance features==========================================
echo "validation metrics along models and all features vs most important features"
python plotting/plot_compare_model_metrics.py --main_folder_path $main_folder 


elif [ "$type_plot" = "join_NN_feature_importance" ]; then
# ==============================================================================After training all models and After obtaining the feature importances for NN for all features or most importance featutes ==========================================
    output_path="$main_folder/NN_feature_importance_${all_data}"
    echo "feature importances"
    python plotting/NN_join_heatmaps.py --output_path $output_path \
                                  --all_data $all_data



elif [ "$type_plot" = "feature_metrics" ]; then
# ==================feature reduction  plot===========================================
echo "feature reduction (number of features vs performance)"
path_results="$main_folder/feature_reduction"

for variable_target in "lwp" "Nd_max"; do
# for variable_target in  "lwp"; do
# for variable_target in  "Nd_max"; do
python plotting/plot_features_vs_metric.py --variable_target $variable_target \
                                  --path_results $path_results

done


elif [ "$type_plot" = "heatmap_along_methods" ]; then
# ===============After obtaining NN_join_heatmaps  ==========================================
# --------------- also it produce the feature importances for each model with the most important features
echo "join_heatmaps"
python plotting/join_heatmaps.py --output_path $main_folder \
                                 --all_data $all_data

fi
