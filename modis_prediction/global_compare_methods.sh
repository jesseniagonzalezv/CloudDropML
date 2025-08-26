#!/bin/bash

# Set environment
# source ~/.bashrc
module load pytorch 



# -----------------------Configuration--------------------------------
output_results="/scratch/b/b381362/"
num_channels="18_chan"

# all_data_lwp="all_features"
all_data_lwp="most_importances"   #probar con esteeee!!!!
main_folder_ml_lwp="/work/bb1143/b381362/"

all_data_nd="most_importances" 
main_folder_ml_nd="/work/bb1143/b381362/used_thesis_retrained_model_nd_lwp" # check function general asignation nd
# main_folder_ml_nd="/work/bb1036/b381362/"  # check function general asignation nd

fold_num=0


# result_folder="$output_results/1000epochs_500_011_retraining_last_plots_60N_60S/global_${all_data}_bb1143_both_fold${fold_num}"    # this is to check VF

result_folder="$output_results/results_lwp_more_data_${all_data_lwp}_nd_retrained_${all_data_nd}/global_${all_data_lwp}_lwp_${all_data_nd}_nd_bb1143_both_fold${fold_num}"  #i


folder_path_modis_data="/work/bb1143/b381362/MODIS_data_2may/"  # 2 of may
# folder_path_modis_data="//work/bb1143/b381362/MODIS_data_9days/"

echo "compare methods to predict Nd LWP"

mkdir -p "$result_folder/log"

# ====================================Global data==========================   

# -----------------------------------
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ml_pre_re 
#SBATCH --partition=compute
#SBATCH --account=bb1036
##SBATCH --account=bb1143
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8 
#SBATCH --array=0-128   #number of files - 1
##SBATCH --array=129-600   #number of files - 1
##SBATCH --array=600-1007   #number of files - 1
##SBATCH --array=0   #number of files - 1
## SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=01:00:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications
#SBATCH --output="${result_folder}/log/log_fold${fold_num}_%j.txt"

# -------------------------------- predict -> reproject 
python scripts/global_compare_modis_ml_lwp_nd.py --main_folder_ml_lwp $main_folder_ml_lwp \
                                         --main_folder_ml_nd $main_folder_ml_nd \
                                         --result_folder $result_folder \
                                         --folder_path_modis_data $folder_path_modis_data \
                                         --all_data_lwp $all_data_lwp \
                                         --all_data_nd $all_data_nd \
                                         --fold_num $fold_num \
                                         --num_channels $num_channels
                                         

                                         
EOF
