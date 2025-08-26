#!/bin/bash

# Set environment
# source ~/.bashrc
module load pytorch 



# -----------------------Configuration--------------------------------
# output_results="/work/bb1143/b381362/"

# all_data="all_features"
# all_data="most_importances" 

num_channels="18_chan"

# fold_num=0
# fold_num=1

# for all_data in "all_features"; do
for all_data in "most_importances"; do
# for all_data in "all_features" "most_importances"; do
# for fold_num in 0 1; do
for fold_num in 0 ; do
# for fold_num in 1 ; do

# for all_data in "most_importances"; do
# for fold_num in 0; do

main_folder_ml_lwp="/work/bb1143/b381362/retrained_model_nd_lwp"
main_folder_ml_nd="/work/bb1143/b381362/retrained_model_nd_lwp"

base_output="/scratch/b/b381362/"


result_folder="$base_output/1000epochs_500_011_retraining_last_plots_60N_60S/only_germany_${all_data}_bb1143_both_fold${fold_num}"    # this is to check VF


echo "compare methods to predict Nd LWP"


# 500_radius_011_last_plots_60N_60S_only_germany_train   solo usanfdo germany sale mejor
# -----------------------------------
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=1140me_${fold_num} 
#SBATCH --partition=compute
##SBATCH --account=bb1036
#SBATCH --account=bb1143
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=01:00:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications
#SBATCH --output="${result_folder}/log_fold${fold_num}_%j.txt"

python scripts/11_40_methods_obtaine_lwp_nd.py --main_folder_ml_lwp $main_folder_ml_lwp \
                                         --main_folder_ml_nd $main_folder_ml_nd \
                                         --all_data $all_data \
                                         --output_results $result_folder \
                                         --num_channels $num_channels \
                                         --fold_num $fold_num

EOF
done
done
