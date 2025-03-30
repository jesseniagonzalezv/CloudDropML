#!/bin/bash

# Set environment
source ~/.bashrc
module load pytorch 

type_model="RF" 
# type_model="XGBoost" 


# variable_target="Nd_max"
variable_target="lwp"

base_path="/work/bb1143/test"

path_dataframes_scaler="$base_path/dataframe_all_data"
main_folder_output="$base_path/version_final"

# -------------------- all features -----------------------------------
all_data="all_features"
# ---------------- most importances features -----------------------------------
# all_data="most_importances"

output_path="$main_folder_output/$type_model/${variable_target}_${all_data}"


# ----------------
# for fold_num in 0 1 2 3 4; do
# for fold_num in 0 1 2 3; do
for fold_num in 0; do

for type_plot in "shap_values"; do
# Submit the job using sbatch
#SBATCH --job-name=${date}_${type_model}_${hr}
#SBATCH --partition=gpu
#SBATCH --account=bb1036
##SBATCH --account=bb1143
#SBATCH --nodes=1 
#SBATCH --gpus=2                   # Specify number of GPUs needed for the job
#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=09:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH --output=${output_path}/log_${type_plot}_${variable_target}_fold_num${fold_num}_%j.log


sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${type_model}${variable_target} 
#SBATCH --partition=compute
#SBATCH --account=bb1143
##SBATCH --account=bb1036
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64 
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=06:00:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH --output="${output_path}/log_feature_importance_${type_model}_${variable_target}_kfold_${fold_num}%j.txt"


python scripts/RF_feature_importances.py --fold-num $fold_num \
                           --variable_target $variable_target\
                           --path_dataframes_scaler $path_dataframes_scaler \
                           --output_path $output_path \
                           --type_plot $type_plot \
                           --type_model $type_model \
                           --all_data $all_data \
                           --path_models $output_path 


EOF
    done


done