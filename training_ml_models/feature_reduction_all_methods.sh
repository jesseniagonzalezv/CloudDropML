#!/bin/bash

# Set environment
source ~/.bashrc
module load pytorch 

#-------------------------------------------------------------------
# ------------ configuration for NN --------------------------------
batch_size=512
lr=1e-5 
n_epochs=60 
n_epochs=1 
# ------------------------------------------------------------------
#-------------------------------------------------------------------

# ------------------------------------------------------------------
#--------------paths---------------------------------
base_path="/work/bb1143/test"


path_dataframes_scaler="$base_path/dataframe_all_data"
main_folder_output="$base_path/version_final"
# ------------------------------------------------------------------
#-------------------------------------------------------------------


for type_model in "RF" "XGBoost" "NN"; do
for variable_target in "lwp" "Nd_max"; do
# for fold_num in 0 1 2 3; do
for fold_num in 0 1 2 3 4; do


mkdir -p "$main_folder_output"

 
if [ "$type_model" = "RF" ] || [ "$type_model" = "XGBoost" ]; then
path_results="$main_folder_output/feature_reduction/RFE"

sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=FR-${type_model}${variable_target}${fold_num} 
#SBATCH --partition=compute
#SBATCH --account=bb1143
##SBATCH --account=bb1036
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=64 
#SBATCH --time=08:00:00  #8 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
##SBATCH -o models-ML.o%j
##SBATCH --error=models-ML%j.log
#SBATCH --output="${path_results}/log_FR_${type_model}_${variable_target}_kfold_${fold_num}%j.txt"



python scripts/feature_reduction_all_methods.py --fold-num $fold_num \
                                        --variable_target $variable_target \
                                        --type_model $type_model \
                                        --path_dataframes_scaler $path_dataframes_scaler \
                                        --path_results $path_results 

EOF
fi

if [ "$type_model" = "NN" ]; then
# for type_NN_ranking in "mean" "median"; do
for type_NN_ranking in "mean"; do
    path_results="$main_folder_output/feature_reduction/NN_ranking_${type_NN_ranking}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=FR-${type_model}${variable_target}${fold_num} 
#SBATCH --partition=gpu
#SBATCH --account=bb1143
##SBATCH --account=bb1036
#SBATCH --nodes=1 
##SBATCH --cpus-per-task=32 
#SBATCH --gpus=1                   # Specify number of GPUs needed for the job
#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=07:00:00            # Set a limit on the total run time
##SBATCH --mail-type=FAIL           # Notify user by email in case of job failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH --output="${path_results}/log_FR_${type_model}_${variable_target}_kfold_${fold_num}%j.txt"


python scripts/feature_reduction_all_methods.py --fold-num $fold_num \
                                        --variable_target $variable_target \
                                        --type_model $type_model \
                                        --lr $lr \
                                        --n-epochs $n_epochs \
                                        --batch-size $batch_size \
                                        --path_dataframes_scaler $path_dataframes_scaler \
                                        --csv_ranking_path $main_folder_output \
                                        --type_NN_ranking $type_NN_ranking \
                                        --path_results $path_results 


EOF

done
fi


done
done
done
