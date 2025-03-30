#!/bin/bash

# Set environment
source ~/.bashrc
module load pytorch 

# ==============================================================================
# ------------------------------- Configure it --------------------------------
# ==============================================================================
base_path="/work/bb1143/test"

prepro_type="StandardScaler_StandardScaler" 
batch_size=512 
lr=1e-5 
n_epochs=60

loss_type="SmoothL1Loss"
# loss_type="MSELoss"
variables_names="LWP-Ndmax"


variable_target="lwp"
# variable_target="Nd_max"


# process="predict_plot"
# process="plotting"
# process="train_predict_plot"
process="train"

for type_model in "RF" "XGBoost" "NN"; do


# for all_data in "most_importances" "all_features"; do
for all_data in "all_features"; do
# for all_data in "most_importances"; do

path_dataframes_scaler="$base_path/dataframe_all_data"
main_folder_output="$base_path/version_final"



#------------------------------- test to predict plot ---------------------------------
type_test_set="validation"
path_icon_specific_test=None
path_channels_specific_test=None
#-------------------------------
#-------------------------------
time="07"
time_format="0700" 
date="20130425"
path_icon_specific_test="/work/bb1036/rttov_share/dataset_ICON/icon_germany_${date}_T${time}_lwp_Nd.nc"
path_channels_specific_test="/work/bb1036/rttov_share/dataset_SPECTRAL/S3COM_OP_${date}_T${time}_36ch_rad.nc"
type_test_set="testing"


path_models="$main_folder_output/$type_model/${variable_target}_${all_data}/"

# ==============================================================================
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ==============================================================================

for fold_num in 0; do
# for fold_num in 0 1 2 3 4; do
    
if [ "$type_model" = "NN" ]; then
    path_model_file="${path_models}/${all_data}_${type_model}_${variable_target}_k_fold_${fold_num}.pth"
else
    path_model_file="${path_models}/${all_data}_${type_model}_${variable_target}_k_fold_${fold_num}.joblib"
fi

echo $type_test_set
if [ "$type_test_set" = "validation" ]; then
formatted_time_data="validation"
else
formatted_time_data="${date}T${time_format}"
fi

path_pred_target_nc="${path_models}/predictions/${all_data}_prediction_target_${formatted_time_data}_${type_model}_${variable_target}_k_fold_${fold_num}.nc"



path_output=$path_models

metrics="R2 MAE"


# Create the directory and any necessary parent directories
mkdir -p "$path_models"


# ## ------------------------------RF-------------------------------


echo $process


if [ "$type_model" = "NN" ]; then
if [ "$process" = "train_predict" ] || [ "$process" = "train" ] || [ "$process" = "train_predict_plot" ]; then
     echo "pytorch_inverse_1target"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${type_model}${variable_target}${fold_num} 
#SBATCH --partition=gpu
##SBATCH --account=bb1143
#SBATCH --account=bb1036
#SBATCH --nodes=1 
##SBATCH --cpus-per-task=32 
#SBATCH --gpus=1                   # Specify number of GPUs needed for the job
#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=01:00:00            # Set a limit on the total run time
##SBATCH --mail-type=FAIL           # Notify user by email in case of job failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH --output="${path_models}/log_train_${type_model}_kfold_${fold_num}%j.txt"



    python -u scripts/inverse_model_NN.py --fold-num $fold_num \
                                                --variables_names $variables_names\
                                                --loss_type $loss_type\
                                                --variable_target $variable_target\
                                                 --type-model $type_model \
                                                 --lr $lr \
                                                 --n-epochs $n_epochs \
                                                 --batch-size $batch_size \
                                                 --path_dataframes_scaler $path_dataframes_scaler \
                                                 --all_data $all_data \
                                                 --path_models $path_models 
    

EOF
fi 
fi

if { [ "$type_model" = "RF" ] || [ "$type_model" = "XGBoost" ]; } && { [ "$process" = "train" ] || [ "$process" = "train_predict_plot" ]; }; then

echo "inverse_model_$type_model"
echo "training ------------------------------------------------------------"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${type_model}${variable_target} 
#SBATCH --partition=compute
##SBATCH --account=bb1143
#SBATCH --account=bb1036
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=64 
## SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=06:00:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH --output="${path_models}/log_${type_model}_${variable_target}_kfold_${fold_num}%j.txt"

python -u scripts/inverse_model_RF.py --type-model $type_model \
                           --results_output_path $path_output \
                           --variables_names $variables_names \
                           --fold-num $fold_num \
                           --path_dataframes_scaler $path_dataframes_scaler \
                           --all_data $all_data \
                           --variable_target $variable_target 
EOF
fi



if [ "$process" = "train_predict_plot" ] || [ "$process" = "predict_plot" ]; then
echo "get_prediction_test -----------------------------------------------"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${type_model}${variable_target}${fold_num} 
#SBATCH --partition=compute
##SBATCH --account=bb1143
#SBATCH --account=bb1036
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=64 
## SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=00:30:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH --output="${path_models}/predictions/log_${type_model}_kfold_${fold_num}%j.txt"

    python -u scripts/get_prediction_test.py --fold-num $fold_num \
                                                --variables_names $variables_names\
                                                --variable_target $variable_target\
                                                 --type-model $type_model \
                                                 --path_model_file $path_model_file \
                                                 --path_dataframes_scaler $path_dataframes_scaler \
                                                 --all_data $all_data \
                                                 --path_icon_specific_test $path_icon_specific_test \
                                                 --path_channels_specific_test $path_channels_specific_test \
                                                 --path_models $path_models 


    python plotting/plotting_prediction_target.py --variable_target $variable_target \
                                         --path_output "$path_output/predictions" \
                                         --path_pred_target_nc $path_pred_target_nc \
                                         --metrics $metrics 
EOF
fi 



if [ "$process" = "plotting" ]; then

    python plotting/plotting_prediction_target.py --variable_target $variable_target \
                                         --path_output "$path_output/predictions" \
                                         --path_pred_target_nc $path_pred_target_nc \
                                         --metrics $metrics 


fi


done
done
done