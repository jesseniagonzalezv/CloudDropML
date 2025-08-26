#!/bin/bash

# Set environment
# source ~/.bashrc
module load pytorch 


# main_folder_ml_lwp="/work/bb1036/b381362/"
# main_folder_ml_nd="/work/bb1036/b381362/"

# main_folder_ml_lwp="/work/bb1143/b381362/"
# main_folder_ml_nd="/work/bb1143/b381362/"

main_folder_ml_lwp="/work/bb1143/b381362/"
main_folder_ml_nd="/work/bb1036/b381362/"

output_results="/work/bb1143/b381362/"

# result_folder="$output_results/lwp_nd_compare_methods/plot_cap3/test_global_data_${all_data}_lwp1143_nd1036"


echo "compare methods to predict Nd LWP"
python scripts/methods_obtaine_lwp_nd.py --main_folder_ml_lwp $main_folder_ml_lwp \
                                         --main_folder_ml_nd $main_folder_ml_nd \
                                         --output_results $output_results
                                         
# sbatch <<EOF
# #!/bin/bash
# #SBATCH --job-name=${type_model}${variable_target}${fold_num} 
# #SBATCH --partition=gpu
# ##SBATCH --account=bb1143
# #SBATCH --account=bb1036
# #SBATCH --nodes=1 
# ##SBATCH --cpus-per-task=32 
# #SBATCH --gpus=1                   # Specify number of GPUs needed for the job
# #SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
# #SBATCH --mem=0                    # Request all memory available on all nodes
# #SBATCH --time=01:00:00            # Set a limit on the total run time
# ##SBATCH --mail-type=FAIL           # Notify user by email in case of job failure
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications
# #SBATCH --output="${path_models}/log_train_${type_model}_kfold_${fold_num}%j.txt"

# echo "compare methods to predict Nd LWP"
# python scripts/methods_obtaine_lwp_nd.py --main_folder $main_folder 
# EOF