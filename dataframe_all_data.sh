#!/bin/bash


source ~/.bashrc
module load pytorch 

#--------------------------------------
#--------------------------------------
variables_names="LWP-Ndmax"
# variable_target="Nd_max"
# variable_target="lwp"

# #--------------------------------------
# variables_names="Re-COT"
# variable_target="cot"
# variable_target="Reff_cloud_top"
#--------------------------------------
#--------------------------------------
# threshold="nofilter"
threshold="filter"
results_output_path="/work/bb1143/test/dataframe_all_data"

mkdir -p "$results_output_path"


sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=data${variables_names} 
#SBATCH --partition=compute
##SBATCH --account=bb1143
#SBATCH --account=bb1036
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4 
## SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=08:00:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH --output="${results_output_path}/log__dataframe_all_data_${variables_names}_${threshold}_%j.txt"

python scripts/dataframe_all_data.py            --results_output_path $results_output_path \
                           --variables_names $variables_names \
                           --threshold $threshold 



EOF