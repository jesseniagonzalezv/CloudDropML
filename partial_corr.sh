#!/bin/bash

#'''
#'''

# Set environment
source ~/.bashrc
module load pytorch #python3


type_analysis="spearman_partial_correlation"
# type_analysis="pearson_partial_correlation"
 
# variables_names="Re-COT"
# variables_names="LWP-Ndmax"
# only_2variables="False"

variables_names="LWP-Ndmax-channels"
# variables_names="Re-COT-channels"
# only_2variables="True"


base_path="/work/bb1143/test"
main_folder_output="$base_path/version_final/$type_analysis"

# the next dataset is not scaled but filtered 0-2000 (only clouds)
path_dataframes_scaler="$base_path/dataframe_all_data_test"

mkdir -p "$main_folder_output"

# threshold="nofilter"
threshold="filter"

hr="-"

# for date in "04-24" "04-25" "05-02" "only_clouds_3days"; do 
# for date in "20130425"; do 
for date in "only_clouds_3days" ; do 
# hrs="09 10 12 14 15"
    if [ "$date" = "20130424" ]; then
        hrs="09 10 12 14 15 16"
    elif [ "$date" = "20130425" ]; then
        hrs="07 08 09 10 11 12 13 14 15 16"
    elif [ "$date" = "20130502" ]; then
        hrs="08 09 10 11 12 13 14 15 16 17"
    else
        hrs='-'
    fi

for hr in $hrs; do

sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=${type_analysis}
#SBATCH --partition=compute
##SBATCH --account=bb1143
#SBATCH --account=bb1036
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH -o ${main_folder_output}/log_${type_analysis}_${date}${hr}_${variables_names}_%j.txt 

python scripts/partial_corr.py --date $date \
                               --variables_names $variables_names\
                               --type_analysis $type_analysis\
                               --threshold $threshold\
                               --hr $hr \
                               --path_dataframes_scaler $path_dataframes_scaler \
                               --results_output_path $main_folder_output 
                            
                            
EOF

done
done



