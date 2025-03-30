#!/bin/bash



# Set environment
source ~/.bashrc
module load pytorch #python3

type_analysis="correlation"
# type_analysis="mutual_information"

# type_analysis="jointplot"
# type_analysis="histogram"

# variables_names="Re-COT"
# variables_names="LWP-Ndmax"
# only_2variables="False"

variables_names="LWP-Ndmax-channels"
# variables_names="Re-COT-channels"
only_2variables="True"
# only_2variables="False"


base_path="/work/bb1143/test/"
path_dataframes_scaler="$base_path/dataframe_all_data"
main_folder_output="$base_path/version_final/$type_analysis"


mkdir -p "$main_folder_output"
# threshold="nofilter"
threshold="filter"


hr="-"
# for date in "04-24" "04-25" "05-02" "only_clouds_3days"; do 
# for date in "20130502"; do 
# for date in "only_clouds_3days" "04-24" "04-25" "05-02" ; do 
for date in "only_clouds_3days" ; do 
    if [ "$date" = "20130424" ]; then
        hrs="09 10 12 14 15 16"
        
    elif [ "$date" = "20130425" ]; then
        hrs="07 08 09 10 11 12 13 14 15 16"
        # hrs="07 08 13"
      
    elif [ "$date" = "20130502" ]; then
        hrs="08 09 10 11 12 13 14 15 16 17"
        # hrs="08 13"
        hrs="13"
    else
        hrs='-'
    fi

for hr in $hrs; do

sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=${type_analysis}
#SBATCH --partition=compute
#SBATCH --account=bb1036
#SBATCH --nodes=1
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@gmail.com # Email address for notifications
#SBATCH -o ${main_folder_output}/log_${type_analysis}_${date}_${variables_names}_${only_2variables}%j.txt 

python scripts/correlation_joinplot.py --date $date \
                               --variables_names $variables_names\
                               --type_analysis $type_analysis\
                               --only_2variables $only_2variables\
                               --threshold $threshold\
                               --hr $hr \
                               --path_dataframes_scaler $path_dataframes_scaler \
                               --results_output_path $main_folder_output 
                               
EOF
# >&${main_folder_output}/log_${type_analysis}_${date}_${variables_names}_${only_2variables}.txt  

done
done                            




