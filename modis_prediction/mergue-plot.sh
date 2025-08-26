#!/bin/bash

# Set environment
# source ~/.bashrc
module load pytorch 


# for all_data in "all_features" "most_importances"; do
# for all_data in  "all_features"; do
for all_data in  "most_importances"; do
# for fold_num in 0 1; do
for fold_num in 0; do
# for fold_num in 1; do
#-----------
# for type_aggregation in "mean_by_date"; do
# for type_aggregation in "plot_hist_plot2d_date" "plot_hexbin"; do
for type_aggregation in "plot_mean_scatter" "plot_hist_plot2d_date"; do
#---------
# for type_aggregation in "mean_all_dates"; do
# for type_aggregation in "plot_averaged_global_hexbin" "plot_averaged_global_hist" "plot_averaged_global_scatter"; do
#----------


# base_path="/scratch/b/b381362//500_011_retraining_last_plots_60N_60S/global_${all_data}_bb1143_both_fold${fold_num}"    

base_path="/scratch/b/b381362/1000epochs_500_011_retraining_last_plots_60N_60S/global_${all_data}_bb1143_both_fold${fold_num}"    # this is to check VF

output_nc="$base_path/hist"


#------------------------
output_folder="${base_path}/plots_global"

mkdir -p "$output_folder"
echo $output_folder

echo ${type_aggregation}


# python scripts/mergue_netcdf_plot.py
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=mer${type_aggregation} 
#SBATCH --partition=compute
##SBATCH --account=bb1036
#SBATCH --account=bb1143
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8 #4 
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=08:00:00            # Set a limit on the total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications

#SBATCH --output="$output_folder/log_mergue_plot_%j.txt"


python scripts/mergue_netcdf_plot.py --type_aggregation $type_aggregation \
--output_folder $output_folder \
--output_nc $output_nc

EOF

done
done
done
