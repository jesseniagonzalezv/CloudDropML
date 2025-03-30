import seaborn as sns
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost as xgb
import sys
import os


from sklearn.neural_network import MLPRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.functions import load_dataframe_from_netcdf

import matplotlib as mpl
import matplotlib.collections as mcoll


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path_results', type=str, default="/work/bb1036/b381362/output/",
        help='path of the folder to save the outputs')
    arg('--variable_target', type=str, default="lwp", help='"Re, COT, LWP, Ndmax" List of the variables to use.')

    args = parser.parse_args()
    path_results = args.path_results
    variable_target = args.variable_target

    
    type_NN_ranking = "mean" 

    NN_folder_save = f"{path_results}/NN_ranking_{type_NN_ranking}"
    figure_name = f'{path_results}/{variable_target}_model_performance_vs_features'

    folds = [0, 1, 2, 3] 
    models = ["RF", "XGBoost", "NN"]
    color_map = {
        'RF': 'saddlebrown',
        'XGBoost': 'darkorange',
        'NN': 'C0'  # default blue
    }

    results_main = pd.concat(
        [
            pd.read_csv(f"{path_results}/RFE/{variable_target}_performance_{type_model}_fold{fold_num}_ranking.csv")
            for fold_num in folds for type_model in models[:2]]
    )

    
    results_ranking = pd.concat(
        [
            pd.read_csv(f"{NN_folder_save}/{variable_target}_performance_{type_model}_fold{fold_num}_ranking.csv")
            for fold_num in folds for type_model in [models[2]]])

    # ------------------------------------------------------
    all_results = pd.concat([results_main, results_ranking], ignore_index=True)

    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    fontsize_lables = 17.5
    new_color = (0.6, 0.4, 0.8)  
    color_map = {
        'RF': 'saddlebrown',
        'XGBoost': 'darkorange', 
        'NN': 'C0'  
    }

    if variable_target == 'lwp':
        variable_target_title = '$L$'
        threshold = 0.982  # 95% of the max performance
    elif variable_target == "Nd_max":
        variable_target_title ='$N_{\mathrm{d},\max}$'
        threshold = 0.95  # 95% of the max performance
    
    print(f"threshold {variable_target}: {threshold}")
       
    for metric in ["mae", "r2"]:
        fig = plt.figure(figsize=(7, 5))

        if metric == "r2":
            max_metrics = all_results.groupby('model')[metric].max()
            # Find the smallest num_features where metric is at least 98% of the max
            optimal_features = all_results[
                all_results.apply(lambda row: row[metric] >= threshold * max_metrics[row['model']], axis=1)
            ].groupby('model')['num_features'].min().reset_index()

            # Ensure optimal_features follows the order in models
            optimal_features['model'] = pd.Categorical(optimal_features['model'], categories=models, ordered=True)
            optimal_features = optimal_features.sort_values('model')

        elif metric == "mae":
            optimal_features = all_results.loc[all_results.groupby('model')[metric].idxmin(), ['model', 'num_features']]

        im_lines = sns.lineplot(
                    data=all_results,
                    x='num_features',
                    y=metric,
                    hue='model',
                    style='model',       # Different line styles per model
                    markers=True,        # Puts a marker at each x-value
                    dashes=True,         # Allows different dash patterns per model
                    errorbar='sd',       # Standard deviation shading
                    palette=color_map,
                    linewidth=1.2
                )



        
        # Add vertical lines for optimal feature selection
        for _, row in optimal_features.iterrows():
            im = plt.axvline(
                x=row['num_features'],
                linestyle='-.',
                color=color_map[row['model']],  # Match the color of the model
                alpha=0.8, 
                linewidth=2,
                label=f"{row['model']} (opt: {row['num_features']})"
            )

        plt.xlabel('Number of Channels', fontsize=fontsize_lables)
        plt.grid(True)
        plt.xticks(all_results['num_features'], fontsize=fontsize_lables - 3.5)
        plt.yticks(fontsize=fontsize_lables-3.5)   
        if metric=="r2":
            plt.ylim(0.2, 1)  # R2 puede variar de 0 a 1
            plt.ylabel('R²', fontsize=fontsize_lables)
            plt.title(f'R² vs. Number of channels for {variable_target_title} (Mean ± SD)', fontsize=fontsize_lables-1.5)
        elif metric=="mae":
            plt.ylabel('MAE', fontsize=fontsize_lables)
            plt.title(f'MAE vs. Number of channels for {variable_target_title} (Mean ± SD)', fontsize=fontsize_lables-1.5)
        plt.legend(fontsize=fontsize_lables-3.5, loc='lower right')
        plt.tight_layout()
        plot_name = f"{figure_name}_{metric}"
        plt.savefig(f"{plot_name}.pdf")
        print(plot_name)
        plt.close()


if __name__ == '__main__':
    main()