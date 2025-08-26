

import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import seaborn as sns
import argparse


import seaborn as sns


def get_heatmap_captum(df_data_importances, variable_target, output_path, data_type="FI", cmap=None):
    titles_font_size = 17
    subtitles_font_size = 16
    labels_font_size = 15 # 16
    axes_font_size = 14
    
    if variable_target=="lwp":
        variable_target_plot = r'LWP'
        size_height_plot = 6
    elif variable_target=="Nd_max": 
        variable_target_plot = r'$N_{d,\ max}$'
        size_height_plot = 3

    
    df_data_importances.index.name = None # Partial
    print(data_type, np.shape(df_data_importances)[0])
    # size_height = size_height_plot*((np.shape(df_data_importances)[0]//18+1)
    size =np.shape(df_data_importances)[0]
    size_height = int((size / 18 ) * 8+2)  

    if cmap == "Blues" or cmap =="Blues_r":
        cmap_1 = cmap
        # cmap = "RdBu"
    else:
        cmap = "RdBu"
        cmap_1 = "Blues"

    
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(2*(3), size_height))
    sns.heatmap(
            df_data_importances[[df_data_importances.columns[0]]],
            cmap=cmap_1,
            # cmap='Blues',
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[0],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels=[variable_target_plot],
        )
    axes[0].set_title(f"(a) {df_data_importances.columns[0]}", fontsize=14, rotation=90)
    axes[0].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[0].tick_params(axis="y", labelsize=labels_font_size)

    sns.heatmap(
            df_data_importances[[df_data_importances.columns[1]]],
            cmap=cmap_1,
            # cmap='Blues',
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[1],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels=[variable_target_plot],
            yticklabels=False 
        )
    axes[1].set_title(f"(b) {df_data_importances.columns[1]}", fontsize=14, rotation=90)
    axes[1].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[1].tick_params(axis="y", labelsize=labels_font_size)

    sns.heatmap(
            df_data_importances[[df_data_importances.columns[2]]],
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[2],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels=[variable_target_plot],
            yticklabels=False  
        )
    axes[2].set_title(f"(c) {df_data_importances.columns[2]}", fontsize=14, rotation=90)
    axes[2].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[2].tick_params(axis="y", labelsize=labels_font_size)

    sns.heatmap(
            df_data_importances[[df_data_importances.columns[3]]],
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[3],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels= [variable_target_plot],
            yticklabels=False 
        )
    axes[3].set_title(f"(d) {df_data_importances.columns[3]}", fontsize=14, rotation=90)
    axes[3].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[3].tick_params(axis="y", labelsize=labels_font_size)

    sns.heatmap(
            df_data_importances[[df_data_importances.columns[4]]],
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[4],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels= [variable_target_plot],
            yticklabels=False  
        )
    axes[4].set_title(f"(e) {df_data_importances.columns[4]}", fontsize=14, rotation=90)
    axes[4].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[4].tick_params(axis="y", labelsize=labels_font_size)

    sns.heatmap(
            df_data_importances[[df_data_importances.columns[5]]],
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[5],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels= [variable_target_plot],
            yticklabels=False 
        )
    axes[5].set_title(f"(f) {df_data_importances.columns[5]}", fontsize=14, rotation=90)
    axes[5].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[5].tick_params(axis="y", labelsize=labels_font_size)

    sns.heatmap(
            df_data_importances[[df_data_importances.columns[6]]],
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[6],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels= [variable_target_plot],
            yticklabels=False 
        )
    axes[6].set_title(f"(g) {df_data_importances.columns[6]}", fontsize=14, rotation=90)

    axes[6].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[6].tick_params(axis="y", labelsize=labels_font_size)

    plt.savefig(f"{output_path}/NN_{data_type}_methods_{variable_target}_{cmap}.pdf", bbox_inches='tight')  # Sav   
    plt.close()


def get_heatmap_joined(df_data_importances, variable_target, output_path, data_type="Ranking", cmap='Blues'):
    titles_font_size = 17
    subtitles_font_size = 16
    labels_font_size = 15 
    labels_font_size = 14 
    axes_font_size = 14
    
    if variable_target=="lwp":
        variable_target_plot = r'$L$'
        size_height_plot = 6
    elif variable_target=="Nd_max": 
        variable_target_plot = r'$N_{\mathrm{d},\max}$'
        size_height_plot = 3

    
    df_data_importances.index.name = None 
    print(data_type, np.shape(df_data_importances)[0])
    size =np.shape(df_data_importances)[0]
    size_height = int((size / 18 ) * 8+6)  
    
    # plt.figure(figsize=(2*(np.shape(df_data_importances)[1]), size_height))
    
    plt.figure(figsize=(7, 5)) 
    sns.heatmap(
            df_data_importances,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            linecolor='black',
            annot_kws={"size": 13},
        )
    plt.title(variable_target_plot, fontsize=labels_font_size)
    plt.xticks(fontsize=labels_font_size, rotation=90)  
    plt.yticks(fontsize=labels_font_size)  
    plt.savefig(f"{output_path}/NN_{data_type}_methods_{variable_target}_{cmap}.pdf", bbox_inches='tight')  # Sav   
    plt.close()
    

def main():
    parser = argparse.ArgumentParser(description='Heatmaps')
    arg = parser.add_argument
    arg('--output_path', type=str, default="/work/bb1036/b381362/output/final_results/")
    arg('--all_data', type=str, default="all_features", help='number of flods to split the dataset')
    # arg('--variable_target', type=str, default="lwp", help='"Re, COT, LWP, Ndmax" List of the variables to use.')
    
    args = parser.parse_args()
    output_path = args.output_path
    all_data = args.all_data
    # variable_target = args.variable_target


    for variable_target in ["lwp", "Nd_max"]:
        result_df = pd.DataFrame()
        for fold_num in range(4):
            file_path_latest = f'{output_path}/FI_NN_others_{variable_target}_{fold_num}_all_train_no_shape.csv'
            data_fold_others = pd.read_csv(file_path_latest)
            data_fold_others = data_fold_others.drop(columns=['Gradient-Based Importance'], errors='ignore')  

            file_path_latest = f'{output_path}/FI_NN_captum_{variable_target}_{fold_num}.csv'
            data_fold = pd.read_csv(file_path_latest)
            data_fold = data_fold.drop(columns=['fa_attr_test_norm_sum'], errors='ignore')  
            data_fold = data_fold.drop(columns=["Feature"])
            
            column_mapping = {
                "ig_attr_test_norm_sum": "Int Grads",
                "ig_nt_attr_test_norm_sum": "Int Grads w/SmoothGrad",
                "dl_attr_test_norm_sum": "DeepLift",
                "gs_attr_test_norm_sum": "GradientSHAP"
            }
            data_fold.rename(columns=column_mapping, inplace=True)

            file_path_latest = f'{output_path}/{all_data}_shap_NN_{variable_target}_k_fold_{fold_num}_s1024_train.csv'
            data_shape_fold = pd.read_csv(file_path_latest)[["Mean_SHAP_Value"]]
            data_fold_others = pd.concat([data_fold_others, data_fold, data_shape_fold], axis=1)
            
            # data_fold_others[data_fold_others.select_dtypes(include=['number']).columns] = data_fold_others.select_dtypes(include=['number'])
            data_fold_others[data_fold_others.select_dtypes(include=['number']).columns] = data_fold_others.select_dtypes(include=['number']).abs()


            num_cols = data_fold_others.select_dtypes(include=['number']).columns

            data_fold_others[num_cols] = (data_fold_others[num_cols] - data_fold_others[num_cols].min()) / (data_fold_others[num_cols].max() - data_fold_others[num_cols].min())

            data_fold_others["Fold"] = fold_num
            result_df = pd.concat([result_df, data_fold_others], ignore_index=True) 

        
        mean_importances = result_df.groupby("Feature",sort=False).mean() # test
        # mean_importances = abs(mean_importances)
        mean_importances[num_cols] = (mean_importances[num_cols] - mean_importances[num_cols].min()) / (mean_importances[num_cols].max() - mean_importances[num_cols].min())

        median_importances = result_df.groupby("Feature",sort=False).median() # test

        # median_importances = abs(median_importances)
        median_importances[num_cols] = (median_importances[num_cols] - median_importances[num_cols].min()) / (median_importances[num_cols].max() - median_importances[num_cols].min())
        
        # Rename the columns using the mapping
        column_mapping = {
            "Permutation Importance": "Permutation \n Importance",
            "Integrated Gradients": "Int Grads \n (mean)",
            "Int Grads": "Int Grads \n (Sum + L1)",
            "Int Grads w/SmoothGrad": "Noise Tunnel \n Int Grads", 
            "DeepLift": "Deep Lift",
            "GradientSHAP": "Grad SHAP",
            "Mean_SHAP_Value": "SHAP Values"
        }
        mean_importances.rename(columns=column_mapping, inplace=True)
        median_importances.rename(columns=column_mapping, inplace=True)

        mean_importances = mean_importances.drop(columns=["Fold"])
        mean_importances = mean_importances.round(2)
        mean_importances.to_csv(f'{output_path}/{all_data}_{variable_target}_mean_along_folds.csv', index=True)


        median_importances = median_importances.drop(columns=["Fold"])
        median_importances = median_importances.round(2)
        median_importances.to_csv(f'{output_path}/{all_data}_{variable_target}_median_along_folds.csv', index=True)
        
# ================== obtaining mean importances and save as csv ========
        print(f" ======================= names all methods to average: {mean_importances.columns}")
        global_importance = mean_importances.copy()
        # df_mean_normalized = (mean_importances - mean_importances.min()) / (mean_importances.max() - mean_importances.min())
        # global_importance["Mean"] = df_mean_normalized.mean(axis=1)
        # global_importance.round(3).to_csv(f'{output_path}/{all_data}_{variable_target}_mean_along_methods.csv', index=True)
        global_importance["Mean"] = global_importance.mean(axis=1)
        # global_importance["Median"] = global_importance.drop(columns=["Mean"]).median(axis=1)
        
        global_importance=global_importance.round(2)
        global_importance.to_csv(f'{output_path}/{all_data}_{variable_target}_mean_median_along_methods.csv', index=True)

        get_heatmap_joined(df_data_importances=global_importance,
                               variable_target=variable_target, 
                               cmap="Blues", 
                               data_type="mean_along_folds", 
                               output_path=output_path)
       
        print(" ----------------- FI NN - Captum heatmap -------------------------------")
        get_heatmap_captum(df_data_importances=mean_importances, variable_target=variable_target, output_path=output_path)
        
        # get_heatmap_captum(df_data_importances=abs(mean_importances), variable_target=variable_target, cmap="Blues", output_path=output_path)
        get_heatmap_captum(df_data_importances=(mean_importances), variable_target=variable_target, cmap="Blues", output_path=output_path)
    

        # ------------ Ranking ------------------------
        if variable_target=="lwp":
            # n_features = 6
            n_features = 3
        elif variable_target=="Nd_max": 
            # n_features = 10
            n_features = 7
        
        # ranked_df = abs(mean_importances).rank(ascending=False)
        ranked_df = (mean_importances).rank(ascending=False)
        
        # --------Compute mean ranking ----------------------------------------
        data_columns = ranked_df.columns  
        print(f"--------- Mean and Median of the columns {data_columns}")
        ranked_df["Mean Ranking"] = ranked_df[data_columns].mean(axis=1)
        ranked_df["Median Ranking"] = ranked_df[data_columns].median(axis=1)

        ranked_df.round(3).to_csv(f'{output_path}/{all_data}_{variable_target}_rankings_mean_median.csv', index=True)
        # ---------------------------------------------------------------------

        print(" ----------------- Ranking heatmap -------------------------------")
        get_heatmap_joined(df_data_importances=ranked_df,
                           variable_target=variable_target, 
                           cmap="Blues_r", 
                           data_type="Ranking", 
                           output_path=output_path)
    
        top_n_features = ranked_df.nsmallest(n_features, "Mean Ranking")

        
        print(" ----------------- Top features heatmap -------------------------------")
        get_heatmap_captum(df_data_importances=top_n_features,
                           variable_target=variable_target, 
                           cmap="Blues_r", 
                           data_type=f"Top {n_features}Features",
                           output_path =output_path)


if __name__ == '__main__':
    main()
 

