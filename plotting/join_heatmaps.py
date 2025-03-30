
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os 

from matplotlib.colors import Normalize
from matplotlib import colors


def plot_heatmap_all_features(output_path, sc_data, spc_data, mi_data, RF_feature_importance, XGBoost_feature_importance, NN_feature_importance):
    # pmi_data
    titles_font_size = 17
    subtitles_font_size = 13
    # labels_font_size = 15
    labels_font_size = 13 
    axes_font_size = 12.5

    for idx_variable in range(len(sc_data.columns)):
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(13/9*4, 8)) #8
        # SC heatmap
        sns.heatmap(
            sc_data[[sc_data.columns[idx_variable]]],
            cmap="RdBu",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[0],
            linecolor='black',
            annot_kws={"size": axes_font_size},
        )
        axis_num = 0
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n SC", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
        axes[axis_num].tick_params(axis="y", labelsize=labels_font_size)
        
        # SPC heatmap
        sns.heatmap(
            spc_data[[spc_data.columns[idx_variable]]],
            cmap="RdBu",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[1],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n  SPC", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
        
        # MI heatmap
        sns.heatmap(
            mi_data[[mi_data.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[2],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n  MI", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
        
        # RF Feature importances all features CV heatmap
        sns.heatmap(
            RF_feature_importance[[RF_feature_importance.columns[idx_variable]]],
            cmap="Blues",
            # vmin=0,
            # vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[3],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n RF", fontsize=subtitles_font_size, rotation=0) #FI - 
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
        
        # XGB Feature importances all features  heatmap
        sns.heatmap(XGBoost_feature_importance[[XGBoost_feature_importance.columns[idx_variable]]],
                    cmap="Blues",
                    annot=True,
                    fmt=".2f",
                    linewidths=0.5,
                    cbar=False,
                    ax=axes[4],
                    linecolor='black',
                    annot_kws={"size": axes_font_size},
                    yticklabels=False  # Hide y-axis labels
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n XGB", fontsize=subtitles_font_size-0.2, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
    
            # NN Feature importances all features  heatmap
        sns.heatmap(
            NN_feature_importance[[NN_feature_importance.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[5],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False 
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n NN", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
        
        name_output = f"{output_path}/FI_all_methods_variable{idx_variable}.pdf" 
        plt.savefig(name_output, bbox_inches='tight') 
        print(f" ------------ Saved in {name_output} ----------------------------")
        plt.close()


def plot_heatmap_mif(output_path, RF_most_importance_feature, XGBoost_most_importance_feature, NN_most_importance_feature):
    titles_font_size = 17
    subtitles_font_size = 13
    labels_font_size = 13 
    axes_font_size = 12.5
        
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # ------------------------------- MIF -----------------------------
        # -----------------------------------------------------------------
    for idx_variable in range(2):
        if idx_variable==0:
            size_height = 7.25
        elif idx_variable==1:
            size_height = 7    
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13/9*2, size_height))
        
        sns.heatmap(
            RF_most_importance_feature[[RF_most_importance_feature.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[0],
            linecolor='black',
            annot_kws={"size": axes_font_size},
        )
        axes[0].set_title("(g) RF", fontsize=subtitles_font_size, rotation=0) #MIF - 
        axes[0].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
        axes[0].tick_params(axis="y", labelsize=labels_font_size)
    
        # Feature importances RFECV heatmap
        sns.heatmap(
            XGBoost_most_importance_feature[[XGBoost_most_importance_feature.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[1],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axes[1].set_title("(h) XGB", fontsize=subtitles_font_size-0.2, rotation=0)
        axes[1].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
        # Feature importances RFECV heatall_heamap
        sns.heatmap(
            NN_most_importance_feature[[NN_most_importance_feature.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[2],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axes[2].set_title("(i) NN", fontsize=subtitles_font_size, rotation=0)
        axes[2].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
        # plt.tight_layout()
    
        name_output = f"{output_path}/MIF_variable{idx_variable}.pdf" 
        plt.savefig(name_output, bbox_inches='tight')  
        print(f" ------------ Saved in {name_output} ----------------------------")
        plt.close()


def plot_heatmap_normalized(output_path, sc_data, spc_data, mi_data, RF_feature_importance, XGBoost_feature_importance, NN_feature_importance, all_methods_mean):  #plot_heatmap_ranking
    titles_font_size = 17
    subtitles_font_size = 13
    labels_font_size = 13 
    axes_font_size = 12.5

    for idx_variable in range(len(sc_data.columns)):
        fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(13/9*4, 9))  # ,8

        # SC heatmap ------------------------------------------------
        sns.heatmap(
            sc_data[[sc_data.columns[idx_variable]]],
            cmap="Blues",
            # center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[0],
            linecolor='black',
            annot_kws={"size": axes_font_size},
        )
        axis_num = 0
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n SC", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
        axes[axis_num].tick_params(axis="y", labelsize=labels_font_size)
        
        # SPC heatmap ------------------------------------------------
        sns.heatmap(
            spc_data[[spc_data.columns[idx_variable]]],
            cmap="Blues",
            # center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[1],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False 
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n  SPC", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
        
        # MI heatmap------------------------------------------------
        sns.heatmap(
            mi_data[[mi_data.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[2],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n  MI", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
        
        # RF ------------------------------------------------
        sns.heatmap(
            RF_feature_importance[[RF_feature_importance.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[3],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n RF", fontsize=subtitles_font_size, rotation=0) #FI - 
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
        
        # XGBoost ------------------------------------------------
        sns.heatmap(
                        XGBoost_feature_importance[[XGBoost_feature_importance.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[4],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n XGB", fontsize=subtitles_font_size-0.2, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
    
        # NN ------------------------------------------------
        sns.heatmap(
            NN_feature_importance[[NN_feature_importance.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[5],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n NN", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)


        # mean ------------------------------------------------
        sns.heatmap(
            all_methods_mean[[all_methods_mean.columns[idx_variable]]],
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[6],
            linecolor='black',
            annot_kws={"size": axes_font_size},
            yticklabels=False  
        )
        axis_num += 1
        axes[axis_num].set_title(f"({chr(98-idx_variable)}.{axis_num+1})\n Mean", fontsize=subtitles_font_size, rotation=0)
        axes[axis_num].tick_params(axis="x", labelsize=labels_font_size, rotation=90)

        
        name_output = f"{output_path}/mean_FI_all_methods_variable{idx_variable}.pdf" 
        plt.savefig(name_output, bbox_inches='tight')  
        print(f" ------------ Saved in {name_output} ----------------------------")
        plt.close()


def read_dataframe_lwp_nd_fi(variable_target, output_path, type_model, all_data):
    path = f"{output_path}/{type_model}/{variable_target}_{all_data}"
    
    result_df = pd.DataFrame()
    # for fold_num in range(5):
    for fold_num in range(4):
        file_path_latest = f'{path}/{all_data}_features_importances_{type_model}_{variable_target}_k_fold_{fold_num}.csv'
        data_fold_others = pd.read_csv(file_path_latest)
        result_df = pd.concat([result_df, data_fold_others], ignore_index=True) 

    mean_importances = result_df.groupby("Feature", sort=False)["Importance"].mean()
    # mean_importances = result_df.groupby("Feature", sort=False)["Importance"].median()
    if variable_target=="lwp":
        mean_importances = mean_importances.reset_index(name='$L$')  
    elif variable_target=="Nd_max":
        mean_importances = mean_importances.reset_index(name='$N_{\mathrm{d},\max}$')  
        
    return mean_importances

    
def read_dataframe_lwp_nd_fi_shap(variable_target, output_path, type_model, all_data):
    path = f"{output_path}/{type_model}/{variable_target}_{all_data}"
    
    result_df = pd.DataFrame()
    # for fold_num in range(5):
    for fold_num in range(4):
        file_path_latest = f'{path}/{all_data}_shap_{type_model}_{variable_target}_k_fold_{fold_num}_s1024_train.csv'
        # file_path_latest = f'{path}/{all_data}_shap_{type_model}_{variable_target}_k_fold_{fold_num}_s4056_train.csv'
        
        data_fold_others = pd.read_csv(file_path_latest)
        result_df = pd.concat([result_df, data_fold_others], ignore_index=True) 

    mean_importances = result_df.groupby("Feature", sort=False)["Mean_SHAP_Value"].mean()
    # mean_importances = result_df.groupby("Feature", sort=False)["Mean_SHAP_Value"].median()
    if variable_target=="lwp":
        mean_importances = mean_importances.reset_index(name='$L$')  
    elif variable_target=="Nd_max":
        mean_importances = mean_importances.reset_index(name='$N_{\mathrm{d},\max}$')  
        
    return mean_importances


def mergue_most_important_with_all(mean_importances_df_ndmax, mean_importances_df_lwp, name_features, target_variables):

    merged_df_all_features = pd.DataFrame(np.nan, index=name_features, columns=target_variables)
    for feature, value in zip(mean_importances_df_ndmax.index.values, mean_importances_df_ndmax.values):
        merged_df_all_features.loc[feature, target_variables[0]] = value  
    
    for feature, value in zip(mean_importances_df_lwp.index.values, mean_importances_df_lwp.values):
        merged_df_all_features.loc[feature, target_variables[1]] = value 

    return merged_df_all_features



def create_dataframe_lwp_nd_fi(output_path, type_model, all_data, name_features=None, target_variables=None):
    
    mean_importances_df_lwp = read_dataframe_lwp_nd_fi(variable_target="lwp", 
                                                           output_path=output_path, 
                                                           type_model=type_model, 
                                                           all_data=all_data)
    
    mean_importances_df_ndmax = read_dataframe_lwp_nd_fi(variable_target="Nd_max", 
                                                           output_path=output_path, 
                                                           type_model=type_model, 
                                                           all_data=all_data)
    
    if all_data == "all_features":
        merged_df_all_features = pd.merge(mean_importances_df_ndmax, mean_importances_df_lwp, on='Feature', how='inner')  
        merged_df_all_features = merged_df_all_features.set_index('Feature')  
        merged_df_all_features.index.name = None # Partial

    elif all_data == "most_importances":
        mean_importances_df_lwp = mean_importances_df_lwp.set_index('Feature')  
        mean_importances_df_lwp.index.name = None # Partial
    
        mean_importances_df_ndmax = mean_importances_df_ndmax.set_index('Feature')  
        mean_importances_df_ndmax.index.name = None 
    
        merged_df_all_features = mergue_most_important_with_all(mean_importances_df_ndmax=mean_importances_df_ndmax,
                                                                mean_importances_df_lwp=mean_importances_df_lwp,
                                                                name_features= name_features, 
                                                                target_variables=target_variables)

    return merged_df_all_features

    
def create_dataframe_lwp_nd_fi_shap(output_path, type_model, all_data, name_features=None, target_variables=None):
    
    mean_importances_df_lwp = read_dataframe_lwp_nd_fi_shap(variable_target="lwp", 
                                                           output_path=output_path, 
                                                           type_model=type_model, 
                                                           all_data=all_data)
    
    mean_importances_df_ndmax = read_dataframe_lwp_nd_fi_shap(variable_target="Nd_max", 
                                                           output_path=output_path, 
                                                           type_model=type_model, 
                                                           all_data=all_data)
    
    if all_data == "all_features":
        merged_df_all_features = pd.merge(mean_importances_df_ndmax, mean_importances_df_lwp, on='Feature', how='inner')  
        merged_df_all_features = merged_df_all_features.set_index('Feature') 
        merged_df_all_features.index.name = None 

    elif all_data == "most_importances":
        mean_importances_df_lwp = mean_importances_df_lwp.set_index('Feature')  
        mean_importances_df_lwp.index.name = None 
        mean_importances_df_ndmax = mean_importances_df_ndmax.set_index('Feature')  
        mean_importances_df_ndmax.index.name = None 
    
        merged_df_all_features = mergue_most_important_with_all(mean_importances_df_ndmax=mean_importances_df_ndmax,
                                                                mean_importances_df_lwp=mean_importances_df_lwp,
                                                                name_features= name_features, 
                                                                target_variables=target_variables)

    return merged_df_all_features


def feature_importance_lwp_nd_normalize_median(path):

    file_path_lwp = f'{path}_lwp_rankings_mean_median.csv'
    fi_lwp_nn  = pd.read_csv(file_path_lwp, index_col=0)
    
    file_path_ndmax = f'{path}_Nd_max_rankings_mean_median.csv'
    fi_ndmax_nn = pd.read_csv(file_path_ndmax, index_col=0)

    # ---- lwp
    # Normalize the median rankings to a range of 0–1
    fi_lwp_nn["Normalized_Median"] = (fi_lwp_nn["Median Ranking"] - fi_lwp_nn["Median Ranking"].min()) / (
        fi_lwp_nn["Median Ranking"].max() - fi_lwp_nn["Median Ranking"].min()
    )
    # I inverted because 1 is most significative
    fi_lwp_nn["Inverted_Normalized"] = 1 - fi_lwp_nn["Normalized_Median"]
    NN_feature_importance_lwp = fi_lwp_nn[["Inverted_Normalized"]].copy()
    # NN_feature_importance_lwp = fi_lwp_nn[["Normalized_Median"]].copy()
    NN_feature_importance_lwp.columns = ["$L$"]

    # ---- Ndmax
    fi_ndmax_nn["Normalized_Median"] = (fi_ndmax_nn["Median Ranking"] - fi_ndmax_nn["Median Ranking"].min()) / (
        fi_ndmax_nn["Median Ranking"].max() - fi_ndmax_nn["Median Ranking"].min()
    )
    fi_ndmax_nn["Inverted_Normalized"] = 1 - fi_ndmax_nn["Normalized_Median"]
    NN_feature_importance_ndmax = fi_ndmax_nn[["Inverted_Normalized"]].copy()
    NN_feature_importance_ndmax.columns = ["$N_{\mathrm{d},\max}$"]

    return NN_feature_importance_ndmax, NN_feature_importance_lwp

    
def feature_importance_lwp_nd_mean_median(path, avr_type="Mean"):
    file_path_lwp = f'{path}_lwp_mean_median_along_methods.csv'
    fi_lwp_nn  = pd.read_csv(file_path_lwp, index_col=0)
    fi_lwp_nn.index.name = None # Partial

    NN_feature_importance_lwp = fi_lwp_nn[[avr_type]].copy()
    NN_feature_importance_lwp.columns = ["$L$"]

    file_path_ndmax = f'{path}_Nd_max_mean_median_along_methods.csv'
    fi_ndmax_nn = pd.read_csv(file_path_ndmax, index_col=0)
    fi_ndmax_nn.index.name = None # Partial
    NN_feature_importance_ndmax = fi_ndmax_nn[[avr_type]].copy()
    NN_feature_importance_ndmax.columns = ["$N_{\mathrm{d},\max}$"]

    return NN_feature_importance_ndmax, NN_feature_importance_lwp

    
def feature_importance_lwp_nd_normalize_mean(path):
    file_path_lwp = f'{path}_lwp_rankings_mean_median.csv'
    fi_lwp_nn  = pd.read_csv(file_path_lwp, index_col=0)
    
    file_path_ndmax = f'{path}_Nd_max_rankings_mean_median.csv'
    fi_ndmax_nn = pd.read_csv(file_path_ndmax, index_col=0)

    # ---- lwp
    # Normalize the median rankings to a range of 0–1
    fi_lwp_nn["Normalized_Mean"] = (fi_lwp_nn["Mean Ranking"] - fi_lwp_nn["Mean Ranking"].min()) / (
        fi_lwp_nn["Mean Ranking"].max() - fi_lwp_nn["Mean Ranking"].min()
    )
    # I inverted because 1 is most significative
    fi_lwp_nn["Inverted_Normalized"] = 1 - fi_lwp_nn["Normalized_Mean"]
    NN_feature_importance_lwp = fi_lwp_nn[["Inverted_Normalized"]].copy()
    NN_feature_importance_lwp.columns = ["LWP"]

    # ---- Ndmax
    fi_ndmax_nn["Normalized_Mean"] = (fi_ndmax_nn["Mean Ranking"] - fi_ndmax_nn["Mean Ranking"].min()) / (
        fi_ndmax_nn["Mean Ranking"].max() - fi_ndmax_nn["Mean Ranking"].min()
    )
    fi_ndmax_nn["Inverted_Normalized"] = 1 - fi_ndmax_nn["Normalized_Mean"]
    NN_feature_importance_ndmax = fi_ndmax_nn[["Inverted_Normalized"]].copy()
    NN_feature_importance_ndmax.columns = ["$N_{d,\\ max}$"]

    return NN_feature_importance_ndmax, NN_feature_importance_lwp


def avr_and_plot_dataframes(dataframes, output_path):
    """
    Sums multiple dataframes by extracting the first variable from column names, merges duplicate variables, and plots bar charts.

    Args:
        dataframes (list of pd.DataFrame): List of DataFrames with feature index and two columns.

    Returns:
        pd.DataFrame: average DataFrame with cleaned variable names
    """
    
    titles_font_size = 19

    renamed_dfs = []
    for df in dataframes:
        df = df.copy()
        df.columns = [col.split(" | ")[0] if " | " in col else col for col in df.columns] 
        renamed_dfs.append(df)

    print(" =================================================== ")
    mean_df = sum(renamed_dfs) / len(renamed_dfs)
    
    df_filtered = mean_df.dropna(how="all")
    df_filtered.plot(kind='bar', figsize=(10, 5), color=['#4e4e50', '#ff6b6b'])
    plt.xlabel("Channels", fontsize=titles_font_size)
    plt.ylabel("Mean of importances", fontsize=titles_font_size)
    plt.legend(fontsize=titles_font_size)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.xticks(fontsize=titles_font_size-2)
    plt.yticks(fontsize=titles_font_size-2)
    name_output = f"{output_path}.pdf" 
    plt.savefig(name_output, bbox_inches='tight') 
    print(f" ------------ Saved in {name_output} ----------------------------")
    plt.close()
    return mean_df


def all_methods_mean_and_plot_dataframes(dataframes, output_path):
    titles_font_size = 19
    normalized_dfs = []

    
    for df in dataframes:
        df = df.copy()
        df.columns = [col.split(" | ")[0] if " | " in col else col for col in df.columns]  # E
        abs_df = df.abs()
        df_abs_normalized = (abs_df - abs_df.min()) / (abs_df.max() - abs_df.min())

        normalized_dfs.append(df_abs_normalized)

    print(" =================================================== ")
    mean_df = sum(normalized_dfs) / len(normalized_dfs)
    
    mean_df.to_csv(f"{output_path}.csv")

    combined_df = pd.concat(normalized_dfs, axis=1)

    mean_df.plot(kind='bar', figsize=(20, 5), color=['#4e4e50', '#ff6b6b'])
    plt.xlabel("Channels", fontsize=titles_font_size)
    plt.ylabel("Mean of importances", fontsize=titles_font_size)
    plt.legend(fontsize=titles_font_size)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.xticks(fontsize=titles_font_size-2)
    plt.yticks(fontsize=titles_font_size-2)
    name_output = f"{output_path}.pdf" 
    plt.savefig(name_output, bbox_inches='tight')  
    print(f" ------------ Saved in {name_output} ----------------------------")
    plt.close()
    return mean_df, combined_df


def print_top_feature_importances(feature_importance, number_fi_lwp, number_fi_ndmax, type_model):
    print(f" ================== top features model: {type_model} ==================")
    for col in feature_importance.columns:
        if col=="$L$":
            top_n = number_fi_lwp
        if col=="$N_{\mathrm{d},\max}$":
            top_n = number_fi_ndmax
        top_features = feature_importance[col].sort_values(ascending=False).head(top_n)
        print(f"Top {top_n} features for column '{col}':")
        print(top_features)

        top_feature_numbers = [feat.split(':')[0].strip() for feat in top_features.index]
        # Join the numbers with a comma
        numbers_str = ", ".join(top_feature_numbers)
        print(f"Top {top_n} feature numbers for column '{col}': {numbers_str}")
        print("\n")



def main():
    parser = argparse.ArgumentParser(description='Heatmaps')
    arg = parser.add_argument
    arg('--output_path', type=str, default="/work/bb1036/b381362/output/final_results/")
    arg('--all_data', type=str, default="all_features", help='number of flods to split the dataset')

    args = parser.parse_args()
    output_path = args.output_path
    all_data = args.all_data

    
    output_folder = f"{output_path}/methods_with_mean_NN"
    os.makedirs(output_folder, exist_ok=True)  
 
    output_folder = f"{output_path}/methods_with_median_NN"
    os.makedirs(output_folder, exist_ok=True)  
 
    if all_data == "all_features":
        # ========================================= All features ================================================
        # ------------------------------- SC --------------------------------
        file_path_latest = f'{output_path}/correlation/spearman_correlation_LWP-Ndmax-channels_only_clouds_3days--_only_2variables_filter.csv'
        sc_data = pd.read_csv(file_path_latest, index_col=0) .iloc[2:,:2]
        sc_data = sc_data.rename(columns={'LWP': '$L$', '$N_{d,\\ max}$': '$N_{\mathrm{d},\max}$'})
    
        # ------------------------------- SPC --------------------------------
        file_path_latest = f'{output_path}/spearman_partial_correlation/spearman_partial_correlation_LWP-Ndmax-channels_only_clouds_3days-filter.csv'
        spc_data = pd.read_csv(file_path_latest, index_col=0) 
        spc_data.index.name = None # Partial
        spc_data = spc_data.rename(columns={'$N_{d,\ max}$ | LWP': '$N_{\mathrm{d},\max}$ | $L$', 'LWP | $N_{d,\ max}$': '$L$ | $N_{\mathrm{d},\max}$'})
    
        # spc_data
        # ------------------------------- MI --------------------------------
        file_path_latest = f'{output_path}/mutual_information/mutual_information_target_LWP-Ndmax-channels_only_clouds_3days--_filter.csv'
        mi_data = pd.read_csv(file_path_latest, index_col=0) 
        mi_data = mi_data.rename(columns={'LWP': '$L$', '$N_{d,\\ max}$': '$N_{\mathrm{d},\max}$'})
    
        # ------------------------ RF all features -------------------------------
        RF_feature_importance = create_dataframe_lwp_nd_fi(output_path=output_path,
                                                           type_model="RF",
                                                           all_data="all_features")
        RF_feature_importance.to_csv(f"{output_path}/RF_feature_importance_all_features.csv")
    
        RF_feature_importance = create_dataframe_lwp_nd_fi_shap(output_path=output_path,
                                                           type_model="RF",
                                                           all_data="all_features")
        RF_feature_importance.to_csv(f"{output_path}/RF_feature_importance_all_features_shap.csv")

        print_top_feature_importances(feature_importance=RF_feature_importance,
                                      number_fi_lwp=4, 
                                      number_fi_ndmax=7, 
                                      type_model="RF")

        # ------------------------ XGBoost all features -------------------------------
        XGBoost_feature_importance = create_dataframe_lwp_nd_fi(output_path=output_path,
                                                           type_model="XGBoost",
                                                           all_data="all_features")
        
        XGBoost_feature_importance.to_csv(f"{output_path}/XGBoost_feature_importance_all_features.csv")
    
        XGBoost_feature_importance = create_dataframe_lwp_nd_fi_shap(output_path=output_path,
                                                           type_model="XGBoost",
                                                           all_data="all_features")
        
        XGBoost_feature_importance.to_csv(f"{output_path}/XGBoost_feature_importance_all_features_shap.csv")
        print_top_feature_importances(feature_importance=XGBoost_feature_importance,
                                  number_fi_lwp=4, 
                                  number_fi_ndmax=8, 
                                  type_model="XGBoost")

        # ------------------------ NN all features -------------------------------
    
        NN_feature_importance_ndmax, NN_feature_importance_lwp = feature_importance_lwp_nd_mean_median(path=f'{output_path}/NN_feature_importance_all_features/all_features',
                                                                                                avr_type="Mean") 
        NN_feature_importance_mean = pd.concat([NN_feature_importance_ndmax, NN_feature_importance_lwp], axis=1)
        name_output_NN_fi = f"{output_path}/NN_feature_importance_all_features_mean.csv"
        NN_feature_importance_mean.to_csv(name_output_NN_fi)
        print(f" ------------ Saved in {name_output_NN_fi} ----------------------------")
        print_top_feature_importances(feature_importance=NN_feature_importance_mean,
                                  number_fi_lwp=3, 
                                  number_fi_ndmax=7, 
                                  type_model="NN")

    
        NN_feature_importance_ndmax, NN_feature_importance_lwp = feature_importance_lwp_nd_mean_median(path=f'{output_path}/NN_feature_importance_all_features/all_features',
                                                                                                avr_type="Median")
        NN_feature_importance_median = pd.concat([NN_feature_importance_ndmax, NN_feature_importance_lwp], axis=1)
        
        NN_feature_importance_median.to_csv(f"{output_path}/NN_feature_importance_all_features_median.csv")
        
        plot_heatmap_all_features(output_path=f"{output_path}/methods_with_mean_NN",
                                  sc_data=sc_data, 
                                  spc_data=spc_data, 
                                  mi_data=mi_data,
                                  RF_feature_importance=RF_feature_importance,
                                  XGBoost_feature_importance=XGBoost_feature_importance,
                                  NN_feature_importance=NN_feature_importance_mean)

     
        plot_heatmap_all_features(output_path=f"{output_path}/methods_with_median_NN",
                                  sc_data=sc_data, 
                                  spc_data=spc_data, 
                                  mi_data=mi_data,
                                  RF_feature_importance=RF_feature_importance,
                                  XGBoost_feature_importance=XGBoost_feature_importance,
                                  NN_feature_importance=NN_feature_importance_median)

        mean_all_method_df, normalized_dfs = all_methods_mean_and_plot_dataframes(dataframes=[NN_feature_importance_mean],
                                        output_path=f"{output_path}/NN_FI_with_mean_NN")
        
        # ================================ all methods ================================
        mean_all_method_df, normalized_dfs = all_methods_mean_and_plot_dataframes(dataframes=[sc_data, spc_data, mi_data, RF_feature_importance, XGBoost_feature_importance, NN_feature_importance_mean],
                                        output_path=f"{output_path}/all_methods_mean_with_mean_NN")

        plot_heatmap_normalized(output_path=f"{output_path}/methods_with_mean_NN",
                     sc_data=normalized_dfs.iloc[:, :2], 
                     spc_data=normalized_dfs.iloc[:, 2:4], 
                     mi_data=normalized_dfs.iloc[:, 4:6],
                     RF_feature_importance=normalized_dfs.iloc[:, 6:8],
                     XGBoost_feature_importance=normalized_dfs.iloc[:, 8:10],
                     NN_feature_importance=normalized_dfs.iloc[:, 10:12],
                     all_methods_mean=mean_all_method_df)



    elif all_data == "most_importances":
        # ========================================= Most important features ================================================
        # ------------------------ RF most features importance --------------------------
        RF_feature_importance = create_dataframe_lwp_nd_fi(output_path=output_path,
                                                           type_model="RF",
                                                           all_data="all_features")
    
        name_features = RF_feature_importance.index.values
        target_variables = RF_feature_importance.columns.values
    
        RF_most_importance_feature = create_dataframe_lwp_nd_fi(output_path,
                                        type_model="RF",
                                        all_data= "most_importances",
                                        name_features=name_features,
                                        target_variables=target_variables)
        
        RF_most_importance_feature.to_csv(f"{output_path}/RF_feature_importance_most_importances.csv")
    
        # ------------------------ XGBoost most features importance --------------------------
        XGBoost_most_importance_feature = create_dataframe_lwp_nd_fi(output_path,
                                        type_model="XGBoost",
                                        all_data= "most_importances",
                                        name_features=name_features,
                                        target_variables=target_variables)
        
        XGBoost_most_importance_feature.to_csv(f"{output_path}/XGBoost_feature_importance_most_importances.csv")
    
        
        # ---------------------- NN most important feature ----------------------------
        NN_feature_importance_ndmax, NN_feature_importance_lwp = feature_importance_lwp_nd_mean_median(path=f'{output_path}/NN_feature_importance_most_importances/most_importances',
                                                                                                      avr_type="Mean") #feature_importance_lwp_nd_normalize_mean
        NN_most_importance_feature_mean = mergue_most_important_with_all(mean_importances_df_ndmax=NN_feature_importance_ndmax,
                                                                    mean_importances_df_lwp=NN_feature_importance_lwp,
                                                                    name_features=name_features, 
                                                                    target_variables=target_variables)
        NN_most_importance_feature_mean.to_csv(f"{output_path}/NN_feature_importance_most_importances_mean.csv")
    
        
        NN_feature_importance_ndmax, NN_feature_importance_lwp = feature_importance_lwp_nd_mean_median(path=f'{output_path}/NN_feature_importance_most_importances/most_importances',
                                                                                                      avr_type="Median") 

        NN_most_importance_feature_median = mergue_most_important_with_all(mean_importances_df_ndmax=NN_feature_importance_ndmax,
                                                                    mean_importances_df_lwp=NN_feature_importance_lwp,
                                                                    name_features=name_features, 
                                                                    target_variables=target_variables)
        NN_most_importance_feature_median.to_csv(f"{output_path}/NN_feature_importance_most_importances_median.csv")
    

        # # ============================ join  Heatmaps ============================
        plot_heatmap_mif(output_path=f"{output_path}/methods_with_mean_NN",
                     RF_most_importance_feature=RF_most_importance_feature,
                     XGBoost_most_importance_feature=XGBoost_most_importance_feature,
                     NN_most_importance_feature=NN_most_importance_feature_mean)
    
       
        plot_heatmap_mif(output_path=f"{output_path}/methods_with_median_NN",
                                  RF_most_importance_feature=RF_most_importance_feature,
                                  XGBoost_most_importance_feature=XGBoost_most_importance_feature,
                                  NN_most_importance_feature=NN_most_importance_feature_median)

    # ====================== feature importances of each model  ------------- plot feature importances
        summed_df = avr_and_plot_dataframes(dataframes=[NN_most_importance_feature_mean],
                                            output_path=f"{output_path}/methods_with_mean_NN/MI_features_importances_NN_mean") 
        summed_df = avr_and_plot_dataframes(dataframes=[NN_most_importance_feature_median],
                                            output_path=f"{output_path}/methods_with_median_NN/MI_features_importances_NN_median") 
    
    
        summed_df = avr_and_plot_dataframes(dataframes=[RF_most_importance_feature],
                                            output_path=f"{output_path}/MI_features_importances_RF") 
        summed_df = avr_and_plot_dataframes(dataframes=[XGBoost_most_importance_feature],
                                            output_path=f"{output_path}/MI_features_importances_XGBoost") # 




if __name__ == '__main__':
    main()
