
import numpy as np
import pandas as pd
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import copy      
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shap
import sys

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from math import sqrt
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.functions import load_dataframe_from_netcdf, select_channels
from inverse_model_NN import MyModel


def permutation_feature_importance(
    model, X_val, y_val, metric=mean_squared_error, n_repeats=5, batch_size=512, device="cuda"
):
    """
    Computes permutation feature importance using standardized validation data, supports batching and GPU.
    """
    
    model = model.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Compute the baseline score
    baseline_score = metric(y_val.cpu().numpy(), model(X_val).detach().cpu().numpy())

    importances = np.zeros(X_val.shape[1])  

    for i in range(X_val.shape[1]):
        X_val_permuted = X_val.clone()

        for _ in range(n_repeats):
            X_val_permuted[:, i] = X_val[:, i][torch.randperm(X_val.size(0), device=device)]

            if batch_size:
                permuted_score = 0.0
                num_batches = (X_val.size(0) + batch_size - 1) // batch_size

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, X_val.size(0))
                    batch_X = X_val_permuted[start_idx:end_idx]
                    batch_y = y_val[start_idx:end_idx]

                    batch_preds = model(batch_X).detach()
                    permuted_score += metric(batch_y.cpu().numpy(), batch_preds.cpu().numpy())

                permuted_score /= num_batches
            else:
                permuted_score = metric(y_val.cpu().numpy(), model(X_val_permuted).detach().cpu().numpy())

            importances[i] += permuted_score - baseline_score

        importances[i] /= n_repeats 

    return importances


def integrated_gradients_importance(model, X_val, target=0, batch_size=512, device="cuda"):
    """
    Computes feature importance using Integrated Gradients on standardized data,
    with GPU and batch processing support.
    
    Args:
        model: PyTorch model
        X_val: Input tensor (validation data)
        target: Target class index (for classification) or output neuron (for regression)
        batch_size: Number of samples per batch (None processes all at once)
        device: "cuda" for GPU or "cpu"
        
    Returns:
        global_feature_importance: Averaged feature importance across all samples
    """
    
    model = model.to(device)
    X_val = X_val.to(device)

    ig = IntegratedGradients(model)  

    attributions_list = []  

    if batch_size:
        num_batches = (X_val.size(0) + batch_size - 1) // batch_size  

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, X_val.size(0))

            batch_X = X_val[start_idx:end_idx]  
            batch_attr, _ = ig.attribute(batch_X, target=target, return_convergence_delta=True)
            attributions_list.append(batch_attr)

        attributions = torch.cat(attributions_list, dim=0)
    else:
        attributions, _ = ig.attribute(X_val, target=target, return_convergence_delta=True)

    global_feature_importance = torch.mean(torch.abs(attributions), dim=0).detach().cpu().numpy()

    return global_feature_importance


def normalize_importance(importance_scores):
    """
    Normalizes feature importance scores to range between 0 and 1.
    """
    
    min_val = np.min(importance_scores)
    max_val = np.max(importance_scores)
    return (importance_scores - min_val) / (max_val - min_val)


def calculate_and_compare_feature_importance(model, X_val, y_val, feature_names, X_train, device):
    """
    Calculates feature importance using various methods on standardized data and compares them by normalizing the results.
    
    Args:
    - model: Trained model.
    - X_val, y_val, X_train: Validation and training data (standardized).
    
    Returns:
    - DataFrame with normalized feature importance scores.
    """
    
    model.eval()    
    
    
    perm_importance = permutation_feature_importance(model=model, 
                                                     X_val=X_val, 
                                                     y_val=y_val, 
                                                     device=device)
    
    ig_importance = integrated_gradients_importance(model=model, 
                                                    X_val=X_val, 
                                                    device=device)

    perm_importance_normalized = normalize_importance(perm_importance)
    ig_importance_normalized = normalize_importance(ig_importance)
    results =  {
        "Feature": feature_names,
        "Permutation Importance": perm_importance,
        "Integrated Gradients": ig_importance,
    }
    print(results)
    df_importances =  pd.DataFrame({
        "Feature": feature_names,
        "Permutation Importance": perm_importance,
        "Integrated Gradients": ig_importance,
    })

    df_importances = df_importances.set_index('Feature')

    return df_importances


def plot_feature_importance_comparison(df_feature_importances, methods, figure_name_file, title="Feature Importance Comparison"):
    """
    Plots the feature importance scores for each method in a single figure.
    
    Args:
    - df_feature_importances: DataFrame containing feature importance scores from various methods.
    - methods: List of column names corresponding to the feature importance methods to be plotted.
    - title: The title of the plot.
    """
    
    titles_font_size = 20
    subtitles_font_size = 18
    axes_font_size = 16
    
    plt.figure(figsize=(12, len(df_feature_importances) * 0.5 + 2))
    
    n_features = len(df_feature_importances)
    
    bar_width = 0.2
    index = np.arange(n_features)
    
    for i, method in enumerate(methods):
        plt.barh(index + i * bar_width, df_feature_importances[method], bar_width, label=method)
    
    plt.yticks(index + bar_width, df_feature_importances['Feature'], fontsize=axes_font_size)
    
    plt.xlabel('Importance Score')
    plt.title(title, fontsize=titles_font_size)
    plt.legend(loc='best', fontsize=subtitles_font_size)
    
    plt.tight_layout()
    plt.savefig(figure_name_file, dpi=150)
    plt.show()



def get_heatmap_captum(df_data_importances, output_file_name, cmap='RdBu'):
    titles_font_size = 17
    subtitles_font_size = 16
    labels_font_size = 15 
    axes_font_size = 14
    
    
    df_data_importances.index.name = None 
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(4, 8))
    sns.heatmap(
            df_data_importances[[df_data_importances.columns[0]]],
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[0],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels=[df_data_importances.columns[0]],
        )
    axes[0].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[0].tick_params(axis="y", labelsize=labels_font_size)
   
    sns.heatmap(
            df_data_importances[[df_data_importances.columns[1]]],
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[1],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels=[df_data_importances.columns[1]],
            yticklabels=False  # Hide y-axis labels
        )
    axes[1].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
    sns.heatmap(
            df_data_importances[[df_data_importances.columns[2]]],
            cmap='RdBu',
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[2],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels=[df_data_importances.columns[2]],
            yticklabels=False  
        )
    axes[2].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
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
            xticklabels=[df_data_importances.columns[3]],
            yticklabels=False  
        )
    axes[3].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    plt.tight_layout(pad=0.5)  

    plt.savefig(f"{output_file_name}.png", bbox_inches='tight')  

    plt.close


def get_heatmap(df_data_importances, output_file_name, cmap='RdBu'):
    titles_font_size = 17
    subtitles_font_size = 16
    labels_font_size = 15 # 16
    axes_font_size = 14
    
    
    df_data_importances.index.name = None 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3, 8))
    sns.heatmap(
            df_data_importances[[df_data_importances.columns[0]]],
            cmap=cmap,
            # cmap='RdBu',
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[0],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels=[df_data_importances.columns[0]],
        )
    axes[0].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    axes[0].tick_params(axis="y", labelsize=labels_font_size)

    sns.heatmap(
            df_data_importances[[df_data_importances.columns[1]]],
            cmap=cmap,
            # cmap='Blues',
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
            ax=axes[1],
            linecolor='black',
            annot_kws={"size": 13},
            xticklabels=[df_data_importances.columns[1]],
            yticklabels=False  # Hide y-axis labels
        )
    axes[1].tick_params(axis="x", labelsize=labels_font_size, rotation=90)
    
    plt.tight_layout(pad=0.5)  

    plt.savefig(f"{output_file_name}.png", bbox_inches='tight')  

    plt.close



def main():
    parser = argparse.ArgumentParser(description='Heatmaps')
    arg = parser.add_argument

    # Add the arguments
    arg('--output_path', type=str, default="/work/bb1036/b381362/output/final_results/")
    arg('--fold-num', type=int, default=1, help='n k-fold to run')  

    arg('--variable_target', type=str, default="lwp", help='"Re, COT, LWP, Ndmax" List of the variables to use.')
    arg('--type_plot', type=str, default="captum", help='"captum, my_feature_importances" List of the variables to use.')
    arg('--path_model_file', type=str, default="/work/bb1036/b381362/output/model.pth", help='path of the  model')
    arg('--path_dataframes_scaler', type=str, default="/work/bb1036/b381362/", help='path where is the dataset save as dataframes and scaler')
    arg('--all_data', type=str, default="all_features", help='number of flods to split the dataset')  
 
    args = parser.parse_args()
    output_path = args.output_path
    fold_num = args.fold_num
    variable_target = args.variable_target
    # path_models = args.path_models
    path_model_file = args.path_model_file
    type_plot = args.type_plot
    path_dataframes_scaler = args.path_dataframes_scaler
 
    all_data = args.all_data
    
   
    print(" ========================================== ")
    print("Arguments received:")
    for arg in vars(args):
         print(f"{arg}: {getattr(args, arg)}")
    
    channel_relate_clouds = select_channels(all_data=all_data,
                                            variable_target=variable_target,
                                            type_model="NN")

    print(f" ==================== {all_data} used  ==================== ")

    with open(f"{path_dataframes_scaler}/val_data_k_fold_{fold_num}.pkl", 'rb') as f:
        times_data = pickle.load(f)
    
        
    df_icon_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_train', fold_num)
    df_icon_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    df_icon_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    df_ref_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_train', fold_num)
    df_ref_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)
    df_ref_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)
    
    loaded_index_all_test = np.load(f'{path_dataframes_scaler}/index_only_clouds_all_val_{fold_num}.npy', allow_pickle=True)
    
    
    X_features_val = df_ref_val.loc[:,channel_relate_clouds]
    y_labels_val = df_icon_val[[variable_target]]
    
    X_features_train = df_ref_train.loc[:,channel_relate_clouds]
    y_labels_train = df_icon_train[[variable_target]]

    subset_size = 1024
    X_features_val = X_features_val.sample(n=subset_size, random_state=42)
    y_labels_val = y_labels_val.loc[X_features_val.index]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MyModel(input_size=X_features_val.shape[1], 
                    output_size=y_labels_val.shape[1]).to(device)
    
    model.load_state_dict(torch.load(path_model_file, map_location=device))
    model.to(device)
    model.eval()  
    
    x_test_np_train = X_features_train.to_numpy(dtype=float)
    X_tensor_train = torch.tensor(x_test_np_train, dtype=torch.float32).to(device)
    y_test_np_train = y_labels_train.to_numpy(dtype=float)
    y_tensor_train = torch.tensor(y_test_np_train, dtype=torch.float32).to(device)

    # -------------------subset train
    X_features_train_subset = X_features_train.sample(n=subset_size, random_state=42)
    x_test_np_train_subset = X_features_train_subset.to_numpy(dtype=float)
    X_tensor_train_subset = torch.tensor(x_test_np_train_subset, dtype=torch.float32).to(device)
    
    y_labels_train_subset = y_labels_train.loc[X_features_train_subset.index]
    y_test_np_train_subset = y_labels_train_subset.to_numpy(dtype=float)
    y_tensor_train_subset = torch.tensor(y_test_np_train_subset, dtype=torch.float32).to(device)
    # -------------------

    
    x_test_np_val = X_features_val.to_numpy(dtype=float)
    y_test_np_val = y_labels_val.to_numpy(dtype=float)
    X_tensor_val = torch.tensor(x_test_np_val, dtype=torch.float32).to(device)
    y_tensor_val = torch.tensor(y_test_np_val, dtype=torch.float32).to(device)

    
    
    if variable_target=='lwp' or variable_target=='LWP':
        axis_name = 'LWP'
    elif variable_target=='Nd_max' or variable_target==r'$N_{d,\ max}$':
        axis_name = '$N_{d,\ max}$'

    # https://captum.ai/tutorials/House_Prices_Regression_Interpret
    if type_plot == "captum":
        X_train = X_tensor_train_subset
        X_test = X_tensor_train 
        batch_size = 512                  # Set your batch size
        num_batches = len(X_test) // batch_size + (1 if len(X_test) % batch_size != 0 else 0)

        ig = IntegratedGradients(model)
        ig_nt = NoiseTunnel(ig)
        dl = DeepLift(model)
        gs = GradientShap(model)

                
        ig_attributions = []
        ig_nt_attributions = []
        dl_attributions = []
        gs_attributions = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_test))
            batch = X_test[start_idx:end_idx] 
        
            batch = batch.to(next(model.parameters()).device)
        
            ig_attr = ig.attribute(batch)  # Integrated Gradients
            ig_nt_attr = ig_nt.attribute(batch)  # Noise Tunnel
            dl_attr = dl.attribute(batch)  # DeepLIFT
            gs_attr = gs.attribute(batch, baselines=torch.zeros_like(batch))  # GradientShap
        
            ig_attributions.append(ig_attr.cpu())
            ig_nt_attributions.append(ig_nt_attr.cpu())
            dl_attributions.append(dl_attr.cpu())
            gs_attributions.append(gs_attr.cpu())
        
        ig_attr_test = torch.cat(ig_attributions, dim=0)
        ig_nt_attr_test = torch.cat(ig_nt_attributions, dim=0)
        dl_attr_test = torch.cat(dl_attributions, dim=0)
        gs_attr_test = torch.cat(gs_attributions, dim=0)

        ig_attr_test_sum = ig_attr_test.cpu().detach().numpy().sum(0)
        ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)
        
        ig_nt_attr_test_sum = ig_nt_attr_test.cpu().detach().numpy().sum(0)
        ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)
        
        dl_attr_test_sum = dl_attr_test.cpu().detach().numpy().sum(0)
        dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)
        
        gs_attr_test_sum = gs_attr_test.cpu().detach().numpy().sum(0)
        gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)
        

        df_importances_norm =  pd.DataFrame({
            "Feature": np.array(X_features_train.columns),
            "Int Grads": ig_attr_test_norm_sum,
            "Int Grads w/SmoothGrad": ig_nt_attr_test_norm_sum,
            "DeepLift": dl_attr_test_norm_sum,
            "GradientSHAP": gs_attr_test_norm_sum,
        })

          
    
        df_importances_norm = df_importances_norm.set_index('Feature')
        
        
        output_file_name = f"{output_path}/FI_NN_captum_{variable_target}_{fold_num}"
        
        df_importances_norm = df_importances_norm.round(3)
        df_importances_norm.to_csv(f'{output_file_name}.csv', index=True)


        get_heatmap_captum(df_data_importances=df_importances_norm,
                          output_file_name=output_file_name)



    elif type_plot == "my_feature_importances":
        X_train = X_tensor_train_subset
        X_test = X_tensor_train 
        y_test = y_tensor_train 
        
        print(f"---------------Shape X_test {np.shape(X_test)} X_train {np.shape(X_train)}---------------------")
        
        df_feature_importances = calculate_and_compare_feature_importance(model=model, 
                                                                  feature_names =channel_relate_clouds,
                                                                  X_val=X_test, 
                                                                  y_val=y_test, 
                                                                  X_train=X_train,
                                                                          device=device
                                                                  )
    

        output_file_name = f"{output_path}/FI_NN_others_{variable_target}_{fold_num}_all_train_no_shape"

        df_feature_importances = df_feature_importances.round(3)
        df_feature_importances.to_csv(f'{output_file_name}.csv', index=True)

        
        get_heatmap(df_data_importances=df_feature_importances, 
                    output_file_name=output_file_name,
                    cmap='Blues')


    elif type_plot == "shap_values":
        name_saving_files = f"NN_{variable_target}_k_fold_{fold_num}"

        X_test = X_tensor_train_subset 
        y_test = y_tensor_train_subset         
        
        
        print(f"---------------Shape X_test {np.shape(X_test)} ---------------------")
        
    
        model.eval()  
        explainer = shap.DeepExplainer(model, X_test)  
        
        shap_values = explainer.shap_values(X_test)
    
        # mean_shap_values = shap_values.mean(axis=0)
        mean_shap_values = abs(shap_values).mean(axis=0)
        
        feature_names = X_features_train.columns
  
        shap_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Fold": fold_num,
            "Mean_SHAP_Value": mean_shap_values
        })
        
        file_path = f"{output_path}/{all_data}_shap_{name_saving_files}_s{subset_size}_train.csv"
        shap_importance_df.to_csv(file_path, index=False)
   
        if os.path.exists(file_path):
            print(f"File saved successfully at: {file_path}")
        else:
            print(f"Failed to save the file at: {file_path}")
        
        X_test_np = X_test.detach().cpu().numpy()  
        
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_test_np, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary for {axis_name} - Fold {fold_num}")
        summary_beeswarm_path = f"{output_path}/shap_summary_beeswarm_{name_saving_files}_train.png"
        plt.savefig(summary_beeswarm_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_test_np, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Summary for {axis_name} - Fold {fold_num}")
        summary_bar_path = f"{output_path}/{all_data}_shap_summary_bar_{name_saving_files}_train.png"
        plt.savefig(summary_bar_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        # -------------------------------------------------------------------------
        # ---------------------------------
        # Plot SHAP dependence for the top feature
        top_feature_idx = np.argsort(mean_shap_values)[-1] 
        top_feature_name = feature_names[top_feature_idx]
        
        cols = 3
        rows = (len(feature_names) + cols - 1) // cols  
        fig, axes = plt.subplots(rows, cols, figsize=(14, 20))  
        axes = axes.flatten()
    
        for i, feature in enumerate(feature_names):
            plt.sca(axes[i])  
            shap.dependence_plot(
                i,
                shap_values,
                X_test_np,
                feature_names=feature_names,
                ax=axes[i], 
                show=False
            )
            if i % cols == 0: 
                axes[i].set_ylabel(f"SHAP value\n(impact on {axis_name})")
            else:
                axes[i].set_ylabel('')  
            
        plt.title(f"SHAP Dependence Plot Channel: {feature}")
        dependence_plot_path = f"{output_path}/{all_data}_shap_dependence_{name_saving_files}_all_ch_train.png"
        plt.tight_layout()  
        plt.savefig(dependence_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        print("SHAP plots saved successfully.")     
    

if __name__ == '__main__':
    main()


