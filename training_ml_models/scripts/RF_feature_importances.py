import numpy as np
import pandas as pd
import pickle
import argparse
import xarray as xr
import xgboost as xgb
import os 
import joblib
import matplotlib.pyplot as plt
import sys
import shap
import scipy.cluster.hierarchy as sch

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.functions import load_dataframe_from_netcdf, select_channels

 
def print_configuration(args):
    configuration = f"""
    Configuration:
    - Model Type: {args.type_model}
    - Learning Rate: {args.lr}
    - Number of Epochs: {args.n_epochs}
    - Batch size: {args.batch_size}
    - Channel Number List: {args.channel_number_list}
    - Fold Number: {args.fold_num}
    - Dataser Path: {args.path_dataframes_pca_scaler}
    - Output Path: {args.path_models}
    """
    print(configuration)


def metric_calculation_all_data(outputs, targets):
    """
    Evaluate R-squared (R2) and Root Mean Squared Error (RMSE) for each output of the model and their uniform averages.

    Args:
    - outputs (torch.Tensor): The model's predictions, expected to be a tensor of shape (n_samples, n_outputs).
    
    # Target is only one Serie
    - targets (torch.Tensor): The ground truths (actual values), expected to be a tensor of shape (n_samples, n_outputs).

    Returns:
    - metrics (dict): A dictionary containing RMSE and R2 for all output in averaged and optionally for each channel.
    """
       
    metrics = {}
    mse_avg = mean_squared_error(targets, outputs, multioutput='uniform_average')
    r2_avg = r2_score(targets, outputs, multioutput='uniform_average')
    mae_avg = mean_absolute_error(targets, outputs, multioutput='uniform_average')

    metrics = {
        "R-squared (R2)": r2_avg,
        "Mean Squared Error (MSE)":mse_avg,
        "Mean Absolute Error (MAE)":mae_avg,
        "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(targets, outputs, multioutput='uniform_average')),
    }

    return metrics


def load_model(path_model_file):
    """
    Load a pre-trained machine learning model from the specified directory.

    Args:
    path_model_file (str): Path of the file where the model's `.joblib` stored

    Returns:
    - model_loaded: Loaded machine learning model.
    """
    
    if os.path.exists(path_model_file):
        model_loaded = joblib.load(path_model_file)
    else:
        raise ValueError("Model file not found!")

    return model_loaded


    
def main():
    parser = argparse.ArgumentParser(description='Heatmaps')
    arg = parser.add_argument
    arg('--output_path', type=str, default="/work/bb1036/b381362/output/final_results/")
    arg('--fold-num', type=int, default=1, help='n k-fold to run')  
    arg('--type_model', type=str, default='RF', help='select the model XGBoost, RF')
    arg('--variable_target', type=str, default="lwp", help='"Re, COT, LWP, Ndmax" List of the variables to use.')
    arg('--type_plot', type=str, default="captum", help='"captum, my_feature_importances" List of the variables to use.')
    arg('--path_models', type=str, default="/work/bb1036/b381362/output/", help='path of the folder to save the outputs')
    arg('--path_dataframes_scaler', type=str, default="/work/bb1036/b381362/", help='path where is the dataset save as dataframes and scaler')
    arg('--all_data', type=str, default="all_features", help='number of flods to split the dataset')  
 
    args = parser.parse_args()
    output_path = args.output_path
    fold_num = args.fold_num
    variable_target = args.variable_target
    path_models = args.path_models
    type_plot = args.type_plot
    path_dataframes_scaler = args.path_dataframes_scaler
    type_model=args.type_model
    all_data = args.all_data

    name_saving_files = f"{type_model}_{variable_target}_k_fold_{fold_num}"
    path_model_file = f"{path_models}/{all_data}_{name_saving_files}.joblib"
   
    print(" ========================================== ")
    print("Arguments received:")
    for arg in vars(args):
         print(f"{arg}: {getattr(args, arg)}")

    print (f" ==================== {all_data} used  ==================== ")
    channel_relate_clouds = select_channels(all_data=all_data,
                                            variable_target=variable_target,
                                            type_model=type_model)
    
    with open(f"{path_dataframes_scaler}/val_data_k_fold_{fold_num}.pkl", 'rb') as f:
        times_data = pickle.load(f)
    
        
    df_icon_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_train', fold_num)
    df_icon_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    df_icon_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    # df_icon_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_test', fold_num)
    df_ref_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_train', fold_num)
    df_ref_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)
    # df_ref_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_test', fold_num)
    df_ref_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)
    
    loaded_index_all_test = np.load(f'{path_dataframes_scaler}/index_only_clouds_all_val_{fold_num}.npy', allow_pickle=True)
    
    if isinstance(loaded_index_all_test[0], tuple):  # Check if the first element is a tuple (MultiIndex)
        loaded_index_all_test = pd.MultiIndex.from_tuples(loaded_index_all_test)
    
    df_icon_test.index = loaded_index_all_test
    df_ref_test.index = loaded_index_all_test

    
    X_features_val = df_ref_val.loc[:,channel_relate_clouds]
    y_labels_val = df_icon_val[[variable_target]]
    
    X_features_train = df_ref_train.loc[:,channel_relate_clouds]
    y_labels_train = df_icon_train[[variable_target]]

    model = load_model(path_model_file)

    subset_size = 1024
    # subset_size = 8000   #xgboost

    # -------------------subset train
    X_features_val_subset = X_features_val.sample(n=subset_size, random_state=42)
    y_labels_val_subset = y_labels_val.loc[X_features_val.index]

    # -------------------subset train
    X_features_train_subset = X_features_train.sample(n=subset_size, random_state=42)
    y_labels_train_subset = y_labels_train.loc[X_features_train_subset.index]

    
    if variable_target=='lwp' or variable_target=='LWP':
        axis_name = 'LWP'
    elif variable_target=='Nd_max' or variable_target==r'$N_{d,\ max}$':
        axis_name = '$N_{d,\ max}$'

    if type_plot == "shap_values":
        # X_test = X_features_train 
        # y_test = y_labels_train         
        
        X_test = X_features_train_subset 
        y_test = y_labels_train_subset 

        #  --- check performance sample -------------------
        pred = model.predict(X_test)
        gt = np.array(y_test)
        pred = np.array(pred)
        metrics_test = metric_calculation_all_data(outputs=pred,
                                                                             targets=gt)
        print("\n----------------metrics with the scaled target -------")
        print(metrics_test)

        print(f"---------------Shape X_test {np.shape(X_test)} ---------------------")
        
        # background = X_features_train.sample(1000, random_state=30)
        # explainer = shap.TreeExplainer(model, X_test)  
        # explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.to_numpy())
        print(f"---------------Shape calculated ---------------------")
        print(type(shap_values))  # Is it a list?
        print(np.array(shap_values).shape)  # Check dimensions
        print(f"X_test type: {type(X_test)}")
        print(f"X_test shape: {np.shape(X_test)}")
        # =======================================================
        print(f"---------------clustering---------------------")
        # Compute hierarchical clustering on SHAP values
        clustering = shap.utils.hclust(X_test.to_numpy(), y_test.to_numpy())

        # Visualize the clustering
        plt.figure(figsize=(12, 6))
        dendrogram = sch.dendrogram(clustering, labels=X_test.columns, leaf_rotation=90)
        plt.title("Hierarchical Clustering of SHAP Values")
        cluster_plot = f"{output_path}/{all_data}_clustering_shap_{name_saving_files}_train.png"
        plt.savefig(cluster_plot, bbox_inches="tight", dpi=150)
        plt.close()
        # =======================================================

        print(f"--------------- Mean_SHAP_Value csv---------------------")
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        feature_names = X_features_train.columns
  
        # Create DataFrame for feature importance
        shap_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Fold": fold_num,
            "Mean_SHAP_Value": mean_shap_values
        })
        
        file_path = f"{output_path}/{all_data}_shap_{name_saving_files}_s{subset_size}_train.csv"
        # shap_importance_df = shap_importance_df.sort_values(by="Mean_SHAP_Value", ascending=False)
        shap_importance_df.to_csv(file_path, index=False)
   
        if os.path.exists(file_path):
            print(f"File saved successfully at: {file_path}")
        else:
            print(f"Failed to save the file at: {file_path}")

        print(f"--------------- Beeswarm---------------------")
        # -------------------------------------------------------------------------
        # SHAP Summary Plot (Beeswarm) and Bar Plot
        # -------------------------------------------------------------------------
        X_test_np = X_test.to_numpy(dtype=float)
        # shap_values_np = shap_values.detach().cpu().numpy()  
    
        # Create beeswarm summary plot
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_test_np, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary for {axis_name} - Fold {fold_num}")
        summary_beeswarm_path = f"{output_path}/shap_summary_beeswarm_{name_saving_files}_train.png"
        plt.savefig(summary_beeswarm_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        print(f"--------------- bar ---------------------")
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_test_np, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Summary for {axis_name} - Fold {fold_num}")
        summary_bar_path = f"{output_path}/{all_data}_shap_summary_bar_{name_saving_files}_train.png"
        plt.savefig(summary_bar_path, bbox_inches="tight", dpi=150)
        plt.close()


        print(f"--------------- Dependence ---------------------")
        # -------------------------------------------------------------------------
        # SHAP Dependence Plot Example
        # ----------------------------------------         
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
            if i % cols == 0:  # First column
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

