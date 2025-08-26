import torch
import torch.nn as nn
import pickle
import xarray as xr
import numpy as np
import pandas as pd

import torch.nn.functional as F

from utils.functions_general import ref_to_dataframe, select_channels_old

class QuantileRegressor(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

        self.fc1 = nn.Linear(in_features, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        # 3---------------------------------------
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        #-------------------------------------
        # self.fc5 = nn.Linear(hidden_size, 256)
        # self.bn5 = nn.BatchNorm1d(256)
        # self.fc6 = nn.Linear(256, 128)
        # self.bn6 = nn.BatchNorm1d(128)
        # self.fc7 = nn.Linear(128, 64)
        # self.bn7 = nn.BatchNorm1d(64)
        # self.out = nn.Linear(64, len(quantiles))
        
        #-------------------------------------
        self.out = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        #-------------------------------------
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        #-------------------------------------
        # x = F.relu(self.bn5(self.fc5(x)))
        # x = F.relu(self.bn6(self.fc6(x)))
        # x = F.relu(self.bn7(self.fc7(x)))
       
        return self.out(x)
        



def build_quantile_model(
    input_size: int,
    hidden_size: int = 512,
    quantiles: list = None
) -> QuantileRegressor:
    """
    Constructs and returns a QuantileRegressor.

    Args:
        input_size:  number of features per sample
        hidden_size: width of the two hidden layers (default=512)
        quantiles:   list of quantiles to predict (default 9‐segment list)

    Returns:
        An instance of QuantileRegressor ready to .to(device)
    """
    if quantiles is None:
        # quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.95, 0.99]
        quantiles = [
                0.01, 0.05, 0.10, 0.20, 0.30,
                0.50, 0.70, 0.80, 0.85, 0.90,
                0.93, 0.95, 0.97, 0.99
            ]
#         quantiles = [
#     0.01, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30,
#     0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.97, 0.99
# ]

    return QuantileRegressor(input_size, hidden_size, quantiles)



class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)


def get_prediction_nd_lwp(ds_channels, all_data, num_channels, path_model_file_lwp, path_model_file_nd, path_dataframes_scaler_lwp, path_dataframes_scaler_nd, variables_names, fold_num, type_model="NN"):
    """
    ds_channels (xrDataArray): chan, lat, lon
    """
    
    # name_model_lwp = path_model_file_lwp.split("/")[-1].split("_")[2]
    # index_channel_relate_clouds = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]

    channel_relate_clouds_lwp, channel_relate_clouds_nd = select_channels_old(all_data=all_data,
                                                                          num_channels=num_channels,
                                                                          type_model=type_model)

    df_spectral_all = ref_to_dataframe(ds_channels=ds_channels)
    
    nan_counts_per_var = df_spectral_all.isnull().sum()
    print(f"***----------------- nan masked level 1 with all nans: {nan_counts_per_var}-------------------------***")
    
    df_spectral = df_spectral_all.dropna() # to delete the nan where there is not cloud and other nans

        
    # print(df_spectral.describe())
    cleaned_index = df_spectral.index


    df_spectral_lwp = df_spectral.loc[:, channel_relate_clouds_lwp]
    df_spectral_nd = df_spectral.loc[:, channel_relate_clouds_nd]

    df_spectral_lwp.columns=channel_relate_clouds_lwp
    df_spectral_nd.columns=channel_relate_clouds_nd
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -------------------- Using device: {device} --------------- ")
    # ---------------------------- Load model, scalers ---------------------------------
    model_lwp = MyModel(input_size=len(channel_relate_clouds_lwp), 
                        output_size=1)
    
    
    model_nd = MyModel(input_size=len(channel_relate_clouds_nd), 
                        output_size=1)

    
    # model_lwp.load_state_dict(torch.load(path_model_file_lwp, map_location=device))
    # model_lwp.eval()  # Set the model to evaluation mode
    # # model.to(device)
    # model_nd.load_state_dict(torch.load(path_model_file_nd, map_location=device))
    # model_nd.eval() 
    print(path_model_file_lwp)
    state_dict_lwp = torch.load(path_model_file_lwp, map_location=device, weights_only=True)
    model_lwp.load_state_dict(state_dict_lwp)
    model_lwp.eval()  
    
    state_dict_nd = torch.load(path_model_file_nd, map_location=device, weights_only=True)
    model_nd.load_state_dict(state_dict_nd)
    model_nd.eval()  

    try:
        scaler_ref_lwp = pickle.load(open(f'{path_dataframes_scaler_lwp}/scaler_ref_k_fold_{fold_num}.npy', 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_y was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler_lwp}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_ref_lwp = None

    try:
        scaler_ref_nd = pickle.load(open(f'{path_dataframes_scaler_nd}/scaler_ref_k_fold_{fold_num}.npy', 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_y was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler_nd}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_ref_nd = None
    # feature_names = scaler_ref.get_feature_names_out()
    
    mean_subset_lwp = scaler_ref_lwp.mean_[df_spectral.columns.get_indexer(channel_relate_clouds_lwp)]
    scale_subset_lwp = scaler_ref_lwp.scale_[df_spectral.columns.get_indexer(channel_relate_clouds_lwp)]

    mean_subset_nd = scaler_ref_nd.mean_[df_spectral.columns.get_indexer(channel_relate_clouds_nd)]
    scale_subset_nd = scaler_ref_nd.scale_[df_spectral.columns.get_indexer(channel_relate_clouds_nd)]


    try:
        scaler_y_lwp = pickle.load(open(f'{path_dataframes_scaler_lwp}/scaler_{variables_names}_k_fold_{fold_num}.npy', 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_y was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler_lwp}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_y_lwp = None

    try:
        scaler_y_nd = pickle.load(open(f'{path_dataframes_scaler_nd}/scaler_{variables_names}_k_fold_{fold_num}.npy', 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_y was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler_nd}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_y_nd = None

    # Get feature names from the scaler
    target_names = scaler_y_lwp.get_feature_names_out()
    print("Target names:", target_names)
    
    target_means_lwp = scaler_y_lwp.mean_[1]  
    target_vars_lwp = scaler_y_lwp.var_[1]  
   
    target_means_nd = scaler_y_nd.mean_[0]  
    target_vars_nd = scaler_y_nd.var_[0]  
    

    # ---------------------------- end Load model, scalers ---------------------------------

    print("print shape scale nd", np.shape(scale_subset_nd), "print shape nd", np.shape(df_spectral_nd))
    print("print shape scale lwp", np.shape(scale_subset_lwp), "print shape lwp", np.shape(df_spectral_lwp))

    # ----------------------------- scale features, convert tensor ----------------------------------
    # df_ref_rad_scaled_lwp = pd.DataFrame(scaler_ref_lwp.transform(df_spectral_lwp), columns=df_spectral_lwp.columns)
    df_ref_rad_scaled_lwp = (df_spectral_lwp - mean_subset_lwp) / scale_subset_lwp
    x_test_np_lwp = df_ref_rad_scaled_lwp.to_numpy(dtype=float)
    # Convert input data to a torch tensor
    X_tensor_lwp = torch.tensor(x_test_np_lwp, dtype=torch.float32)

    # df_ref_rad_scaled_nd = pd.DataFrame(scaler_ref_nd.transform(df_spectral_nd), columns=df_spectral_nd.columns)
    df_ref_rad_scaled_nd = (df_spectral_nd - mean_subset_nd) / scale_subset_nd
    x_test_np_nd = df_ref_rad_scaled_nd.to_numpy(dtype=float)
    # Convert input data to a torch tensor
    X_tensor_nd = torch.tensor(x_test_np_nd, dtype=torch.float32)

    # print(df_ref_rad_scaled_nd)
    # print("print shape scale nd", np.shape(scale_subset_nd), "print shape nd", np.shape(df_ref_rad_scaled_nd))
    # print("print shape scale lwp", np.shape(scale_subset_lwp), "print shape lwp", np.shape(df_ref_rad_scaled_lwp), df_ref_rad_scaled_lwp)
    # print("print shape scale nd", np.shape(scale_subset_nd), "print shape tensor nd", np.shape(X_tensor_nd))
    # print("print shape scale lwp", np.shape(scale_subset_lwp), "print shape tensor lwp", np.shape(X_tensor_lwp))
    
    # --------------------------- predict  ------------------------------
    with torch.no_grad():
        predictions_lwp = model_lwp(X_tensor_lwp)

    with torch.no_grad():
        predictions_nd = model_nd(X_tensor_nd)
        
    # Move predictions back to CPU and convert to NumPy
    # predictions = predictions.cpu().numpy()
    
    # --------------------------- invert scaler ------------------------------
    # Create an empty DataFrame for storing predictions with the same size as the original
    df_unstandardized_lwp = pd.DataFrame(index=df_spectral_all.index)
    # Insert the predictions back to the corresponding rows (matching cleaned index)
    df_unstandardized_lwp.loc[cleaned_index, "lwp"] = (predictions_lwp.numpy()*np.sqrt(target_vars_lwp)) + target_means_lwp
    # df_unstandardized_lwp.loc[cleaned_index, "lwp"] = predictions_lwp.numpy()  #when is quantile
    
    df_unstandardized_nd = pd.DataFrame(index=df_spectral_all.index)
    df_unstandardized_nd.loc[cleaned_index, "Nd_max"] = (predictions_nd.numpy()*np.sqrt(target_vars_nd)) + target_means_nd
       

    df_unstandardized_modis_lwp = df_unstandardized_lwp.rename_axis(['lat', 'lon'])
    df_unstandardized_modis_lwp = df_unstandardized_modis_lwp.clip(lower=2)
    prediction_modis_xarray_lwp = df_unstandardized_modis_lwp.to_xarray()
    prediction_modis_xarray_lwp

    # df_unstandardized_lwp, df_unstandardized_nd
    df_unstandardized_modis_nd = df_unstandardized_nd.rename_axis(['lat', 'lon'])  # give lat, lon to then get that dimensions
    df_unstandardized_modis_nd = df_unstandardized_modis_nd.clip(lower=2)
    prediction_modis_xarray_nd = df_unstandardized_modis_nd.to_xarray()
    
    

    return prediction_modis_xarray_nd, prediction_modis_xarray_lwp



def predict_with_modis_level1(cleaned_index, df_spectral, df_spectral_lwp,  df_spectral_nd, channel_relate_clouds_lwp, channel_relate_clouds_nd, path_model_file_lwp, path_model_file_nd, path_dataframes_scaler_lwp, path_dataframes_scaler_nd, variables_names, fold_num):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" ***-------------------- Using device: {device} --------------- ")
    # ---------------------------- Load model, scalers ---------------------------------
    model_lwp = MyModel(input_size=len(channel_relate_clouds_lwp), 
                        output_size=1)
    
    model_nd = MyModel(input_size=len(channel_relate_clouds_nd), 
                        output_size=1)

    
#     model_lwp = build_quantile_model(
#     input_size = len(channel_relate_clouds_lwp),
#     hidden_size = 512     # or whatever you like
# ).to(device)
#     quantiles = [0.01, 0.05, 0.10, 0.20, 0.30,
#                 0.50, 0.70, 0.80, 0.85, 0.90,
#                 0.93, 0.95, 0.97, 0.99]


    state_dict_lwp = torch.load(path_model_file_lwp, map_location=device, weights_only=True)
    model_lwp.load_state_dict(state_dict_lwp)
    model_lwp.eval()  

    state_dict_nd = torch.load(path_model_file_nd, map_location=device, weights_only=True)
    model_nd.load_state_dict(state_dict_nd)
    model_nd.eval()  

    # if quantiles is None and hasattr(model_lwp, "quantiles"):
    #     quantiles = model_lwp.quantiles.tolist() #.cpu()
    # elif quantiles is None:
    #     raise ValueError("Please pass in your list of quantiles.")
    # # find the 50% index
    # try:
    #     median_idx = quantiles.index(0.50)
    # except ValueError:
    #     raise ValueError("Your quantiles list must contain 0.5 to pick the median.")

    
    try:
        scaler_ref_lwp = pickle.load(open(f'{path_dataframes_scaler_lwp}/scaler_ref_k_fold_{fold_num}.npy', 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_ref_lwp was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler_lwp}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_ref_lwp = None

    try:
        scaler_ref_nd = pickle.load(open(f'{path_dataframes_scaler_nd}/scaler_ref_k_fold_{fold_num}.npy', 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_ref_nd was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler_nd}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_ref_nd = None
    
    mean_subset_lwp = scaler_ref_lwp.mean_[df_spectral.columns.get_indexer(channel_relate_clouds_lwp)]
    scale_subset_lwp = scaler_ref_lwp.scale_[df_spectral.columns.get_indexer(channel_relate_clouds_lwp)]

    mean_subset_nd = scaler_ref_nd.mean_[df_spectral.columns.get_indexer(channel_relate_clouds_nd)]
    scale_subset_nd = scaler_ref_nd.scale_[df_spectral.columns.get_indexer(channel_relate_clouds_nd)]


    try:
        scaler_y_lwp = pickle.load(open(f'{path_dataframes_scaler_lwp}/scaler_{variables_names}_k_fold_{fold_num}.npy', 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_y_lwp was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler_lwp}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_y_lwp = None

    try:
        scaler_y_nd = pickle.load(open(f'{path_dataframes_scaler_nd}/scaler_{variables_names}_k_fold_{fold_num}.npy', 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_y_nd was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler_nd}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_y_nd = None

    # # Get feature names from the scaler
    target_names = scaler_y_lwp.get_feature_names_out()
    print("Target names:", target_names)
    
    target_means_lwp = scaler_y_lwp.mean_[1]  
    target_vars_lwp = scaler_y_lwp.var_[1]  
   
    target_means_nd = scaler_y_nd.mean_[0]  
    target_vars_nd = scaler_y_nd.var_[0]  
    

    # ---------------------------- end Load model, scalers ---------------------------------
    print("print shape scale nd", np.shape(scale_subset_nd), "print shape nd", np.shape(df_spectral_nd))
    print("print shape scale lwp", np.shape(scale_subset_lwp), "print shape lwp", np.shape(df_spectral_lwp))
    # ----------------------------- scale features, convert tensor ----------------------------------
    df_ref_rad_scaled_lwp = (df_spectral_lwp - mean_subset_lwp) / scale_subset_lwp
    x_test_np_lwp = df_ref_rad_scaled_lwp.to_numpy(dtype=float)
    # Convert input data to a torch tensor
    X_tensor_lwp = torch.tensor(x_test_np_lwp, dtype=torch.float32)

    df_ref_rad_scaled_nd = (df_spectral_nd - mean_subset_nd) / scale_subset_nd
    x_test_np_nd = df_ref_rad_scaled_nd.to_numpy(dtype=float)
    # Convert input data to a torch tensor
    X_tensor_nd = torch.tensor(x_test_np_nd, dtype=torch.float32)

    
    # --------------------------- predict  ------------------------------
    # Perform the forward pass (no gradients needed)
    with torch.no_grad():
        predictions_lwp = model_lwp(X_tensor_lwp)
    # with torch.no_grad():
    #     predictions_lwp_all = model_lwp(X_tensor_lwp)
    # predictions_lwp   = predictions_lwp_all[:,   median_idx]
    # Perform the forward pass (no gradients needed)
    with torch.no_grad():
        predictions_nd = model_nd(X_tensor_nd)
        
    # --------------------------- invert scaler ------------------------------
    # Create an empty DataFrame for storing predictions with the same size as the original
    df_unstandardized_lwp = pd.DataFrame(index=df_spectral.index)
    # Insert the predictions back to the corresponding rows (matching cleaned index)
    df_unstandardized_lwp.loc[cleaned_index, "lwp"] = (predictions_lwp.numpy()*np.sqrt(target_vars_lwp)) + target_means_lwp
    # df_unstandardized_lwp.loc[cleaned_index, "lwp"] = predictions_lwp.numpy()  #when is quantile

    df_unstandardized_nd = pd.DataFrame(index=df_spectral.index)
    df_unstandardized_nd.loc[cleaned_index, "Nd_max"] = (predictions_nd.numpy()*np.sqrt(target_vars_nd)) + target_means_nd
       

    # df_unstandardized_modis_lwp = df_unstandardized_lwp.rename_axis(['lat', 'lon'])
    # df_unstandardized_modis_lwp = df_unstandardized_modis_lwp.clip(lower=2)
    df_unstandardized_modis_lwp =  df_unstandardized_lwp
    # .clip(lower=2)
    # df_unstandardized_lwp
   
    # prediction_modis_xarray_lwp = df_unstandardized_modis_lwp.to_xarray()
    # prediction_modis_xarray_lwp

    # df_unstandardized_modis_nd = df_unstandardized_nd.rename_axis(['lat', 'lon'])  # give lat, lon to then get that dimensions
    # df_unstandardized_modis_nd = df_unstandardized_modis_nd.clip(lower=2)
    df_unstandardized_modis_nd = df_unstandardized_nd
    # .clip(lower=2)
    # prediction_modis_xarray_nd = df_unstandardized_modis_nd.to_xarray()
    


    # return prediction_modis_xarray_nd, prediction_modis_xarray_lwp
    return df_unstandardized_modis_lwp, df_unstandardized_modis_nd

