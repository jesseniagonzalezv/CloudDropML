import pickle
import argparse
import torch
import numpy as np
import pandas as pd
import xarray as xr
import torch.nn as nn
import joblib
import os
import sys


from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.functions import load_dataframe_from_netcdf, select_channels
from utils.functions import metric_calculation_all_data
from utils.functions import convert_fractional_day_to_time
from inverse_model_NN import MyModel, evaluate_model
from dataframe_all_data import get_xarray_rad_ref, get_xarray_rad_ref_18


def obtain_test_data(path_icon, path_rttov, scaler_icon, scaler_ref_rad, channel_relate_clouds, lat_min=None,
                     lat_max=None):
    variables = ["Nd_max", "lwp"]
    # ------------------------------------- ICON LES---------------------------------------------- Useful to mask only liquid clouds
    icon_ds = xr.open_dataset(path_icon)

    # Check if the 'time' variable is a datetime64[ns] dtype
    if icon_ds['time'].dtype == np.dtype('datetime64[ns]'):

        time_data = icon_ds['time'].values
        print("The 'time' variable is correctly type datetime64[ns]:", time_data)
    else:
        print("The 'time' variable is not of type datetime64[ns] is proleptic_gregorian")

        time_data = convert_fractional_day_to_time(icon_ds.time.values)
        time_data = pd.to_datetime(time_data)

    if lat_min and lat_max:
        icon_ds = icon_ds[variables].sel(lat=slice(lat_min, lat_max))
        ds_channels_rttov = xr.open_dataset(path_rttov).sel(lat=slice(lat_min, lat_max))
    else:
        icon_ds = icon_ds[variables]
        ds_channels_rttov = xr.open_dataset(path_rttov)

    # print(icon_ds)
    icon_ds['lat'] = icon_ds['lat'].astype(np.float32)
    icon_ds['lon'] = icon_ds['lon'].astype(np.float32)

    # icon_ds = icon_ds.expand_dims({"time": [time_data]})
    if "time" not in icon_ds.dims:
        icon_ds = icon_ds.expand_dims({"time": [time_data]})

    mask_clouds = (icon_ds.Nd_max == 0) | (icon_ds.lwp == 0) | (icon_ds.Nd_max <= 2) | (icon_ds.Nd_max >= 2000) | (
                icon_ds.lwp <= 2) | (icon_ds.lwp >= 2000)

    # ------------------------------------- RTTOV simulations----------------------------------------------

    ds_channels_rttov = ds_channels_rttov.expand_dims({"time": [time_data]})
    # ds_rttov = get_xarray_rad_ref(ds_channels_rttov, data_type="ref_rad_total")
    ds_rttov = get_xarray_rad_ref_18(ds_channels_rttov, data_type="ref_rad_total")

    mask_expanded = xr.DataArray(mask_clouds, dims=['time', 'lat', 'lon'])
    mask_broadcasted = xr.broadcast(mask_expanded, ds_rttov)[0]  # Automatically aligns and broadcasts
    print("Before transpose", mask_broadcasted.dims)  # Before transpose
    mask_broadcasted = mask_broadcasted.transpose('time', 'chan', 'lat', 'lon')
    print("After transpose", mask_broadcasted.dims)  # Before transpose
    print(f"mask_broadcasted, {np.shape(mask_broadcasted)}")

    icon_ds = icon_ds.where(~mask_expanded)
    ds_channels = ds_rttov.where(~mask_broadcasted)

    df_spectral_all = ds_channels.to_dataframe().unstack(level=['chan'])
    df_spectral_all.columns = df_spectral_all.columns.droplevel(0)
    nan_count = df_spectral_all.isna().sum()
    # print(nan_count)
    df_spectral = df_spectral_all.dropna()  # to delete the nan where there is not cloud
    df_ref_rad_scaled = pd.DataFrame(scaler_ref_rad.transform(df_spectral),
                                      columns=df_spectral.columns.values,
                                      index=df_spectral.index)

    df_icon_all = icon_ds.to_dataframe()
    df_icon_all = df_icon_all.dropna()  # to delete the nan where there is not cloud
    df_icon_all_scaled = pd.DataFrame(scaler_icon.transform(df_icon_all),
                                      columns=df_icon_all.columns.values,
                                      index=df_icon_all.index)

    test_times = icon_ds.time.values

    print("================================== obtained test data =========================")
    print(np.shape(df_ref_rad_scaled), np.shape(df_icon_all_scaled))
    
    return df_ref_rad_scaled, df_icon_all_scaled, test_times





def load_data_validation(path_dataframes_scaler, fold_num):
    #     times_data = pickle.load(f)
    with open(f"{path_dataframes_scaler}/val_data_k_fold_{fold_num}.pkl", 'rb') as f:
        times_data = pickle.load(f)
    test_times = times_data['val_times']

    df_icon_all_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    df_ref_all_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)

    loaded_index_all_test = np.load(f'{path_dataframes_scaler}/index_only_clouds_all_val_{fold_num}.npy',
                                    allow_pickle=True)

    if isinstance(loaded_index_all_test[0], tuple):  
        loaded_index_all_test = pd.MultiIndex.from_tuples(loaded_index_all_test)
    # print(loaded_index_all_test)
    df_icon_all_test.index = loaded_index_all_test
    df_ref_all_test.index = loaded_index_all_test

    print("================================== obtained validation fold data =========================")

    return df_ref_all_test, df_icon_all_test, test_times


# -------------------------------------------------- MAIN CODE --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--type-model', type=str, default='NN', help='select the model NN, RF')
    arg('--path_dataframes_scaler', type=str, default="/work/bb1036/b381362/",
        help='path where is the dataset save as dataframes and scaler')
    arg('--path_models', type=str, default="/work/bb1036/b381362/output/",
        help='path of the folder to save the outputs')
    arg('--fold-num', type=int, default=1, help='n k-fold to run')
    arg('--variables_names', type=str, default="LWP-Ndmax", help='"Re-COT" List of the variables to use.')
    # arg('--threshold', type=str, default="filter", help='filter, nofilter')
    arg('--variable_target', type=str, default="lwp", help='"Re, COT, LWP, Ndmax" List of the variables to use.')
    arg('--path_model_file', type=str, default="/work/bb1036/b381362/output/model.pth")
    arg('--all_data', type=str, default="all_features", help='number of flods to split the dataset')
    arg('--path_icon_specific_test', type=str, default=None,
        help="Provide the path of the NC file with the ICON data.")
    arg('--path_channels_specific_test', type=str, default=None,
        help="Provide the path of the NC file with the ref rad data.")

    args = parser.parse_args()

    type_model = args.type_model
    fold_num = args.fold_num
    path_dataframes_scaler = args.path_dataframes_scaler
    path_models = args.path_models
    path_model_file = args.path_model_file
    variables_names = args.variables_names
    # threshold = args.threshold
    variable_target = args.variable_target
    all_data = args.all_data

    batch_size = 1024

    if args.path_icon_specific_test == "None" and args.path_channels_specific_test == "None":
        args.path_icon_specific_test = None
        args.path_channels_specific_test = None

    path_icon_specific_test = args.path_icon_specific_test
    path_channels_specific_test = args.path_channels_specific_test

    
    print(" ========================================== ")
    print("Arguments received:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Obtener la extensión del archivo
    _, file_extension = os.path.splitext(path_model_file)

    # =============================================================================================
    # =========================== Obtaining scalers and channesl ==================================
    # =============================================================================================
    channel_relate_clouds = select_channels(all_data=all_data,
                                            variable_target=variable_target,
                                            type_model=type_model)

    try:
        scaler_icon = pickle.load(open(f'{path_dataframes_scaler}/scaler_{variables_names}_k_fold_{fold_num}.npy',
                                       'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_icon was not found. Continue with the ejecution using None as scaler.")
        print(f'{path_dataframes_scaler}/scaler_{variables_names}_k_fold_{fold_num}.npy')
        scaler_icon = None

    try:
        scaler_ref_rad = pickle.load(open(f'{path_dataframes_scaler}/scaler_ref_k_fold_{fold_num}.npy',
                                          'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_ref_rad was not found.")
        print(f'{path_dataframes_scaler}/scaler_ref_k_fold_{fold_num}.npy')
        scaler_ref_rad = None

    # =============================================================================================
    # ================================= Obtaining the data ========================================
    # =============================================================================================

    if path_icon_specific_test is not None and path_channels_specific_test is not None:
        print("Using provided paths:", path_icon_specific_test, path_channels_specific_test)
        df_ref_all_test, df_icon_all_test, test_times = obtain_test_data(path_icon=path_icon_specific_test,
                                                                         path_rttov=path_channels_specific_test,
                                                                         scaler_icon=scaler_icon,
                                                                         scaler_ref_rad=scaler_ref_rad,
                                                                         channel_relate_clouds=channel_relate_clouds)
        # ,
        # lat_min=50, lat_max=54.50)

        time_data = pd.to_datetime(test_times).item()
        formatted_time_data = time_data.strftime('%Y%m%dT%H%M')

    else:
        print(f"------------------Using default data of validation fold ------- {fold_num}")
        formatted_time_data = "validation"

        df_ref_all_test, df_icon_all_test, test_times = load_data_validation(
            path_dataframes_scaler=path_dataframes_scaler,
            fold_num=fold_num)

        # ============================================================================================
    # ----------------------- Select channels to predict
    # ============================================================================================
    y_labels = df_icon_all_test[[variable_target]]
    # ------------------------- log ----------------
    # y_labels = np.logdf_icon_all_test[[variable_target]])
    # X_features = df_ref_all_test
    X_features = df_ref_all_test.loc[:, channel_relate_clouds]

    target_index = df_icon_all_test.columns.get_loc(variable_target)
    if isinstance(scaler_icon, StandardScaler):
        target_means = scaler_icon.mean_[target_index]
        target_vars = scaler_icon.var_[target_index]

    print(f"------------------------- Features Size: {np.shape(X_features)} --------------------------------------")
    # =============================================================================================
    # ===================================== RF or XGBoost =========================================
    # =============================================================================================
    if file_extension == '.joblib':
        print("Es un modelo de scikit-learn")
        # ------------------------- log ----------------
        # y_labels = np.log(y_labels[[variable_target]]))
        model = joblib.load(path_model_file)
        pred = model.predict(X_features)

        # --------------- invert standard scaler --------------
        df_unstandardized_all_preds_test = pd.DataFrame(pred * np.sqrt(target_vars) + target_means,
                                                        columns=y_labels.columns,
                                                        index=y_labels.index)
        df_unstandardized_all_targets_test = pd.DataFrame(y_labels * np.sqrt(target_vars) + target_means,
                                                          columns=y_labels.columns,
                                                          index=y_labels.index)

    # =============================================================================================
    # =========================================== NN ==== =========================================
    # =============================================================================================
    elif file_extension == '.pth':
        print("Es un modelo de PyTorch")

        # Define el modelo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" -------------------- Using device: {device} --------------- ")

        criterion = nn.SmoothL1Loss(beta=0.5)

        # Reinitialize the model with the inferred sizes
        model = MyModel(input_size=X_features.shape[1],
                        output_size=y_labels.shape[1]).to(device)

        model.load_state_dict(torch.load(path_model_file, map_location=device))
        model.eval()  

        x_test_np = X_features.to_numpy(dtype=float)
        y_test_np = y_labels.to_numpy(dtype=float)

        X_tensor = torch.tensor(x_test_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_test_np, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss, rmse_avg, r2_avg, all_preds_test, all_targets_test = evaluate_model(model, dataloader, criterion, device)

        df_unstandardized_all_preds_test = pd.DataFrame(
            all_preds_test.cpu().detach().numpy() * np.sqrt(target_vars) + target_means,
            columns=y_labels.columns,
            index=y_labels.index)
        df_unstandardized_all_targets_test = pd.DataFrame(
            all_targets_test.cpu().detach().numpy() * np.sqrt(target_vars) + target_means,
            columns=y_labels.columns,
            index=y_labels.index)

    # =============================================================================================
    # ==================================== metrics and  nd file ===================================
    # =============================================================================================

    print(f"---------- all data test -----------")
    metric_calculation_all_data(outputs=df_unstandardized_all_preds_test,
                                targets=df_unstandardized_all_targets_test)

    for i, n_time in enumerate(test_times):
        print(f"---------- {n_time}-----------")
        df_target_ntime = df_unstandardized_all_targets_test.loc[
            df_unstandardized_all_targets_test.index.get_level_values(0) == n_time]
        df_pred_ntime = df_unstandardized_all_preds_test.loc[
            df_unstandardized_all_preds_test.index.get_level_values(0) == n_time]

        metric_calculation_all_data(outputs=df_pred_ntime,
                                    targets=df_target_ntime)

    # time = pd.to_datetime(test_times)
    df_unstandardized_all_preds_test = df_unstandardized_all_preds_test.rename_axis(['time', 'lat', 'lon'])
    df_unstandardized_all_targets_test = df_unstandardized_all_targets_test.rename_axis(['time', 'lat', 'lon'])

    # # Convert the DataFrame (target and predictions) to xarray DataArrays
    df_unstandardized_all_preds_test = df_unstandardized_all_preds_test.clip(lower=2)
  
    print("after clipping low values -----------")
    metric_calculation_all_data(outputs=df_unstandardized_all_preds_test,
                                targets=df_unstandardized_all_targets_test)

    target_xarray = df_unstandardized_all_targets_test.to_xarray()
    prediction_xarray = df_unstandardized_all_preds_test.to_xarray()

    xr_output = xr.Dataset({
        f'target_{variable_target}': target_xarray[variable_target],  # Target DataArray
        f'prediction_{variable_target}': prediction_xarray[variable_target]  # Prediction DataArray
    })


    if variable_target == 'lwp' or variable_target == 'LWP':
        units = r'$gm^{-2}$'
    elif variable_target == 'Nd_max' or variable_target == r'$N_{d,\ max}$':
        units = r'$cm^{-3}$'
    elif variable_target == 'Reff_cloud_top' or variable_target == r'$r_e$':
        units = r'$μm$'
    elif variable_target == 'cot' or variable_target == 'COT':
        units = 'COT'

    xr_output.attrs['units'] = units

    try:
        predictions_folder = os.path.join(path_models, "predictions")
        os.makedirs(predictions_folder, exist_ok=True)
        format_name = f"{all_data}_prediction_target_{formatted_time_data}_{type_model}_{variable_target}_k_fold_{fold_num}"
        file_path = f"{predictions_folder}/{format_name}.nc"
        xr_output.to_netcdf(file_path, 'w')
        print(f"Predictions file saved successfully at {file_path}")
    except Exception as e:
        print(f"Failed to save the file: {str(e)}")
    finally:
        xr_output.close()


if __name__ == '__main__':
    main()

