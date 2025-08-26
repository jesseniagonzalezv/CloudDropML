import xarray as xr
import numpy as np
import pandas as pd
import os

         

def select_channels_new(all_data_lwp, all_data_nd, type_model, num_channels="18_chan"):
    name_channels = np.array(['1: 0.645 μm', '2: 0.856 μm', '3: 0.466 μm', '4: 0.554 μm',
    '5: 1.241 μm', '6: 1.628 μm', '7: 2.113 μm', '8: 0.412 μm',
    '9: 0.442 μm', '10: 0.487 μm', '11: 0.530 μm', '12: 0.547 μm',
    '13: 0.666 μm', '14: 0.678 μm', '15: 0.747 μm', '16: 0.867 μm',
    '17: 0.904 μm', '18: 0.936 μm', '19: 0.935 μm', '20: 3.777 μm',
    '21: 3.981 μm', '22: 3.971 μm', '23: 4.061 μm', '24: 4.448 μm',
    '25: 4.526 μm', '26: 1.382 μm', '27: 6.783 μm', '28: 7.344 μm',
    '29: 8.550 μm', '30: 9.720 μm', '31: 11.017 μm', '32: 12.036 μm',
    '33: 13.362 μm', '34: 13.683 μm', '35: 13.923 μm', '36: 14.213 μm'])

    # ------------------------------------ lwp
    if all_data_lwp == "all_features":
          
        if num_channels=="18_chan":
            index_channel_relate_clouds_lwp = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]
        elif num_channels=="17_chan":
            index_channel_relate_clouds_lwp = [0, 1, 2, 3, 4, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]

    else:
        # ------------------- MIF  -------------------
        if num_channels=="18_chan":
            if type_model == "NN":
                index_channel_relate_clouds_lwp = [0, 4, 5]  #bb1143 #bb1036
            elif type_model == "RF":
                index_channel_relate_clouds_lwp = [2, 3, 5, 6] #bb1143
            elif type_model == "XGBoost":
                index_channel_relate_clouds_lwp = [0, 2, 4, 5] #bb1143
                
        elif num_channels=="17_chan":
            if type_model == "NN":
                index_channel_relate_clouds_lwp = [0, 1, 4]  #bb1143

    # ------------------------------------ nd
    if all_data_nd == "all_features":
          
        if num_channels=="18_chan":
            index_channel_relate_clouds_nd = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]          
        elif num_channels=="17_chan":
            index_channel_relate_clouds_nd = [0, 1, 2, 3, 4, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]
    
    else:
        # ------------------- MIF  -------------------
        if num_channels=="18_chan":
            if type_model == "NN":
                index_channel_relate_clouds_nd = [2, 4, 5, 19, 22, 28, 32] #bb1143
                # index_channel_relate_clouds_nd = [2, 3, 5, 19, 22, 28, 32] #bb1036
            elif type_model == "RF":
                index_channel_relate_clouds_nd = [2, 5, 6, 28, 30, 34, 35] #bb1143
            elif type_model == "XGBoost":
                index_channel_relate_clouds_nd = [1, 2, 5, 6, 19, 22, 28, 30] #bb1143
                
        elif num_channels=="17_chan":
            if type_model == "NN":
                index_channel_relate_clouds_nd = [1, 2, 4, 19, 22, 28, 32] #bb1143

    
    channel_relate_clouds_lwp = name_channels[index_channel_relate_clouds_lwp]
    channel_relate_clouds_nd = name_channels[index_channel_relate_clouds_nd]
    print(f" ==================== Channels used lwp: {channel_relate_clouds_lwp}  ==================== ")    
    print(f" ==================== Channels used nd: {channel_relate_clouds_nd}  ==================== ")    

    return channel_relate_clouds_lwp, channel_relate_clouds_nd

def select_channels_old(all_data, type_model, num_channels="18_chan"):
    name_channels = np.array(['1: 0.645 μm', '2: 0.856 μm', '3: 0.466 μm', '4: 0.554 μm',
    '5: 1.241 μm', '6: 1.628 μm', '7: 2.113 μm', '8: 0.412 μm',
    '9: 0.442 μm', '10: 0.487 μm', '11: 0.530 μm', '12: 0.547 μm',
    '13: 0.666 μm', '14: 0.678 μm', '15: 0.747 μm', '16: 0.867 μm',
    '17: 0.904 μm', '18: 0.936 μm', '19: 0.935 μm', '20: 3.777 μm',
    '21: 3.981 μm', '22: 3.971 μm', '23: 4.061 μm', '24: 4.448 μm',
    '25: 4.526 μm', '26: 1.382 μm', '27: 6.783 μm', '28: 7.344 μm',
    '29: 8.550 μm', '30: 9.720 μm', '31: 11.017 μm', '32: 12.036 μm',
    '33: 13.362 μm', '34: 13.683 μm', '35: 13.923 μm', '36: 14.213 μm'])
    
    if all_data == "all_features":
          
        if num_channels=="18_chan":
            index_channel_relate_clouds_lwp = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]
            index_channel_relate_clouds_nd = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]          
        elif num_channels=="17_chan":
            index_channel_relate_clouds_lwp = [0, 1, 2, 3, 4, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]
            index_channel_relate_clouds_nd = [0, 1, 2, 3, 4, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]       
    else:
        # ------------------- MIF  -------------------
        if num_channels=="18_chan":
            if type_model == "NN":
                index_channel_relate_clouds_lwp = [0, 4, 5]  #bb1143 #bb1036
                index_channel_relate_clouds_nd = [2, 4, 5, 19, 22, 28, 32] #bb1143
                # index_channel_relate_clouds_nd = [2, 3, 5, 19, 22, 28, 32] #bb1036
            elif type_model == "RF":
                index_channel_relate_clouds_lwp = [2, 3, 5, 6] #bb1143
                index_channel_relate_clouds_nd = [2, 5, 6, 28, 30, 34, 35] #bb1143
            elif type_model == "XGBoost":
                index_channel_relate_clouds_lwp = [0, 2, 4, 5] #bb1143
                index_channel_relate_clouds_nd = [1, 2, 5, 6, 19, 22, 28, 30] #bb1143
                
        elif num_channels=="17_chan":
            if type_model == "NN":
                index_channel_relate_clouds_lwp = [0, 1, 4]  #bb1143
                index_channel_relate_clouds_nd = [1, 2, 4, 19, 22, 28, 32] #bb1143


    channel_relate_clouds_lwp = name_channels[index_channel_relate_clouds_lwp]
    channel_relate_clouds_nd = name_channels[index_channel_relate_clouds_nd]
    print(f" ==================== Channels used lwp: {channel_relate_clouds_lwp}  ==================== ")    
    print(f" ==================== Channels used nd: {channel_relate_clouds_nd}  ==================== ")    

    return channel_relate_clouds_lwp, channel_relate_clouds_nd



def ref_to_dataframe(ds_channels):

    df_spectral_all = ds_channels.to_dataframe().unstack(level=['chan'])
    df_spectral_all.columns = df_spectral_all.columns.droplevel(0)

    # if "chan" in df_spectral_all.columns:
    #     df_spectral_all = df_spectral_all.drop(columns="chan")


    return df_spectral_all


def save_df_to_netcdf_fold(df, fold_num, prefix, path_output):
    """
    Convert a pandas DataFrame to an xarray Dataset and save it to a netCDF file.

    Parameters:
    df (pandas.DataFrame): The DataFrame you wish to convert and save.
    fold_num (int): The fold number, used to differentiate between the files when saving.
    prefix (str): A prefix for the filename, used to indicate the type of data being saved.

    Returns:
    None: This function does not return a value. It saves the dataset to a file.

    Example:
    save_df_to_netcdf(df_icon_pca_train, 0, 'df_icon_pca_train')
    This will save the 'df_icon_pca_train' DataFrame for fold number 0 to a netCDF file named 'df_icon_pca_train_fold_0.nc'.
    """  # Convert the pandas DataFrame to xarray Dataset
    xr_dataset = xr.Dataset.from_dataframe(df)

    # save this dataset to a netCDF file
    filename = f'{path_output}/{prefix}_fold_{fold_num}.nc'

    if os.path.exists(filename):
        os.remove(filename)
        print(f"Previous file {filename} deleted.")
    
    print(f"Saving dataframe to {filename}")
    xr_dataset.to_netcdf(filename)


def save_xarray_to_netcdf_fold(xr_dataset, fold_num, prefix, path_output):

    # save this dataset to a netCDF file
    filename = f'{path_output}/{prefix}_fold_{fold_num}.nc'

    if os.path.exists(filename):
        os.remove(filename)
        print(f"Previous file {filename} deleted.")
    
    print(f"Saving dataframe to {filename}")
    xr_dataset.to_netcdf(filename)


    
def confirm_indices_and_join(df, df_labels):
    # Check if the index sizes are the same
    if not df.index.equals(df_labels.index):
        print(
            " ---------------------------------------------------------------------------------------------------------------- ")
        print("Indices values are different. Cannot join the DataFrames.")
        return None
    else:
        # If both size and values are the same, proceed with the join
        print(
            " ---------------------------------------------------------------------------------------------------------------- ")

        print("Indices match in values. Proceeding with join.")
        df_combined = df.join(df_labels, how='inner')  # Use 'inner' to ensure matching indices
    if df_combined is not None:
        print("Join successful")
    else:
        print("Join not performed due to index mismatch.")
    return df_combined


def combine_lwp_nd_dataframe(prediction_xarray_nd, prediction_xarray_lwp):
    predi_lwp= prediction_xarray_lwp.lwp
    predi_nd= prediction_xarray_nd.Nd_max
    
    df_nd_max = predi_nd.to_dataframe()
    df_lwp = predi_lwp.to_dataframe()
    
    # Combine into a new dataframe
    combined_df = pd.DataFrame({
        'Nd_max': df_nd_max['Nd_max'],
        'lwp': df_lwp['lwp']
    })
    print(f"Nd range: {combined_df['Nd_max'].min()} to {combined_df['Nd_max'].max()}")
    print(f"LWP range: {combined_df['lwp'].min()} to {combined_df['lwp'].max()}")

    print(combined_df.describe().round(3))

    return combined_df
    
    
def filter_dataframe_cloud(df_icon, min_value_threshold=5, max_value_threshold=2000):
    """
    The function will convert to dataframe ->then-> filer only clouds ->then-> filter range from 2 -2000
    Input: 
        -- df_icon (dataframe: this need to have min 2 variables columns "Nd_max", "lwp"
    Output
        --filtered_df_icon (dataframe)
    
    """

    print(" ===================== filter only clouds =================================")
    print(f" -------------------- samples before all data: {len(df_icon)}")
    df_icon_only_clouds = df_icon[(df_icon.iloc[:, :2] != 0).all(axis=1)]
    print(f" -------------------- samples after -- only clouds: {len(df_icon_only_clouds)}")

    # max_value_threshold = 2000
    # min_value_threshold = 2
    # min_value_threshold = 5
    if (min_value_threshold is not None) and (max_value_threshold is not None):
        filtered_df_icon = df_icon_only_clouds.loc[
            (df_icon_only_clouds['lwp'] > min_value_threshold) &
            (df_icon_only_clouds['lwp'] < max_value_threshold) &
            (df_icon_only_clouds['Nd_max'] > min_value_threshold) &
            (df_icon_only_clouds['Nd_max'] < max_value_threshold)
            ].copy()
        print(f" -------------------- samples after threshold {min_value_threshold}-{max_value_threshold}: {len(df_icon_only_clouds)}")
    else:
        print("==================== no threshold ========================")
        filtered_df_icon = df_icon_only_clouds

    print(f" --------------------------- After filter\n: {filtered_df_icon.describe().round(3)}")
    
    return filtered_df_icon


def calculate_nd_with_ref_cod(tau_c, r_e, alpha=1.37* 1e-5):
    """
    Calculate cloud droplet number concentration (Nd) over a 2D grid.

    Parameters:
    - tau_c: 2D numpy array of cloud optical depth
    - r_e: 2D numpy array of cloud droplet effective radius (in micrometers)
    - alpha: Proportionality constant (default is 1.37)

    Returns:
    - Nd: 2D numpy array of cloud droplet number concentration (in cm^-3)
    """
    # Convert effective radius from micrometers to meters
    r_e_m = r_e * 1e-6

    # Calculate Nd using the formula
    Nd = (alpha * np.sqrt(tau_c)) * (r_e_m**(-2.5))

    # Convert Nd from m^-3 to cm^-3
    Nd_cm3 = Nd * 1e-6

    return Nd_cm3

