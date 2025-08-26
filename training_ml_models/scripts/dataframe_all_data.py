import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import argparse
import numpy as np
import math
import os
import pickle as pk

import seaborn as sns
import copy      
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def get_xarray_rad_ref(simulation_ds, data_type="ref_rad_total"):
    # ---------------------- All 36 names channels -------------------------------
    name_channels = np.array(['1: 0.645 μm', '2: 0.856 μm', '3: 0.466 μm', '4: 0.554 μm',
   '5: 1.241 μm', '6: 1.628 μm', '7: 2.113 μm', '8: 0.412 μm',
   '9: 0.442 μm', '10: 0.487 μm', '11: 0.530 μm', '12: 0.547 μm',
   '13: 0.666 μm', '14: 0.678 μm', '15: 0.747 μm', '16: 0.867 μm',
   '17: 0.904 μm', '18: 0.936 μm', '19: 0.935 μm', '20: 3.777 μm',
   '21: 3.981 μm', '22: 3.971 μm', '23: 4.061 μm', '24: 4.448 μm',
   '25: 4.526 μm', '26: 1.382 μm', '27: 6.783 μm', '28: 7.344 μm',
   '29: 8.550 μm', '30: 9.720 μm', '31: 11.017 μm', '32: 12.036 μm',
   '33: 13.362 μm', '34: 13.683 μm', '35: 13.923 μm', '36: 14.213 μm'])
    # ---------------------- only 18 channels -------------------------------    
    index_channel_relate_clouds = [0,1,2,3,4,5,6,19,20,21,22,28,30,31,32,33,34,35]

    if data_type == "ref_rad_total" or data_type == "ref_all":
        # new_ds = xr.full_like(simulation_ds["rad_total"], 0.0)

        if len(simulation_ds.chan.values) == 36:
            # print(len(simulation_ds.chan.values) )
            new_ds = xr.DataArray(dims=["time", "chan", "lat", "lon"], 
                                  coords={"time": simulation_ds.time, 
                                          "chan": simulation_ds.chan[index_channel_relate_clouds],
                                          "lat": simulation_ds.lat, "lon": simulation_ds.lon}, 
                                  name=data_type)

            # dimesions: time, chan, lat, lon
            index_channel_relate_clouds_ref = [0,1,2,3,4,5,6]  # reflectances 1-29, 26 (0-18, 25) 
            index_channel_relate_clouds_rad = [19,20,21,22,28,30,31,32,33,34,35]  # radiances 20-23, 29, 31-36
            # dimesions: time, chan, lat, lon
            new_ds[:, 0:7, :, :] = simulation_ds["ref_total"].isel(chan=index_channel_relate_clouds_ref)
            new_ds[:, 7:18, :, :] = simulation_ds["rad_total"].isel(chan=index_channel_relate_clouds_rad)

    
        elif len(simulation_ds.chan.values) == 18:
            new_ds.name = data_type
            new_ds = xr.DataArray(dims=["time", "chan", "lat", "lon"], 
                                  coords={"time": simulation_ds.time, 
                                          "chan": simulation_ds.chan[index_channel_relate_clouds],
                                          "lat": simulation_ds.lat, "lon": simulation_ds.lon}, 
                                  name=data_type)

            index_channel_relate_clouds_ref = [0,1,2,3,4,5,6]  
            index_channel_relate_clouds_rad = [19,20,21,22,28,30,31,32,33,34,35]  

            # dimesions: time, chan, lat, lon
            new_ds[:, 0:7, :, :] = simulation_ds["ref_total"].isel(chan=slice(0,7))
            new_ds[:, 7:18, :, :] = simulation_ds["rad_total"].isel(chan=slice(7,18))


        # Store units as separate attributes
        new_ds.attrs['ref_units'] = simulation_ds['ref_total'].attrs['units']
        new_ds.attrs['rad_units'] = simulation_ds['rad_total'].attrs['units']

        desired_channels_ref = np.array(list(range(0, 19)) + [25])  # Python indexing starts at 0, so we use 0-18, 25 to get channels 1-19, 26
        desired_channels_rad = np.array(list(range(19, 25)) + list(
        range(26, 36)))  # Python indexing starts at 0, so we use 19-24, 26,35 to get channels 20-25, 27-36
        # Adding which bands are ref and which are rad
        new_ds.attrs['ref_bands_idx'] = desired_channels_ref + 1
        new_ds.attrs['rad_bands_idx'] = desired_channels_rad + 1

        new_ds.attrs['ref_long_name'] = 'All-sky upwelling reflectance at TOA'
        new_ds.attrs['rad_long_name'] = 'All-sky upwelling radiance at TOA'

        # -------------------------------------------------------------------
                
        new_ds.attrs['chan_index'] = np.array(index_channel_relate_clouds) + 1
        
        channel_relate_clouds = name_channels[index_channel_relate_clouds]        
        new_ds['chan'] = channel_relate_clouds
        # -------------------------------------------------------

    return new_ds


def get_xarray_rad_ref_17(simulation_ds, data_type="ref_rad_total"):
    """
    Generate a new xarray Dataset merging reflectance and radiance data based on the provided data type.
    The function creates a new Dataset based on the 'ref_rad_total' data type, combining reflectance
    ("ref") and radiance ("rad") values from the input Dataset. The resulting Dataset has reflectance
    and radiance values in designated channel ranges.

    Parameters:
        simulation_ds (xarray.Dataset): Input xarray Dataset containing either reflectance or radiance data.
        data_type (str): Type of data to be processed. 'ref_rad_total' is supported, which
                         indicates a combination of reflectance and radiance.

    Returns:
        xarray.Dataset: A new Dataset with combined reflectance and radiance data, with attributes indicating
                        units, channels and descriptions for both reflectance and radiance.

    Notes:
        This function assumes specific channel ranges for reflectance and radiance values.
        The new Dataset retains units as separate attributes and deletes some unwanted attributes.
        For 'ref_rad_total', reflectance channels are in the range 1-19 (and 26), while radiance channels
        are in the range 21-25 and 27-36.
    """
    # ---------------------- All 36 names channels -------------------------------
    name_channels = np.array(['1: 0.645 μm', '2: 0.856 μm', '3: 0.466 μm', '4: 0.554 μm',
   '5: 1.241 μm', '6: 1.628 μm', '7: 2.113 μm', '8: 0.412 μm',
   '9: 0.442 μm', '10: 0.487 μm', '11: 0.530 μm', '12: 0.547 μm',
   '13: 0.666 μm', '14: 0.678 μm', '15: 0.747 μm', '16: 0.867 μm',
   '17: 0.904 μm', '18: 0.936 μm', '19: 0.935 μm', '20: 3.777 μm',
   '21: 3.981 μm', '22: 3.971 μm', '23: 4.061 μm', '24: 4.448 μm',
   '25: 4.526 μm', '26: 1.382 μm', '27: 6.783 μm', '28: 7.344 μm',
   '29: 8.550 μm', '30: 9.720 μm', '31: 11.017 μm', '32: 12.036 μm',
   '33: 13.362 μm', '34: 13.683 μm', '35: 13.923 μm', '36: 14.213 μm'])
    index_channel_relate_clouds = [0,1,2,3,4,6,19,20,21,22,28,30,31,32,33,34,35]
    

    if data_type == "ref_rad_total" or data_type == "ref_all":
        # new_ds = xr.full_like(simulation_ds["rad_total"], 0.0)

        if len(simulation_ds.chan.values) == 36:
            new_ds = xr.DataArray(dims=["time", "chan", "lat", "lon"], 
                                  coords={"time": simulation_ds.time, 
                                          "chan": simulation_ds.chan[index_channel_relate_clouds],
                                          "lat": simulation_ds.lat, "lon": simulation_ds.lon}, 
                                  name=data_type)

            index_channel_relate_clouds_ref = [0,1,2,3,4,6] 
            index_channel_relate_clouds_rad = [19,20,21,22,28,30,31,32,33,34,35] 
            new_ds[:, 0:6, :, :] = simulation_ds["ref_total"].isel(chan=index_channel_relate_clouds_ref)
            new_ds[:, 6:17, :, :] = simulation_ds["rad_total"].isel(chan=index_channel_relate_clouds_rad)

    
        elif len(simulation_ds.chan.values) == 18:
            new_ds.name = data_type
            new_ds = xr.DataArray(dims=["time", "chan", "lat", "lon"], 
                                  coords={"time": simulation_ds.time, 
                                          "chan": simulation_ds.chan[index_channel_relate_clouds],
                                          "lat": simulation_ds.lat, "lon": simulation_ds.lon}, 
                                  name=data_type)

            index_channel_relate_clouds_ref = [0,1,2,3,4,6]  # reflectances 1-29, 26 (0-18, 25) 
            index_channel_relate_clouds_rad = [19,20,21,22,28,30,31,32,33,34,35]  # radiances 20-23, 29, 31-36
            new_ds[:, 0:6, :, :] = simulation_ds["ref_total"].isel(chan=[0,1,2,3,4,6])
            new_ds[:, 6:17, :, :] = simulation_ds["rad_total"].isel(chan=slice(7,18))
    

        # Store units as separate attributes
        new_ds.attrs['ref_units'] = simulation_ds['ref_total'].attrs['units']
        new_ds.attrs['rad_units'] = simulation_ds['rad_total'].attrs['units']

        desired_channels_ref = np.array(list(range(0, 19)) + [25])  # Python indexing starts at 0, so we use 0-18, 25 to get channels 1-19, 26
        desired_channels_rad = np.array(list(range(19, 25)) + list(
        range(26, 36)))  # Python indexing starts at 0, so we use 19-24, 26,35 to get channels 20-25, 27-36
        # Adding which bands are ref and which are rad
        new_ds.attrs['ref_bands_idx'] = desired_channels_ref + 1
        new_ds.attrs['rad_bands_idx'] = desired_channels_rad + 1

        new_ds.attrs['ref_long_name'] = 'All-sky upwelling reflectance at TOA'
        new_ds.attrs['rad_long_name'] = 'All-sky upwelling radiance at TOA'
                
        new_ds.attrs['chan_index'] = np.array(index_channel_relate_clouds) + 1
        
        channel_relate_clouds = name_channels[index_channel_relate_clouds]        
        new_ds['chan'] = channel_relate_clouds
        # -------------------------------------------------------

    return new_ds


def get_xarray_rad_ref_36channels(simulation_ds, data_type="ref_rad_total"):
    """
    Generate a new xarray Dataset merging reflectance and radiance data based on the provided data type.
    The function creates a new Dataset based on the 'ref_rad_total' data type, combining reflectance
    ("ref") and radiance ("rad") values from the input Dataset. The resulting Dataset has reflectance
    and radiance values in designated channel ranges.

    Parameters:
        simulation_ds (xarray.Dataset): Input xarray Dataset containing either reflectance or radiance data.
        data_type (str): Type of data to be processed. 'ref_rad_total' is supported, which
                         indicates a combination of reflectance and radiance.

    Returns:
        xarray.Dataset: A new Dataset with combined reflectance and radiance data, with attributes indicating
                        units, channels and descriptions for both reflectance and radiance.

    Notes:
        This function assumes specific channel ranges for reflectance and radiance values.
        The new Dataset retains units as separate attributes and deletes some unwanted attributes.
        For 'ref_rad_total', reflectance channels are in the range 1-19 (and 26), while radiance channels
        are in the range 21-25 and 27-36.
    """
    if data_type == "ref_rad_total" or data_type == "ref_all" or data_type == "group_5" or data_type == "group_7":
        new_ds = xr.full_like(simulation_ds["rad_total"], 0.0)
        new_ds.name = data_type

        # dimesions: time, chan, lat, lon
        new_ds[:, 0:19, :, :] = simulation_ds["ref_total"].isel(chan=slice(0, 19)).values  # refl 1-18
        new_ds[:, 19:25] = simulation_ds["rad_total"].isel(chan=slice(19, 25)).values  # rad 21-26
        new_ds[:, 25] = simulation_ds["ref_total"].isel(chan=25).values  # refl 20
        new_ds[:, 26:36] = simulation_ds["rad_total"].isel(chan=slice(26, 36)).values  # rad 27-36

        desired_channels_ref = np.array(
            list(range(0, 19)) + [25])  # Python indexing starts at 0, so we use 1-20 to get channels 1-19
        desired_channels_rad = np.array(list(range(19, 25)) + list(
            range(26, 36)))  # Python indexing starts at 0, so we use 1-20 to get channels 1-19

        # Delete unwanted attributes
        del new_ds.attrs['standard_name']
        del new_ds.attrs['long_name']
        del new_ds.attrs['units']

        # Store units as separate attributes
        new_ds.attrs['ref_units'] = simulation_ds['ref_total'].attrs['units']
        new_ds.attrs['rad_units'] = simulation_ds['rad_total'].attrs['units']

        # Adding which bands are ref and which are rad
        new_ds.attrs['ref_bands_idx'] = desired_channels_ref + 1
        new_ds.attrs['rad_bands_idx'] = desired_channels_rad + 1

        new_ds.attrs['ref_long_name'] = 'All-sky upwelling reflectance at TOA'
        new_ds.attrs['rad_long_name'] = 'All-sky upwelling radiance at TOA'

        channels_idx = np.arange(1, 37)
        combined_labels = [str(chan_idx) + ': ' + "{:.3f} μm".format(chan) for chan_idx, chan in
                           zip(channels_idx, new_ds.chan.values)]
        new_ds['chan'] = combined_labels

    return new_ds


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

def change_names_features(data_df):
    # Rename 'Reff_cloud_top' if it exists in the DataFrame
    if 'Reff_cloud_top' in data_df.columns:
        data_df = data_df.rename(columns={'Reff_cloud_top': r'$r_e$'})

    # Rename 'cot' if it exists in the DataFrame
    if 'cot' in data_df.columns:
        data_df = data_df.rename(columns={'cot': 'COT'})

    # Rename 'Nd_max' if it exists in the DataFrame
    if 'Nd_max' in data_df.columns:
        data_df = data_df.rename(columns={'Nd_max': r'$N_{d,\ max}$'})

    # Rename 'lwp' if it exists in the DataFrame
    if 'lwp' in data_df.columns:
        data_df = data_df.rename(columns={'lwp': 'LWP'})

    return data_df


def filter_datafraframe(df, variables_names):
    if variables_names == "LWP-Ndmax" or variables_names == "LWP-Ndmax-channels":
        max_value_threshold_ndmax = 2000
        max_value_threshold_lwp = 2000
        min_value_threshold = 2
        # max_value_threshold_ndmax = 1000
        # max_value_threshold_lwp = 800
        # min_value_threshold = 5
        print(f" ==================min_value_threshold:{min_value_threshold}, max_value_threshold_lwp: {max_value_threshold_lwp} , max_value_threshold_ndmax: {max_value_threshold_ndmax} =======================")        
        filtered_df = df.loc[
            (df['lwp'] > min_value_threshold) &
            (df['lwp'] < max_value_threshold_lwp) &
            (df['Nd_max'] > min_value_threshold) & 
            (df['Nd_max'] < max_value_threshold_ndmax) 
        # &
            # (df['cot'] > 1) 
        
            ].copy()
        # print(" ***************** Using COT higher that 1 ***************** ")

    elif variables_names == "Re-COT" or variables_names == "Re-COT-channels":
        max_value_threshold_re = 25
        max_value_threshold_cot = 100
        min_value_threshold = 0
        max_value_threshold = f"{max_value_threshold_re}_and_{max_value_threshold_cot}"
        filtered_df = df.loc[
            (df['cot'] > min_value_threshold) &
            (df['cot'] < max_value_threshold_cot) &
            (df['Reff_cloud_top'] > min_value_threshold) &
            (df['Reff_cloud_top'] < max_value_threshold_re)
            ].copy()

    return filtered_df



def save_df_to_netcdf(df, prefix, path_output):
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
    filename = f'{path_output}/{prefix}.nc'
    print(f"dataframe save in {filename}")
    xr_dataset.to_netcdf(filename)


def obtaining_x_y_dataframes(variables_names, time_test):
    """
    Read the data from the netcdf files and convert to dataframes
    Return: df_icon_ref_rad -- the first 2 columns correspond to the icon data and the next are the spectral bands

    """
    # ---------------------- Obtaining ICON  -------------------------------    if variables_names == "Re-COT-channels":
    if variables_names == "Re-COT":
        variable = ["Reff_cloud_top", "cot"]
        path_icon_timesteps = "/work/bb1036/rttov_share/dataset_ICON/icon_reff_cot_all_data.nc"

    elif variables_names == "LWP-Ndmax":
        variable = ["Nd_max", "lwp"]
        if time_test == "all_global_test":
            path_icon_timesteps = "/work/bb1036/rttov_share/dataset_ICON/icon_global_dataset.nc"
            path_icon_timesteps = "/work/bb1036/rttov_share/dataset_ICON/icon_test_global_dataset.nc"
            icon_test_ds = xr.open_dataset(path_icon_timesteps)
        elif time_test == "all_data_germany":
            path_icon_timesteps = "/work/bb1036/rttov_share/dataset_ICON/icon_germany_all_data.nc"

    
    print(f"path path_icon_timesteps: {path_icon_timesteps}")
    
    icon_all_ds = xr.open_dataset(path_icon_timesteps)
    icon_all_ds = icon_all_ds[variable]

    if icon_all_ds['time'].values.dtype == np.dtype('datetime64[ns]'):
        time_data = icon_all_ds['time'].values
        print("Formatted datetime64ns")

    elif icon_all_ds['time'].values.dtype == np.dtype('O'):  # 'O' is the dtype for Python objects
        print("Time is object type, converting...")
        icon_all_ds['time'] = pd.to_datetime(icon_all_ds['time'].values)
        time_data = icon_all_ds['time'].values
    else:
        print("Check variable type time ...")

    timesteps_to_remove = [np.datetime64(t) for t in ['2013-04-24T17:00', '2013-04-25T17:00', '2013-04-25T06:00', '2013-05-02T13:40']]
    for timestep in timesteps_to_remove:
        if timestep in icon_all_ds.time:
            icon_all_ds = icon_all_ds.drop_sel(time=timestep)
    
    if time_test == "all_global_test":
        icon_train_val_ds = icon_all_ds
        
    elif time_test == "all_data_germany":
        time_test_all_data = ['2013-04-24T12:00', '2013-04-25T12:00', '2013-05-02T12:00']
        icon_train_val_ds = icon_all_ds.drop_sel(time=time_test_all_data)
        icon_test_ds = icon_all_ds.sel(time=time_test_all_data)
    else:
        icon_train_val_ds = icon_all_ds.drop_sel(time=time_test)
        icon_test_ds = icon_all_ds.sel(time=time_test)


    df_icon_all = icon_all_ds.to_dataframe()
    df_icon_train_val = icon_train_val_ds.to_dataframe()
    df_icon_test = icon_test_ds.to_dataframe()


    # ---------------------- Obtaining spectral ------------------------------- 
    if time_test == "all_global_test":
        path_rad_ref_timesteps = "/work/bb1036/rttov_share/dataset_global_SPECTRAL/S3COM_global_dataset.nc"
        path_rad_ref_test_timesteps = "/work/bb1036/rttov_share/dataset_global_SPECTRAL/S3COM_test_global_dataset.nc"
        ref_rad_test_ds = xr.open_dataset(path_rad_ref_test_timesteps)
    elif time_test == "all_data_germany":      
        path_rad_ref_timesteps = "/work/bb1036/rttov_share/dataset_SPECTRAL/S3COM_OP_all_data_rad_ref.nc"

    print(f"path path_rad_ref_timesteps: {path_rad_ref_timesteps}")

    ref_rad_all_ds = xr.open_dataset(path_rad_ref_timesteps)
    ref_rad_all_ds = ref_rad_all_ds[['ref_total', "rad_total"]]
    ref_rad_all_ds['time'] = pd.to_datetime(ref_rad_all_ds['time'])

    for timestep in timesteps_to_remove:
        if timestep in ref_rad_all_ds.time:
            ref_rad_all_ds = ref_rad_all_ds.drop_sel(time=[timestep])
            
    if time_test == "all_global_test":
        ref_rad_train_val_ds = ref_rad_all_ds
    elif time_test == "all_data_germany":
        time_test = ['2013-04-24T12:00', '2013-04-25T12:00', '2013-05-02T12:00']
        ref_rad_train_val_ds = ref_rad_all_ds.drop_sel(time=time_test_all_data)
        ref_rad_test_ds = ref_rad_all_ds.sel(time=time_test_all_data)  
    else:
        ref_rad_train_val_ds = ref_rad_all_ds.drop_sel(time=time_test)
        ref_rad_test_ds = ref_rad_all_ds.sel(time=time_test)  

    ref_rad_all_ds = get_xarray_rad_ref(simulation_ds=ref_rad_all_ds,
                                    data_type="ref_rad_total")
    ref_rad_train_val_ds = get_xarray_rad_ref(simulation_ds=ref_rad_train_val_ds,
                                    data_type="ref_rad_total")
    ref_rad_test_ds = get_xarray_rad_ref(simulation_ds=ref_rad_test_ds,
                                         data_type="ref_rad_total")

    df_spectral_all = ref_rad_all_ds.to_dataframe().unstack(level=['chan'])
    df_spectral_all.columns = df_spectral_all.columns.droplevel(0)
    df_spectral_train_val = ref_rad_train_val_ds.to_dataframe().unstack(level=['chan'])
    df_spectral_train_val.columns = df_spectral_train_val.columns.droplevel(0)
    df_spectral_test = ref_rad_test_ds.to_dataframe().unstack(level=['chan'])
    df_spectral_test.columns = df_spectral_test.columns.droplevel(0)

    print(f"All data ---- input size {len(df_icon_all)}, output {len(df_spectral_all)}")
    print(f"Training and val ---- input size {len(df_icon_train_val)}, output {len(df_spectral_train_val)}")
    print(f"Test all data ---- input size {len(df_icon_test)}, output {len(df_spectral_test)}")

    df_icon_ref_rad_all = confirm_indices_and_join(df_icon_all, df_spectral_all)
    df_icon_ref_rad_train_val = confirm_indices_and_join(df_icon_train_val, df_spectral_train_val)
    df_icon_ref_rad_test_all = confirm_indices_and_join(df_icon_test, df_spectral_test)

    # --------------- Only Clouds ------------------------
    df_icon_ref_rad_all_clouds = df_icon_ref_rad_all[(df_icon_ref_rad_all.iloc[:, :2] != 0).all(axis=1)]
    df_icon_ref_rad_train_val_clouds = df_icon_ref_rad_train_val[(df_icon_ref_rad_train_val.iloc[:, :2] != 0).all(axis=1)]
    df_icon_ref_rad_test_all_clouds = df_icon_ref_rad_test_all[(df_icon_ref_rad_test_all.iloc[:, :2] != 0).all(axis=1)]

    print(f"Number of samples only clouds all data {len(df_icon_ref_rad_all_clouds)}")
    print(f"Number of sampless only clouds training and validation: {len(df_icon_ref_rad_train_val_clouds)}")
    print(f"Number of samples only clouds test: {len(df_icon_ref_rad_test_all_clouds)}")

    df_icon_ref_rad_all_clouds = df_icon_ref_rad_all_clouds.reset_index(level=0)  
    df_icon_ref_rad_train_val_clouds = df_icon_ref_rad_train_val_clouds.reset_index(level=0)  
    df_icon_ref_rad_test_all_clouds = df_icon_ref_rad_test_all_clouds.reset_index(level=0)  

    return df_icon_ref_rad_all_clouds, df_icon_ref_rad_train_val_clouds, df_icon_ref_rad_test_all_clouds



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
    print(f"dataframe save in {filename}")
    xr_dataset.to_netcdf(filename)
    

def main():
    "Checking the correlation in only data with clouds"

    parser = argparse.ArgumentParser(description='Create dataframes')
    arg = parser.add_argument

    # Add the arguments
    arg('--results_output_path', type=str, default="/work/bb1036/b381362/output/results_sensitivity/")
    arg('--variables_names', type=str, default="LWP-Ndmax", help='"Re-COT" List of the variables to use.')
    arg('--threshold', type=str, default="nofilter", help='filter, nofilter')
    # arg('--time_test', type=str, default="nofilter", help='filter, nofilter')
    arg('--cv_splits', type=int, default=5, help='number of flods to split the dataset')  

    args = parser.parse_args()
    results_output_path = args.results_output_path
    variables_names = args.variables_names
    threshold = args.threshold
    cv_splits = args.cv_splits

    time_test = "all_data_germany"

    df_icon_ref_rad_all_clouds, df_icon_ref_rad_train_val_clouds, df_icon_ref_rad_test_all_clouds = obtaining_x_y_dataframes(variables_names, time_test)


    # --------------------- filtering --------------------
    if threshold == "filter":
        df_icon_ref_rad_all_clouds = filter_datafraframe(df=df_icon_ref_rad_all_clouds,
                                                                           variables_names=variables_names)
        df_icon_ref_rad_train_val_clouds = filter_datafraframe(df=df_icon_ref_rad_train_val_clouds,
                                                                           variables_names=variables_names)

        df_icon_ref_rad_test_all_clouds = filter_datafraframe(df=df_icon_ref_rad_test_all_clouds,
                                                                          variables_names=variables_names)

    print(f"All data: Number of rows to work after {threshold}: {len(df_icon_ref_rad_all_clouds)}")
    print(f"Train-val: Number of rows to work after {threshold}: {len(df_icon_ref_rad_train_val_clouds)}")
    print(f"Test: Number of rows to work after {threshold}: {len(df_icon_ref_rad_test_all_clouds)}")


    # when it is save the index start from 0
    save_df_to_netcdf(df_icon_ref_rad_all_clouds.reset_index(drop=True), 'df_icon_ref_rad_clouds_filtered', results_output_path)

    # dividing the data in folds
    df_icon_ref_rad_all_clouds = df_icon_ref_rad_all_clouds.sort_values(by='time')  # Ensure chronological order
    df_icon_ref_rad_all_clouds['group'] = (df_icon_ref_rad_all_clouds['time'] != df_icon_ref_rad_all_clouds['time'].shift()).cumsum()  # 

    
    if variables_names == "LWP-Ndmax":
        variable = ["Nd_max", "lwp"]    

    X = df_icon_ref_rad_all_clouds.drop(columns=variable)  # Drop non-feature columns


    y = df_icon_ref_rad_all_clouds[variable]
    groups = df_icon_ref_rad_all_clouds['group']  
    gkf = GroupKFold(n_splits=cv_splits)
                    

    # Create folds
    folds = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # Set 'time' as the first level of the MultiIndex, followed by 'lat' and 'lon'
        X_val = X_val.reset_index()  
        X_val = X_val.set_index(['time', 'lat', 'lon'])  
        y_val.index = X_val.index
        folds.append((X_train, X_val, y_train, y_val))

    # Access individual folds
    for fold_number, (X_train, X_val, y_train, y_val) in enumerate(folds):
        print(f"Fold {fold_number + 1}")
        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
        print(f"Groups in Train: {len(X_train['group'].unique())}: {X_train['group'].unique()}")
        print(f"Groups in Validation: {len(X_val['group'].unique())}: {X_val['group'].unique()}")
        X_train = X_train.drop(columns=['group', 'time'])  # Drop non-feature columns
        X_val = X_val.drop(columns=['group'])  # Drop non-feature columns

        scaler_x_type = "StandardScaler"
        if scaler_x_type == "StandardScaler":
            scaler_ref = preprocessing.StandardScaler().fit(X_train) 
        elif scaler_x_type == "MinMaxScaler":
            scaler_ref = preprocessing.MinMaxScaler().fit(X_train)  
    
        if isinstance(scaler_ref, (MinMaxScaler, StandardScaler)):
            df_ref_train = pd.DataFrame(scaler_ref.transform(X_train), columns=X_train.columns)
            df_ref_val = pd.DataFrame(scaler_ref.transform(X_val), columns=X_val.columns)

        # ----------------- Scaling the output -----------------
        scaler_y_type = "StandardScaler"
        if scaler_y_type == "StandardScaler":
            scaler_icon = preprocessing.StandardScaler().fit(y_train) 
        elif scaler_y_type == "MinMaxScaler":
            scaler_icon = preprocessing.MinMaxScaler().fit(y_train)
            
        if isinstance(scaler_icon, (MinMaxScaler, StandardScaler)):
            df_icon_train = pd.DataFrame(scaler_icon.transform(y_train), columns=y_train.columns)
            df_icon_val = pd.DataFrame(scaler_icon.transform(y_val), columns=y_val.columns)


        # ------------------- save files -------------------
        save_df_to_netcdf_fold(df_icon_train, fold_number, 'df_icon_train', results_output_path)
        save_df_to_netcdf_fold(df_icon_val, fold_number, 'df_icon_val', results_output_path)
    
        save_df_to_netcdf_fold(df_ref_train, fold_number, 'df_ref_train', results_output_path)
        save_df_to_netcdf_fold(df_ref_val, fold_number, 'df_ref_val', results_output_path)
    
        # ------------------- save scaler, pca -------------------
        
        if isinstance(scaler_ref, (MinMaxScaler, StandardScaler)):
            pk.dump(scaler_ref, open(f'{results_output_path}/scaler_ref_k_fold_{fold_number}.npy', 'wb'))
        if isinstance(scaler_icon, (MinMaxScaler, StandardScaler)):
            pk.dump(scaler_icon, open(f'{results_output_path}/scaler_{variables_names}_k_fold_{fold_number}.npy', 'wb'))
        
        # Get unique values from the 'time' level (level 0)
        val_times = np.array(X_val.index.get_level_values(0).unique())

        with open(f"{results_output_path}/val_data_k_fold_{fold_number}.pkl", 'wb') as f:
            pk.dump({'val_times': val_times}, f)

        index_array = X_val.index.values  # Extract the index as a NumPy array
        
        # Save the index to a .npy file
        np.save(f'{results_output_path}/index_only_clouds_all_val_{fold_number}.npy', index_array)




if __name__ == '__main__':
    main()