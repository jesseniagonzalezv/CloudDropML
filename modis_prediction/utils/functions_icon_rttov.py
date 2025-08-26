
import xarray as xr
import numpy as np
import pandas as pd
import xarray as xr
import sys
import os 

from utils.functions_ml import get_prediction_nd_lwp
from utils.functions_general import filter_dataframe_cloud, combine_lwp_nd_dataframe
from plotting.plots import plot_joint_histogram, plot_joint_histogram_conditional


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_lwp_nd_icon(path, variables, result_folder, lat_min, lat_max, min_value_threshold, max_value_threshold):
    """
    Read the netcdf file and select a specific area if it is needed and filter the data in the range defined
    Args:
        -- path: path of netcdf file
        -- variable (list): variable to read in this case Nd and lwp ["Nd_max", "lwp"]
        -- lat_min (float): value minimun in degrees
        -- lat_max (float): value maximun in degrees
        -- min_value_threshold (float): value of the minimun range for lwp and nd
        -- max_value_threshold (float): value of the minimun range for lwp and nd

    Return:
        -- filtered_df_icon_les (dataframe): data with 2 columns Nd and lwp after filter
    """
    
    icon_ds = xr.open_dataset(path)
    icon_ds = icon_ds[variables].sel(lat=slice(lat_min, lat_max))
    icon_ds = icon_ds[variables]
    # .sel(lat=slice(lat_min, lat_max))
    print(icon_ds)
    # print(icon_ds)
    icon_ds['lat'] = icon_ds['lat'].astype(np.float32)
    icon_ds['lon'] = icon_ds['lon'].astype(np.float32)
    
    filtered_df_icon_les = filter_dataframe_cloud(icon_ds.to_dataframe(),
                                                  min_value_threshold=min_value_threshold, 
                                                  max_value_threshold=max_value_threshold)
    print(filtered_df_icon_les.describe())
    
    # fig, ax = plot_joint_histogram(filtered_df_icon_les, variable[0], variable[1], threshold="filterlog", figure_size=(6, 6))
    # fig, ax = plot_joint_histogram(filtered_df_icon_les, variable[0], variable[1], threshold="range3-870", figure_size=(6, 6))
    fig, ax = plot_joint_histogram(filtered_df_icon_les, 
                                   filtered_df_icon_les.columns[0], 
                                   filtered_df_icon_les.columns[1], 
                                   threshold="range7-870", 
                                   figure_size=(6, 6))
    path_file = f"{result_folder}/Nd_lwp_icon_les.png"
    fig.savefig(path_file, bbox_inches="tight")
    
    fig, ax = plot_joint_histogram_conditional(filtered_df_icon_les, 
       filtered_df_icon_les.columns[0], 
       filtered_df_icon_les.columns[1], 
       threshold="range7-870", 
       figure_size=(6, 6))         

    path_file = f"{result_folder}/Nd_lwp_icon_les_CP.png"
    fig.savefig(path_file, bbox_inches="tight")

    
    return filtered_df_icon_les


def get_xarray_rad_ref_17channels(simulation_ds, data_type="ref_rad_total"):
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

        if len(simulation_ds.chan.values) == 36:
            new_ds = xr.DataArray(dims=["chan", "lat", "lon"], 
                                  coords={"chan": simulation_ds.chan[index_channel_relate_clouds],
                                          "lat": simulation_ds.lat, "lon": simulation_ds.lon}, 
                                  name=data_type)

            index_channel_relate_clouds_ref = [0,1,2,3,4,6]  # reflectances 1-29, 26 (0-18, 25) 
            index_channel_relate_clouds_rad = [19,20,21,22,28,30,31,32,33,34,35]  # radiances 20-23, 29, 31-36
            new_ds[0:6, :, :] = simulation_ds["ref_total"].isel(chan=index_channel_relate_clouds_ref)
            new_ds[6:17, :, :] = simulation_ds["rad_total"].isel(chan=index_channel_relate_clouds_rad)

    
        elif len(simulation_ds.chan.values) == 18:
            new_ds = xr.full_like(simulation_ds["rad_total"], 0.0)
            new_ds.name = data_type
            new_ds = xr.DataArray(dims=["chan", "lat", "lon"], 
                                  coords={
                                          "chan": simulation_ds.chan[index_channel_relate_clouds],
                                          "lat": simulation_ds.lat, "lon": simulation_ds.lon}, 
                                  name=data_type)

            index_channel_relate_clouds_ref = [0,1,2,3,4,6]  # reflectances 1-29, 26 (0-18, 25) 
            index_channel_relate_clouds_rad = [19,20,21,22,28,30,31,32,33,34,35]  # radiances 20-23, 29, 31-36
            new_ds[0:6, :, :] = simulation_ds["ref_total"].isel(chan=[0,1,2,3,4,6])
            new_ds[6:17, :, :] = simulation_ds["rad_total"].isel(chan=slice(7,18))
    
        # Store units as separate attributes
        new_ds.attrs['ref_units'] = simulation_ds['ref_total'].attrs['units']
        new_ds.attrs['rad_units'] = simulation_ds['rad_total'].attrs['units']

        desired_channels_ref = np.array(list(range(0, 19)) + [25])  # Python indexing starts at 0, so we use 0-18, 25 to get channels 1-19, 26
        desired_channels_rad = np.array(list(range(19, 25)) + list(
        range(26, 36)))  # Python indexing starts at 0, so we use 19-24, 26,35 to get channels 20-25, 27-36

        new_ds.attrs['ref_bands_idx'] = desired_channels_ref + 1
        new_ds.attrs['rad_bands_idx'] = desired_channels_rad + 1

        new_ds.attrs['ref_long_name'] = 'All-sky upwelling reflectance at TOA'
        new_ds.attrs['rad_long_name'] = 'All-sky upwelling radiance at TOA'
                
        new_ds.attrs['chan_index'] = np.array(index_channel_relate_clouds) + 1
        
        channel_relate_clouds = name_channels[index_channel_relate_clouds]        
        new_ds['chan'] = channel_relate_clouds

    return new_ds


def get_xarray_rad_ref_18channels(simulation_ds, data_type="ref_rad_total"):
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
    # ---------------------- only 18 channels -------------------------------    
    index_channel_relate_clouds = [0,1,2,3,4,5,6,19,20,21,22,28,30,31,32,33,34,35]
    

    if data_type == "ref_rad_total" or data_type == "ref_all":
        # new_ds = xr.full_like(simulation_ds["rad_total"], 0.0)

        if len(simulation_ds.chan.values) == 36:
            # print(len(simulation_ds.chan.values) )
            new_ds = xr.DataArray(dims=["chan", "lat", "lon"], 
                                  coords={"chan": simulation_ds.chan[index_channel_relate_clouds],
                                          "lat": simulation_ds.lat, "lon": simulation_ds.lon}, 
                                  name=data_type)
            # new_ds = xr.DataArray(dims=["time", "chan", "lat", "lon"], 
            #                       coords={"time": simulation_ds.time, 
            #                               "chan": simulation_ds.chan[index_channel_relate_clouds],
            #                               "lat": simulation_ds.lat, "lon": simulation_ds.lon}, 
            #                       name=data_type)

            # dimesions: time, chan, lat, lon
            index_channel_relate_clouds_ref = [0,1,2,3,4,5,6]  # reflectances 1-29, 26 (0-18, 25) 
            index_channel_relate_clouds_rad = [19,20,21,22,28,30,31,32,33,34,35]  # radiances 20-23, 29, 31-36
            # dimesions: time, chan, lat, lon
            # new_ds[:, 0:7, :, :] = simulation_ds["ref_total"].isel(chan=index_channel_relate_clouds_ref)
            # new_ds[:, 7:18, :, :] = simulation_ds["rad_total"].isel(chan=index_channel_relate_clouds_rad)

            new_ds[0:7, :, :] = simulation_ds["ref_total"].isel(chan=index_channel_relate_clouds_ref)
            new_ds[7:18, :, :] = simulation_ds["rad_total"].isel(chan=index_channel_relate_clouds_rad)
    
        elif len(simulation_ds.chan.values) == 18:
            new_ds = xr.full_like(simulation_ds["rad_total"], 0.0)
            new_ds.name = data_type

            index_channel_relate_clouds_ref = [0,1,2,3,4,5,6]  # reflectances 1-29, 26 (0-18, 25) 
            index_channel_relate_clouds_rad = [19,20,21,22,28,30,31,32,33,34,35]  # radiances 20-23, 29, 31-36
            # dimesions: time, chan, lat, lon
            # new_ds[:, 0:7, :, :] = simulation_ds["ref_total"].isel(chan=slice(0,7))
            # new_ds[:, 7:18, :, :] = simulation_ds["rad_total"].isel(chan=slice(7,18))

            new_ds[0:7, :, :] = simulation_ds["ref_total"].isel(chan=slice(0,7))
            new_ds[7:18, :, :] = simulation_ds["rad_total"].isel(chan=slice(7,18))
    
            
        # # Delete unwanted attributes
        # del new_ds.attrs['standard_name']
        # del new_ds.attrs['long_name']
        # del new_ds.attrs['units']


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

        # ---------------------- all channels -------------------------------
        # channels_idx = np.arange(1, 37)
        # combined_labels = [str(chan_idx) + ': ' + "{:.3f} μm".format(chan) for chan_idx, chan in
        #                    zip(channels_idx, new_ds.chan.values)]
        
        # new_ds.attrs['chan_index'] = np.array(list(channels_idx)) + 1
        # -------------------------------------------------------------------
                
        new_ds.attrs['chan_index'] = np.array(index_channel_relate_clouds) + 1
        
        channel_relate_clouds = name_channels[index_channel_relate_clouds]        
        new_ds['chan'] = channel_relate_clouds
        # -------------------------------------------------------

    return new_ds




    
def nd_lwp_ml_prediction_with_rttov_channels(path_icon, path_rttov, all_data, num_channels, path_model_file_lwp, path_model_file_nd, path_dataframes_scaler_lwp, path_dataframes_scaler_nd, fold_num, variables, variables_names, lat_min=50, lat_max=54.50, min_value_threshold=5, max_value_threshold=2000):
    # ------------------------------------- ICON LES---------------------------------------------- Useful to mask only liquid clouds
    icon_ds = xr.open_dataset(path_icon)
    icon_ds = icon_ds[variables].sel(lat=slice(lat_min, lat_max))
    # print(icon_ds)
    icon_ds['lat'] = icon_ds['lat'].astype(np.float32)
    icon_ds['lon'] = icon_ds['lon'].astype(np.float32)

    # ------------------------------------- RTTOV simulations----------------------------------------------
    ds_channels_rttov = xr.open_dataset(path_rttov).sel(lat=slice(lat_min, lat_max))
    if num_channels=="18_chan":
        ds_rttov = get_xarray_rad_ref_18channels(ds_channels_rttov, data_type="ref_rad_total")
    elif num_channels=="17_chan":
        ds_rttov = get_xarray_rad_ref_17channels(ds_channels_rttov, data_type="ref_rad_total")

    # -------------------------------  Mask only liquid water clouds -----------------------------------------
    # df_icon = icon_ds.to_dataframe()
    # df_icon_only_clouds = df_icon[(df_icon.iloc[:, :2] != 0).all(axis=1)]
    
    # Create a mask where either var1 or var2 is zero
    # mask_clouds = (icon_ds.Nd_max == 0) | (icon_ds.lwp == 0)
    # when i trained the data was clean for points less that 2000 and higher that 2
    mask_clouds = (icon_ds.Nd_max == 0) | (icon_ds.lwp == 0) | (icon_ds.Nd_max <= 2) | (icon_ds.Nd_max >= 2000) | (icon_ds.lwp <= 2) | (icon_ds.lwp >= 2000)
    
    # print(np.shape(mask_clouds))
    
    mask_expanded = xr.DataArray(mask_clouds, dims=['lat', 'lon'])
    mask_broadcasted = xr.broadcast(mask_expanded, ds_rttov)[0]  # Automatically aligns and broadcasts
    print("Before transpose",mask_broadcasted.dims)  # Before transpose
    mask_broadcasted = mask_broadcasted.transpose('chan', 'lat', 'lon')
    # print(np.shape(mask_broadcasted))
    print("After transpose",mask_broadcasted.dims)  # Before transpose
    print(f"mask_broadcasted, {np.shape(mask_broadcasted)}")

    # Apply mask to ref_rad_total
    masked_ds_rttov = ds_rttov.where(~mask_broadcasted)
    print(f"masked_ds_rttov, {np.shape(ds_rttov)}")

    # ---------------------------- prediction -------------------------------------------------------------------------
    prediction_rttov_xarray_nd, prediction_rttov_xarray_lwp = get_prediction_nd_lwp(ds_channels=masked_ds_rttov,
                                                                                    all_data=all_data,
                                                                                    num_channels=num_channels,
                                                                                    path_model_file_lwp=path_model_file_lwp, 
                                                                                    path_model_file_nd=path_model_file_nd, 
                                                                                    path_dataframes_scaler_lwp=path_dataframes_scaler_lwp,
                                                                                    path_dataframes_scaler_nd=path_dataframes_scaler_nd,
                                                                                    variables_names=variables_names, 
                                                                                    fold_num=fold_num)
    
    pred_lwp_nd_from_rttov_df = combine_lwp_nd_dataframe(prediction_xarray_nd=prediction_rttov_xarray_nd,
                                                      prediction_xarray_lwp=prediction_rttov_xarray_lwp)
    
    # ---------------------------- filter and plot -------------------------------------------------------------------------
    
    filtered_df_ml_rttov = filter_dataframe_cloud(pred_lwp_nd_from_rttov_df,
                                                  min_value_threshold=min_value_threshold,
                                                  max_value_threshold=max_value_threshold)

    return filtered_df_ml_rttov, ds_rttov, masked_ds_rttov


def obtain_nd_lwp_ml_prediction_with_rttov_channels(path_icon, path_rttov, all_data, path_model_file_lwp, path_model_file_nd, path_dataframes_scaler_lwp, path_dataframes_scaler_nd, fold_num, variables, variables_names, lat_min, lat_max, min_value_threshold, max_value_threshold, result_folder, num_channels="18_chan"):
    
    filtered_df_ml_rttov, ds_rttov, masked_ds_rttov = nd_lwp_ml_prediction_with_rttov_channels(path_icon=path_icon, 
                                                                                               path_rttov=path_rttov,
                                                                                               all_data=all_data, 
                                                                                               num_channels=num_channels, 
                                                                                               path_model_file_lwp=path_model_file_lwp,
                                                                                               path_model_file_nd=path_model_file_nd,
                                                                                               path_dataframes_scaler_lwp=path_dataframes_scaler_lwp, 
                                                                                               path_dataframes_scaler_nd=path_dataframes_scaler_nd,
                                                                                               fold_num=fold_num, 
                                                                                               variables=variables, 
                                                                                               variables_names=variables_names, 
                                                                                               lat_min=lat_min, 
                                                                                               lat_max=lat_max,
                                                                                               min_value_threshold=min_value_threshold,
                                                                                               max_value_threshold=max_value_threshold)


    path_file = f"{result_folder}/nd_lwp_ml_prediction_with_rttov_channels.png"
    # fig, ax = plot_joint_histogram(filtered_df_ml_rttov, variable[0], variable[1], threshold="range3-870", figure_size=(6, 6))
    fig, ax = plot_joint_histogram(filtered_df_ml_rttov, 
                                   filtered_df_ml_rttov.columns[0], 
                                   filtered_df_ml_rttov.columns[1], 
                                   threshold="range7-870", 
                                   figure_size=(6, 6))
    fig.savefig(path_file, bbox_inches="tight")

    fig, ax = plot_joint_histogram_conditional(filtered_df_ml_rttov, 
       filtered_df_ml_rttov.columns[0], 
       filtered_df_ml_rttov.columns[1], 
       threshold="range7-870", 
       figure_size=(6, 6))         
    path_file = f"{result_folder}/nd_lwp_ml_prediction_with_rttov_channels_CP.png"
    fig.savefig(path_file, bbox_inches="tight")


    return filtered_df_ml_rttov, ds_rttov, masked_ds_rttov
    

def get_lwp_Nd(path_icon, v_reff=False):
    # variable_2D,variable_3D,
    """
    Arguments:
    path_icon -- ncfile with the ICON dataset  ('/work/bb1036/dataset/data_rttov_T12.nc')
    Returns:
    new netcdf file with the variables in the same folder as the input data but with the extension of _Reff      
   ["lwp", "Nd_max"]
    Variables created
     lwp -- liquid water path (gm-2)
     Nd_max -- cloud droplet number concentration maximum on the height values (cm-3)
    """
    fname = path_icon
    name_output = os.path.splitext(fname)[0] + "_lwp_Nd.nc"
    
    ds = xr.open_dataset(fname).sel(lat=slice(47.6, 54.50))
    T_c = np.float64(ds.ta) - 273.15
    esat_2013 = 0.611 * np.exp((17.3 * T_c) / (T_c + 237.3)) * 1000.0
    # esat_2013 = np.ma.masked_array(esat_2013,  esat_2013 == 0) ## check it!!!!!!!!
    pres = np.ma.masked_array(ds.pres, ds.pres == 0)  ## check it!!!!!!!!
    qs_2013 = 0.622 * (esat_2013 / pres)  # this is diffent compared with Alexandre code
    r_2013 = ds.hus / (1 - ds.hus)
    RH_2013 = 100 * (r_2013 / qs_2013)
    pv_2013 = (esat_2013 * RH_2013) / 100.0
    pd_2013 = ds.pres - pv_2013
    rho_2013 = (pd_2013 / (287.058 * ds.ta)) + (pv_2013 / (461.495 * ds.ta))  # nana
    cdnc_2013_cm = (rho_2013 * ds.qnc) / 1000000  # convert to cm^-3

    Nd_max = np.nanmax(cdnc_2013_cm, axis=0)
    ds["Nd_max"] = (['lat', 'lon'], Nd_max)  # this is an array
    ds.Nd_max.attrs['units'] = "cm-3"
    ds.Nd_max.attrs['standard_name'] = "Nd_max"
    ds.Nd_max.attrs['long_name'] = "Cloud dropler number maximun"

    ds["Nd"] = cdnc_2013_cm  # thi is a xarray.DataArray
    ds.Nd.attrs['units'] = "cm-3"
    ds.Nd.attrs['standard_name'] = "Nd"
    ds.Nd.attrs['long_name'] = "Cloud dropler number in each layer"


    
    ds = ds.assign(lwp=ds.clwvi * 1000)
    ds.lwp.attrs['units'] = "gm-2"
    ds.lwp.attrs['standard_name'] = "LWP"
    ds.lwp.attrs['long_name'] = "Liquid water path"


    # -----------------Calculation of the Reff -------------------------
    L = rho_2013 * ds.clw  # in kgm^-3 #cannot set variable with 3-dimensional data without explicit dimension names. Pass a tuple of (dims, data) instead.

    N = rho_2013 * ds.qnc  # im m^-3
    N2 = np.ma.masked_array(N, N < 2.0e+06)  ## check it!!!!!!!! 2cm'3

    # L.where(L == 0, np.Nan, L)
    #N.where(N < 2.0e+06, np.NAN, N) # !! ask DIPU
    ######constant for size distribution #############
    nu = 1.0
    mu = 1.0
    a = 1.24E-01
    b = 1 / 3
    
    variable_calculated = ["lwp", "Nd_max"]
    
    if v_reff==True:
        ###################################
        reff_2013 = (a / 2) * (gamma((3 * b + nu + 1) / (mu)) / gamma((2 * b + nu + 1) / (mu))) * (
                    (L / N2) * (gamma((nu + 1) / (mu)) / gamma((nu + 2) / mu))) ** (b)  # m
        reff_2013 = reff_2013 * 1E6

        ds["Reff"] = reff_2013  # thi is a xarray.DataArray
        ds.Reff.attrs['units'] = "Micron"
        ds.Reff.attrs['standard_name'] = "Reff"
        ds.Reff.attrs['long_name'] = "Cloud effective radius"

        print("Reff min, max", reff_2013.min().values, np.max(reff_2013).values)
        
        # variable_calculated = ["Reff", "lwp", "Nd_max"]
        variable_calculated = ["lwp", "Nd_max"]


    # variables_total = variable_2D + variable_3D  + variable_calculated + ["time"]
    variables_total = variable_calculated + ["time"]

    ds = ds.get(variables_total)
    
    ds.to_netcdf(name_output)  # '/work/bb1036/b381362/dataset/3D_mod.nc') # rewrite to netcdf
    print("generated the next file:", name_output)
    # ds.close()
    return ds #["lwp"], ds["Nd_max"]




