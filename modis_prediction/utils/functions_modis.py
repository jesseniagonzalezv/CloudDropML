

import xarray as xr
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os 

from netCDF4 import Dataset
from pyresample import geometry
from pyresample import kd_tree

from utils.functions_general import select_channels_new, ref_to_dataframe, filter_dataframe_cloud, calculate_nd_with_ref_cod, combine_lwp_nd_dataframe, confirm_indices_and_join
from utils.functions_ml import get_prediction_nd_lwp

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting.plots import plot_joint_histogram, plot_joint_histogram_conditional



def reprojection_data(data, map_boundaries, lat_swath, lon_swath):
    
    data = np.array(data)
    
    grid_lat_min, grid_lat_max = map_boundaries[0], map_boundaries[1]  # degrees North
    grid_lon_min, grid_lon_max = map_boundaries[2], map_boundaries[3]  # degrees East
    
    # grid_resolution_lat = 0.0089
    # grid_resolution_lon = 0.014

    # grid_resolution_lat = 0.011  #OLD  1km
    # grid_resolution_lon = 0.017  #OLD

    grid_resolution_lat = 0.017  
    grid_resolution_lon = 0.017

    # grid_lons = np.arange(grid_lon_min, grid_lon_max + grid_resolution_lon, grid_resolution_lon)
    # grid_lats = np.arange(grid_lat_min, grid_lat_max + grid_resolution_lat, grid_resolution_lat)

    grid_lons = np.arange(grid_lon_min, grid_lon_max, grid_resolution_lon)
    grid_lats = np.arange(grid_lat_min, grid_lat_max, grid_resolution_lat)

    # Create a meshgrid (2D arrays) for the target grid
    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)
    
    # Define the area for the target grid
    area_def = geometry.AreaDefinition(
        'latlon_grid', 'Regular lat-lon grid', 'latlon',
        {'proj': 'latlong'}, grid_lon.shape[1], grid_lon.shape[0],
        [grid_lon_min, grid_lat_min, grid_lon_max, grid_lat_max]
    )
    
    # Define the swath geometry
    swath_def = geometry.SwathDefinition(lons=lon_swath, lats=lat_swath)
    
    # radius_of_influence = 1500  # in meters
    radius_of_influence = 500  # in meters
    
    # Perform the resampling using nearest neighbor
    data_resampled = kd_tree.resample_nearest(
        swath_def, data, area_def,
        radius_of_influence=radius_of_influence, fill_value=np.nan
    )

    return data_resampled, grid_lon, grid_lat


def reprojection_data_global(data, lat_swath, lon_swath, resolution=(0.011, 0.017)):
    data = np.array(data)

    # Full extent of the swath
    lat_min, lat_max = np.nanmin(lat_swath), np.nanmax(lat_swath)
    lon_min, lon_max = np.nanmin(lon_swath), np.nanmax(lon_swath)

    grid_resolution_lat, grid_resolution_lon = resolution

    # Create target grid
    grid_lats = np.arange(lat_min, lat_max, grid_resolution_lat)
    grid_lons = np.arange(lon_min, lon_max, grid_resolution_lon)
    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)

    # Area definition for output
    area_def = geometry.AreaDefinition(
        'latlon_grid', 'Regular lat-lon grid', 'latlon',
        {'proj': 'latlong'}, grid_lon.shape[1], grid_lon.shape[0],
        [lon_min, lat_min, lon_max, lat_max]
    )

    # Swath definition from input lats/lons
    swath_def = geometry.SwathDefinition(lons=lon_swath, lats=lat_swath)

    # Resample with a reasonable radius
    data_resampled = kd_tree.resample_nearest(
        swath_def, data, area_def,
        radius_of_influence=500, fill_value=np.nan   #500
    )

    return data_resampled, grid_lon, grid_lat


def convert_to_xarray_swath(df, all_index, ref_rad_18_chan):
    df_full_lwp = df["lwp"].reindex(all_index)
    df_full_nd = df["Nd_max"].reindex(all_index)
    
    pred_2d_lwp = df_full_lwp.unstack()  # Returns a DataFrame with shape (lat, lon)
    pred_2d_nd = df_full_nd.unstack()  # Returns a DataFrame with shape (lat, lon)
    
    new_ds = xr.Dataset(
        data_vars={
            "lwp": (("lat", "lon"), pred_2d_lwp.values),
            "Nd_max": (("lat", "lon"), pred_2d_nd.values),
        },
        coords={
            "lat": (("lat", "lon"), ref_rad_18_chan["lat"].values),
            "lon": (("lat", "lon"), ref_rad_18_chan["lon"].values),
        }
    )
    
    return new_ds
# def projection_level2(masked_subset_level_ds, map_boundaries):
#     # variable_name = "lwp"
#     data_resample_all = {}
#     for variable_name in ["lwp", "Nd", "cldph"]:
#         data = masked_subset_level_ds.variables[variable_name]  # Replace with your data variable name
#         lat_all_masked_liquid_germany = masked_subset_level_ds.variables['lat'][:] 
#         lon_all_masked_liquid_germany = masked_subset_level_ds.variables['lon'][:]

        
        

#         if map_boundaries is not None:
#             # Reproject the data for this channel
#             data_resampled, grid_lon, grid_lat = reprojection_data(data=data,
#                                                                lat_swath = lat_all_masked_liquid_germany,
#                                                                lon_swath = lon_all_masked_liquid_germany,
#                                                                map_boundaries=map_boundaries)
            
#         else:
    
#             data_resampled, grid_lon, grid_lat = reprojection_data_global(data=data, 
#                                                                           lat_swath=lat_all_masked_liquid_germany, 
#                                                                           lon_swath=lon_all_masked_liquid_germany, 
#                                                                           resolution=(0.011, 0.017)) # 0.017



        
#         data_resample_all[variable_name] = data_resampled

    
#     new_ds = xr.Dataset(
#             data_vars={
#                 "lwp": (("lat", "lon"), np.full((len(grid_lat[:,0]), len(grid_lon[0,:])), np.flipud(data_resample_all["lwp"]))),  
#                 "Nd_max": (("lat", "lon"), np.full((len(grid_lat[:,0]), len(grid_lon[0,:])), np.flipud(data_resample_all["Nd"]))),  
#                 "cldph": (("lat", "lon"), np.full((len(grid_lat[:,0]), len(grid_lon[0,:])), np.flipud(data_resample_all["cldph"]))),  
#             },
#             coords={
#                 "lat": grid_lat[:,0],  
#                 "lon": grid_lon[0,:], 
#             },
#         )
       
#     return new_ds
#     # , data_resample_all, grid_lon, grid_lat



def projection_level2_swath_predicted(masked_subset_level_ds, map_boundaries): # change name to projection_level2_swath_predicted
    # variable_name = "lwp"
    data_resample_all = {}
    # for variable_name in ["lwp", "Nd", "cldph"]:
    for variable_name in ["lwp", "Nd_max"]:
        data = masked_subset_level_ds.variables[variable_name]  # Replace with your data variable name
        lat_all_masked_liquid_germany = masked_subset_level_ds.variables['lat'][:] 
        lon_all_masked_liquid_germany = masked_subset_level_ds.variables['lon'][:]

        
        

        if map_boundaries is not None:
            # Reproject the data for this channel
            data_resampled, grid_lon, grid_lat = reprojection_data(data=data,
                                                               lat_swath = lat_all_masked_liquid_germany,
                                                               lon_swath = lon_all_masked_liquid_germany,
                                                               map_boundaries=map_boundaries)
            
        else:
    
            data_resampled, grid_lon, grid_lat = reprojection_data_global(data=data, 
                                                                          lat_swath=lat_all_masked_liquid_germany, 
                                                                          lon_swath=lon_all_masked_liquid_germany, 
                                                                          resolution=(0.017, 0.017)) # 0.017



        
        data_resample_all[variable_name] = data_resampled

    
    new_ds = xr.Dataset(
            data_vars={
                "lwp": (("lat", "lon"), np.full((len(grid_lat[:,0]), len(grid_lon[0,:])), np.flipud(data_resample_all["lwp"]))),  
                "Nd_max": (("lat", "lon"), np.full((len(grid_lat[:,0]), len(grid_lon[0,:])), np.flipud(data_resample_all["Nd_max"]))),  
                # "cldph": (("lat", "lon"), np.full((len(grid_lat[:,0]), len(grid_lon[0,:])), np.flipud(data_resample_all["cldph"]))),  
            },
            coords={
                "lat": grid_lat[:,0],  
                "lon": grid_lon[0,:], 
            },
        )
       
    return new_ds



def projection_level1(masked_subset_level_ds, map_boundaries):
    data_resampled_all = []
    for i in range(len(masked_subset_level_ds.chan)):
        data = masked_subset_level_ds["ref_rad_modis"][i]  # shape: (y, x)
        lat = masked_subset_level_ds["lat"]
        lon = masked_subset_level_ds["lon"]
        

        if map_boundaries is not None:
            # Reproject the data for this channel
            data_resampled, grid_lon, grid_lat = reprojection_data(data=data,
                                                                  lat_swath=lat,
                                                                  lon_swath=lon,
                                                                  map_boundaries=map_boundaries)
        else:
    
            data_resampled, grid_lon, grid_lat = reprojection_data_global(data=data, 
                                                                          lat_swath=lat, 
                                                                          lon_swath=lon, 
                                                                          resolution=(0.017, 0.017)) #0.017


        
        # Store the result for this channel
        data_resampled_all.append(np.flipud(data_resampled))

    print(np.shape(data_resampled_all))
    # Stack the list into a 3D NumPy array: (chan, lat, lon)
    ref_rad_stack = np.stack(data_resampled_all, axis=0)
    
    new_ds = xr.Dataset(
            data_vars={
                "ref_rad_modis": (("chan", "lat", "lon"), np.full((len(masked_subset_level_ds.chan),len(grid_lat[:,0]), len(grid_lon[0,:])), data_resampled_all)),  
            },
            coords={
                "chan": masked_subset_level_ds.chan.values,
                "lat": grid_lat[:,0],  
                "lon": grid_lon[0,:], 
            },
        )

        # new_ds.to_netcdf("/scratch/b/b381362/test_gridding/11_40_files/modis_level1_11_40.nc")

       
    return new_ds



def rename_variables_dimension_2d(input_file, output_file):
    # Open the NetCDF file for reading
    with Dataset(input_file, "r") as src:
        # Create a new NetCDF file for writing
        with Dataset(output_file, "w") as dst:
            # Copy global attributes
            dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})
            
            # Copy dimensions
            for name, dimension in src.dimensions.items():
                dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
            
            # Copy variables
            for name, variable in src.variables.items():
                # Rename problematic variables
                if name == "Latitude":
                    new_name = "Latitude_var"
                elif name == "Longitude":
                    new_name = "Longitude_var"
                else:
                    new_name = name
                
                # Create variable in the new file
                dst_var = dst.createVariable(new_name, variable.datatype, variable.dimensions)
                dst_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
                dst_var[:] = variable[:]
    return dst_var

# rename_variables_dimension_2d(input_file, output_file)


def subset_level2_nd_lwp(path, map_boundaries):
    modis_level2_ds = xr.open_dataset(path)
    latitude = modis_level2_ds["lat"]
    longitude = modis_level2_ds["lon"]
    # print("---------------- modis_level2_ds -------------- \n", modis_level2_ds)
    
    # --------------------- Subset the data within the map boundaries ----------------
    # subset_level2 = modis_level2_ds.where(
    #     (latitude >= map_boundaries[0]) & (latitude <= map_boundaries[1]) & 
    #     (longitude >= map_boundaries[2]) & (longitude <= map_boundaries[3]),
    #     drop=True
    # )

    if map_boundaries is not None:
        subset_level2 = modis_level2_ds.where(
            (latitude >= map_boundaries[0]) & (latitude <= map_boundaries[1]) & 
            (longitude >= map_boundaries[2]) & (longitude <= map_boundaries[3]),
            drop=True
        )
    else:
        subset_level2 = modis_level2_ds

    # print("---------------- subset_level2 -------------- \n", subset_level2)
    
    # Extract the trimmed latitude and longitude coordinates explicitly
    latitude_subset = subset_level2["lat"]
    longitude_subset = subset_level2["lon"]
    
    # Check the actual min/max of latitude and longitude for validation
    # print("\n Latitude subset min/max:", latitude_subset.min().item(), latitude_subset.max().item())
    # print("Longitude subset min/max:", longitude_subset.min().item(), longitude_subset.max().item())
    
    # -------------------------------- calculating N_top ----------------------------------------        
    tau_c_array = subset_level2.cod.values # Example cloud optical depth
    r_e_array = subset_level2.ref.values    # Example effective radius in micrometers
    
    Nd_array = calculate_nd_with_ref_cod(tau_c_array, r_e_array)
    # print(f" shape Nd_array: {Nd_array.shape}")
    # print(type(Nd_array))
    
    subset_level2["Nd"] = (['lat', 'lon'], Nd_array)  # this is an array
    subset_level2.Nd.attrs['units'] = "cm-3"
    subset_level2.Nd.attrs['standard_name'] = "Nd"
    subset_level2.Nd.attrs['long_name'] = "Cloud dropler number at top"
    print(" \n ---------------Calculated Cloud Droplet Number Concentration (cm^-3):")
    print(f" shape Nd_array: {Nd_array.shape}")
    # print("---------------- subset_level2 -------------- \n", subset_level2)
    subset_level2 = subset_level2[["Nd", "lwp", "cldph"]]
    
    return modis_level2_ds, subset_level2


def plot_lwp_nd_modis_retrievals(path, map_boundaries, result_folder, mask_cldph=[1], variable = ["Nd_max", "lwp"], min_value_threshold=5, max_value_threshold=2000, plot="True"):
    
    modis_level2_ds, subset_level2 = subset_level2_nd_lwp(path=path, 
                                                          map_boundaries=map_boundaries)

    nan_counts_per_var = subset_level2.isnull().sum()
    print(f"***----------------- nan data initial level2 : {nan_counts_per_var}-------------------------***")
        
    # ------------------------ obtaining the mask of only liquid water cldph ---------------
    values_to_keep = mask_cldph 
    # boolean mask
    mask = subset_level2['cldph'].isin(values_to_keep)
    masked_subset_ds = subset_level2.where(mask)
    
    # masked_subset_ds = subset_level2.where('ctt'>273)
    # masked_subset_ds = subset_level2.where(subset_level2['ctt'] > 273)

    # -------------------- converting to dataframe ---------------------------------------
    # Flatten the 2D arrays to 1D
    # Nd_flat = Nd_array.flatten()
    Nd_flat = masked_subset_ds.Nd.values.flatten()
    # LWP_flat = subset.lwp.values.flatten()
    LWP_flat = masked_subset_ds.lwp.values.flatten()

    count_equal_1 = (subset_level2['cldph'] == 1).sum().item()
    print(f"***----------------- values of cldph=1: {count_equal_1}-------------------------***")

    nan_counts_per_var = masked_subset_ds.isnull().sum()
    print(f"***----------------- nan masked subset with cldph: {nan_counts_per_var}-------------------------***")
        
    print(f"shape Nd_flat {Nd_flat.shape}, LWP_flat {LWP_flat.shape}")
    # Remove NaN values from the arrays
    Nd_clean = Nd_flat[~np.isnan(Nd_flat)]
    LWP_clean = LWP_flat[~np.isnan(LWP_flat)]
    print(f"shape Nd_clean nan {Nd_clean.shape}, LWP_clean nan {LWP_clean.shape}")

    
    df_modis_retrievals = pd.DataFrame({"Nd_max": Nd_clean, "lwp": LWP_clean})
    
    # -------------------- creating plot ---------------------------------------
    filtered_df_modis_retrievals = filter_dataframe_cloud(df_modis_retrievals,
                                                          min_value_threshold=min_value_threshold, 
                                                          max_value_threshold=max_value_threshold)

    if plot=="True":
        path_file = f"{result_folder}/Nd_lwp_modis_retrievals.png"
        # fig, ax = plot_joint_histogram(filtered_df_modis_retrievals, variable[0], variable[1], threshold="range3-870", figure_size=(6, 6))
        fig, ax = plot_joint_histogram(filtered_df_modis_retrievals, 
                                       filtered_df_modis_retrievals.columns[0], 
                                       filtered_df_modis_retrievals.columns[1], 
                                       threshold="range7-870", 
                                       figure_size=(6, 6))
        fig.savefig(path_file, bbox_inches="tight")
    
        # all 20 new 
        fig, ax = plot_joint_histogram_conditional(filtered_df_modis_retrievals, 
               filtered_df_modis_retrievals.columns[0], 
               filtered_df_modis_retrievals.columns[1], 
               threshold="range7-870", 
               figure_size=(6, 6))         
        path_file = f"{result_folder}/Nd_lwp_modis_retrievals_CP.png"
        fig.savefig(path_file, bbox_inches="tight")
        

    return modis_level2_ds, masked_subset_ds, subset_level2, filtered_df_modis_retrievals


def plot_reprojection_modis_level2(subset_level2, masked_subset_ds):    
    variables = ["Nd", "lwp"]
    
    for variable_name in variables:
        
        # ----------------- Germany all data -- not maked --------------
        data = subset_level2.variables[variable_name]  # Replace with your data variable name
        lat_all_masked_germany = subset_level2.variables['lat'][:]
        lon_all_masked_germany = subset_level2.variables['lon'][:]
        data_resampled, grid_lon, grid_lat = reprojection_data(data=data,
                                                               lat_swath = lat_all_masked_germany,
                                                               lon_swath = lon_all_masked_germany,
                                                               map_boundaries=map_boundaries)
        # nan_mask_ds = subset_level2.isnull()
        # nan_counts_per_var = subset_level2.isnull().sum()
        # print(nan_counts_per_var)
        
        plot_pcolormesh(longitude=grid_lon, 
                        latitude=grid_lat, 
                        variable=np.flipud(data_resampled), 
                        name_var=variable_name, 
                        level_name="2")
        
        # ---------------------- Germany masked only liquid water --------------------
        data = masked_subset_ds.variables[variable_name]  # Replace with your data variable name
        lat_all_masked_liquid_germany = masked_subset_ds.variables['lat'][:] 
        lon_all_masked_liquid_germany = masked_subset_ds.variables['lon'][:]
        data_resampled, grid_lon, grid_lat = reprojection_data(data=data,
                                                               lat_swath = lat_all_masked_liquid_germany,
                                                               lon_swath = lon_all_masked_liquid_germany,
                                                               map_boundaries=map_boundaries)
        
        plot_pcolormesh(longitude=grid_lon, 
                        latitude=grid_lat, 
                        variable=np.flipud(data_resampled), 
                        name_var=variable_name, 
                        level_name="2")

    # return data_resampled

def modis_level1_ref_rad(modis_level1, level2_path, map_boundaries, mask_cldph=[1]):
        
    # modis_level1 = xr.open_dataset(level1_path)
    # modis_level1_ds = modis_level1[["ref_1_2", "ref_3_7", "ref_8_19_26", "rad_20_36", "Latitude", "Longitude"]]
    # print("---------------- modis level 1 ------------------", modis_level1)
    
    latitude = modis_level1["lat"]
    longitude = modis_level1["lon"]
    
    
    # --------------- Subset the data within the map boundaries ---------------------
    # subset_level1 = modis_level1.where(
    #     (latitude >= map_boundaries[0]) & (latitude <= map_boundaries[1]) & 
    #     (longitude >= map_boundaries[2]) & (longitude <= map_boundaries[3]),
    #     drop=True
    # )

    if map_boundaries is not None:
        subset_level1 = modis_level1.where(
            (latitude >= map_boundaries[0]) & (latitude <= map_boundaries[1]) & 
            (longitude >= map_boundaries[2]) & (longitude <= map_boundaries[3]),
            drop=True
        )
    else:
        subset_level1 = modis_level1

    # print("---------------- subset modis level 1 ------------------", subset_level1)
    nan_counts_per_var = subset_level1.isnull().sum()
    print(f"***----------------- nan data initial level1 : {nan_counts_per_var}-------------------------***")
         
    # --------------- Reading the mask level 2  ---------------------
    # level2_path = "/work/bb1036/rttov_share/MODIS_data/Level1-2-test/processed_MYD06_L2_may_v2.nc"

    # mask_cldph=[1, 6]  # it has more points 
    # mask_cldph=[1]

    modis_level2_ds, subset_level2 = subset_level2_nd_lwp(path=level2_path, 
                                         map_boundaries=map_boundaries)
    
    # ------------------------ obtaining the mask of only liquid water cldph ---------------
    values_to_keep = mask_cldph  # [1, 6]
    # boolean mask
    mask_level2 = subset_level2['cldph'].isin(values_to_keep)
    count_equal_1 = (subset_level2['cldph'] == 1).sum().item()
    print(f"***----------------- values of cldph=1: {count_equal_1}-------------------------***")

    # mask_level2 = mask_level2.rename({'Latitude': 'lat', 'Longitude': 'lon'})
    # Check the result
    # print(mask_level2.dims)
    
    # Apply the mask to all variables in the dataset
    masked_subset_level1_ds = subset_level1.where(mask_level2)
    
    return masked_subset_level1_ds





def create_ref_rad_subset_modis(level1_path):
    # , index_channel_relate_clouds):
    """
    The function return data along 18 channels considering the channels with the index index_channel_relate_clouds
    Input:
        -- level1_ds (xarray): data con the different channels as the MODIS provided
    Output:
        --new_ds (xarray Datset): with the variable ref_rad_modis, which has only the channels assign in index_channel_relate_clouds 
    """
    name_channels = np.array(['1: 0.645 μm', '2: 0.856 μm', '3: 0.466 μm', '4: 0.554 μm',
   '5: 1.241 μm', '6: 1.628 μm', '7: 2.113 μm', '8: 0.412 μm',
   '9: 0.442 μm', '10: 0.487 μm', '11: 0.530 μm', '12: 0.547 μm',
   '13: 0.666 μm', '14: 0.678 μm', '15: 0.747 μm', '16: 0.867 μm',
   '17: 0.904 μm', '18: 0.936 μm', '19: 0.935 μm', '20: 3.777 μm',
   '21: 3.981 μm', '22: 3.971 μm', '23: 4.061 μm', '24: 4.448 μm',
   '25: 4.526 μm', '26: 1.382 μm', '27: 6.783 μm', '28: 7.344 μm',
   '29: 8.550 μm', '30: 9.720 μm', '31: 11.017 μm', '32: 12.036 μm',
   '33: 13.362 μm', '34: 13.683 μm', '35: 13.923 μm', '36: 14.213 μm'])
    index_channel_relate_clouds = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]
    channel_relate_clouds = name_channels[index_channel_relate_clouds]

    level1_ds = xr.open_dataset(level1_path)

    # Define dimensions and their sizes
    dims = ("chan", "lat", "lon")
    sizes = {"chan": 18, "lat": level1_ds.dims["lat"], "lon": level1_ds.dims["lon"]}
    correct_band_values = np.array(index_channel_relate_clouds) + 1
    
    # Create an empty DataArray filled with NaNs
    new_ds = xr.Dataset(
        # data_vars=np.full((sizes["chan"], sizes["lat"], sizes["lon"]), np.nan),
        data_vars={
            "ref_rad_modis": (("chan", "lat", "lon"), np.full((sizes["chan"], sizes["lat"], sizes["lon"]), np.nan)),  # Variable with dimensions
        },
    
        # dims=dims,
        coords={
            "chan": channel_relate_clouds, #correct_band_values,  # Assuming bands are numbered 1 to 36
            "lat": level1_ds["lat"],  # Use the latitude values from the existing dataset
            "lon": level1_ds["lon"],  # Use the longitude values from the existing dataset
        },
    )
        # name="empty_dataset"  # Optional: name your DataArray

    new_ds.ref_rad_modis[:2, :, :] = level1_ds.ref_1_2.sel(bands1=[1,2]).values
    print(level1_ds.ref_1_2.sel(bands1=[1,2]).bands1.values)
    new_ds.ref_rad_modis[2:7, :, :] = level1_ds.ref_3_7.sel(bands2=slice(3,7)).values
    print(level1_ds.ref_3_7.sel(bands2=slice(3,7)).bands2.values)
    new_ds.ref_rad_modis[7:, :, :] = level1_ds.rad_20_36.sel(bands4=[20, 21, 22, 23, 29, 31, 32, 33, 34, 35, 36]).values
    print(level1_ds.rad_20_36.sel(bands4=[20, 21, 22, 23, 29, 31, 32, 33, 34, 35, 36]).bands4.values)
    # print(f"--------------------------------- new_ds: {new_ds}")

    return new_ds


def create_ref_rad_subset_modis_17chan(level1_path):
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
    index_channel_relate_clouds = [0, 1, 2, 3, 4, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]
    channel_relate_clouds = name_channels[index_channel_relate_clouds]

    level1_ds = xr.open_dataset(level1_path)

    # Define dimensions and their sizes
    dims = ("chan", "lat", "lon")
    sizes = {"chan": 17, "lat": level1_ds.dims["lat"], "lon": level1_ds.dims["lon"]}
    correct_band_values = np.array(index_channel_relate_clouds) + 1
    
    # Create an empty DataArray filled with NaNs
    new_ds = xr.Dataset(
        # data_vars=np.full((sizes["chan"], sizes["lat"], sizes["lon"]), np.nan),
        data_vars={
            "ref_rad_modis": (("chan", "lat", "lon"), np.full((sizes["chan"], sizes["lat"], sizes["lon"]), np.nan)),  # Variable with dimensions
        },
    
        # dims=dims,
        coords={
            "chan": channel_relate_clouds, #correct_band_values,  # Assuming bands are numbered 1 to 36
            "lat": level1_ds["lat"],  # Use the latitude values from the existing dataset
            "lon": level1_ds["lon"],  # Use the longitude values from the existing dataset
        },
    )
        # name="empty_dataset"  # Optional: name your DataArray

    new_ds.ref_rad_modis[:2, :, :] = level1_ds.ref_1_2.sel(bands1=[1,2]).values
    print(level1_ds.ref_1_2.sel(bands1=[1,2]).bands1.values)
    new_ds.ref_rad_modis[2:6, :, :] = level1_ds.ref_3_7.sel(bands2=[3, 4, 5, 7]).values
    print(level1_ds.ref_3_7.sel(bands2=[3, 4, 5, 7]).bands2.values)
    new_ds.ref_rad_modis[6:, :, :] = level1_ds.rad_20_36.sel(bands4=[20, 21, 22, 23, 29, 31, 32, 33, 34, 35, 36]).values
    print(level1_ds.rad_20_36.sel(bands4=[20, 21, 22, 23, 29, 31, 32, 33, 34, 35, 36]).bands4.values)
    # print(f"--------------------------------- new_ds: {new_ds}")

    return new_ds


def obtain_lwp_nd_from_modis_level1_and_ML(level1_path, level2_path, path_model_file_nd, path_model_file_lwp, path_dataframes_scaler_lwp, path_dataframes_scaler_nd, fold_num, map_boundaries, result_folder, all_data, variables_names, mask_cldph=[1], min_value_threshold=5, max_value_threshold=2000, plot="True"):

    ref_rad_18_chan = create_ref_rad_subset_modis(level1_path=level1_path)
    print("=====================here===============")
    masked_subset_level1_ds = modis_level1_ref_rad(modis_level1=ref_rad_18_chan,
                                                   level2_path=level2_path,
                                                   map_boundaries=map_boundaries,
                                                   mask_cldph=mask_cldph)

    nan_counts_per_var = masked_subset_level1_ds.isnull().sum()
    print(f"***----------------- nan masked level 1 subset with level2: {nan_counts_per_var}-------------------------***")
        
    
    
    
    # , 
                                   # index_channel_relate_clouds=index_channel_relate_clouds)
    
    
    # ------------------------ Prediction with the ML model
    prediction_modis_xarray_nd, prediction_modis_xarray_lwp = get_prediction_nd_lwp(ds_channels=masked_subset_level1_ds["ref_rad_modis"],
                                                                                    # ref_rad_18_chan["ref_rad_modis"],
                                                                                    all_data=all_data, 
                                                                                    path_model_file_lwp=path_model_file_lwp, 
                                                                                    path_model_file_nd=path_model_file_nd, 
                                                                                    path_dataframes_scaler_lwp=path_dataframes_scaler_lwp,
                                                                                    path_dataframes_scaler_nd=path_dataframes_scaler_nd,
                                                                                    variables_names=variables_names, 
                                                                                    fold_num=fold_num)
    
    pred_lwp_nd_from_modis_ml_df = combine_lwp_nd_dataframe(prediction_xarray_nd=prediction_modis_xarray_nd,
                                                         prediction_xarray_lwp=prediction_modis_xarray_lwp)
    
    # ----------------------- filter and plot -------------------------------------------
    filtered_df_modis_ml = filter_dataframe_cloud(pred_lwp_nd_from_modis_ml_df,
                                                  min_value_threshold=min_value_threshold,
                                                  max_value_threshold=max_value_threshold)
    
    if plot=="True":
        path_file = f"{result_folder}/nd_lwp_ml_modis_level1.png"
        fig, ax = plot_joint_histogram(filtered_df_modis_ml, 
                                       filtered_df_modis_ml.columns[0], 
                                       filtered_df_modis_ml.columns[1], 
                                       threshold="range7-870", 
                                       figure_size=(6, 6))
        # fig, ax = plot_joint_histogram(filtered_df_modis_ml, variables_names[0], variables_names[1], threshold="range3-870", figure_size=(6, 6))
        fig.savefig(path_file, bbox_inches="tight")
    
        fig, ax = plot_joint_histogram_conditional(filtered_df_modis_ml, 
                                                   filtered_df_modis_ml.columns[0], 
                                                   filtered_df_modis_ml.columns[1], 
                                                   threshold="range7-870", 
                                                   figure_size=(6, 6))        
                                                
        path_file = f"{result_folder}/nd_lwp_ml_modis_level1_CP.png"
        fig.savefig(path_file, bbox_inches="tight")

    
    return filtered_df_modis_ml, ref_rad_18_chan



def modis_level1_level2(map_boundaries, level1_path, level2_path, all_data, result_folder, mask_cldph=[1], variable = ["Nd_max", "lwp"], min_value_threshold=5, max_value_threshold=2000, num_channels="18_chan", type_model="NN"):
    if num_channels=="18_chan":
        ref_rad_18_chan = create_ref_rad_subset_modis(level1_path=level1_path)
    elif num_channels=="17_chan":
        ref_rad_18_chan = create_ref_rad_subset_modis_17chan(level1_path=level1_path)

    print("---------------- modis level 1 ------------------")
    modis_level1 = ref_rad_18_chan
    latitude = modis_level1["lat"]
    longitude = modis_level1["lon"]

    print("---------------- subset modis level 1 ------------------")
    if map_boundaries is not None:
        subset_level1 = modis_level1.where(
            (latitude >= map_boundaries[0]) & (latitude <= map_boundaries[1]) & 
            (longitude >= map_boundaries[2]) & (longitude <= map_boundaries[3]),
            drop=True
        )
    else:
        subset_level1 = modis_level1

    nan_counts_per_var = subset_level1.isnull().sum()
    print(f"***----------------- nan data initial level1 : {nan_counts_per_var}-------------------------***")

    
    print("---------------- modis level 2 ------------------")
    modis_level2_ds, subset_level2 = subset_level2_nd_lwp(path=level2_path, 
                                         map_boundaries=map_boundaries)

    print("------------------ masking  ---------------------")

    values_to_keep = mask_cldph  # [1, 6]
    mask_level2 = subset_level2['cldph'].isin(values_to_keep)
    print(mask_level2.dims)


    masked_subset_level1_ds = subset_level1.where(mask_level2)
    masked_subset_level2_ds = subset_level2.where(mask_level2)

    df_spectral_all = ref_to_dataframe(ds_channels=masked_subset_level1_ds["ref_rad_modis"])

    # -------------------- converting to dataframe ---------------------------------------
    # Flatten the 2D arrays to 1D
    # Nd_flat = Nd_array.flatten()
    df_level2 = masked_subset_level2_ds.to_dataframe()

    df_combined = confirm_indices_and_join(df_level2, df_spectral_all)
    df_combined_cleaned = df_combined.dropna() # to delete the nan where there is not cloud and other nans

    df_combined_cleaned = df_combined_cleaned.rename(columns={"Nd": "Nd_max"})


    #=======check delete :2 no neede because doesnt include zero but clean that part to dont consider it 
    filtered_df_l1_l2_modis = filter_dataframe_cloud(df_combined_cleaned,
                                                          min_value_threshold=min_value_threshold, 
                                                          max_value_threshold=max_value_threshold)

    df_modis_retrievals = filtered_df_l1_l2_modis.iloc[:, :2]
    df_spectral_all = filtered_df_l1_l2_modis.iloc[:, 3:]


    cleaned_index = df_spectral_all.index

    
    channel_relate_clouds_lwp, channel_relate_clouds_nd = select_channels(all_data=all_data,
                                                                         num_channels=num_channels,
                                                                         type_model=type_model)

    df_spectral_lwp = df_spectral_all.loc[:, channel_relate_clouds_lwp]
    df_spectral_nd = df_spectral_all.loc[:, channel_relate_clouds_nd]

    df_spectral_lwp.columns=channel_relate_clouds_lwp
    df_spectral_nd.columns=channel_relate_clouds_nd
    
 
    return cleaned_index, df_spectral_all, df_spectral_lwp, df_spectral_nd, df_modis_retrievals, filtered_df_l1_l2_modis, channel_relate_clouds_lwp, channel_relate_clouds_nd, masked_subset_level1_ds, masked_subset_level2_ds


def obtain_df_data_level1_level2(map_boundaries, level1_path, level2_path, all_data_lwp, all_data_nd, result_folder, mask_cldph=[1], variable = ["Nd_max", "lwp"], min_value_threshold=5, max_value_threshold=2000, num_channels="18_chan", type_model="NN"):
# def data_level1_level2(map_boundaries, level1_path, level2_path, all_data, result_folder, mask_cldph=[1], variable = ["Nd_max", "lwp"], min_value_threshold=5, max_value_threshold=2000, num_channels="18_chan", type_model="NN"):

    if num_channels=="18_chan":
        ref_rad_18_chan = create_ref_rad_subset_modis(level1_path=level1_path)
    elif num_channels=="17_chan":
        ref_rad_18_chan = create_ref_rad_subset_modis_17chan(level1_path=level1_path)

    print("---------------- modis level 1 ------------------")
    modis_level1 = ref_rad_18_chan
    latitude = modis_level1["lat"]
    longitude = modis_level1["lon"]

    print("---------------- subset modis level 1 ------------------")
    if map_boundaries is not None:
        subset_level1 = modis_level1.where(
            (latitude >= map_boundaries[0]) & (latitude <= map_boundaries[1]) & 
            (longitude >= map_boundaries[2]) & (longitude <= map_boundaries[3]),
            drop=True
        )
    else:
        subset_level1 = modis_level1

    nan_counts_per_var = subset_level1.isnull().sum()
    print(f"***----------------- nan data initial level1 : {nan_counts_per_var}-------------------------***")

    
    print("---------------- modis level 2 ------------------")
    modis_level2_ds, subset_level2 = subset_level2_nd_lwp(path=level2_path, 
                                         map_boundaries=map_boundaries)

    # new_ds_level1 = projection_level1(masked_subset_level_ds=subset_level1,
    #                                   map_boundaries=map_boundaries)
    # new_ds_level2 = projection_level2(masked_subset_level_ds=subset_level2,
    #                                   map_boundaries=map_boundaries)
    
    new_ds_level1 = subset_level1
    new_ds_level2 = subset_level2

    masked_ds_level1 = new_ds_level1.where(~np.isnan(new_ds_level2.lwp))

    subset_level2 = new_ds_level2
    subset_level1 = masked_ds_level1
    
 
    print("------------------ masking  ---------------------")
    values_to_keep = mask_cldph  # [1, 6]
    mask_level2 = subset_level2['cldph'].isin(values_to_keep)
    print(mask_level2.dims)

    masked_subset_level1_ds = subset_level1.where(mask_level2)
    masked_subset_level2_ds = subset_level2.where(mask_level2)

    # =========================================
    # masked_subset_level1_ds.to_netcdf("/scratch/b/b381362/test_gridding/11_40_files/modis_level1_11_40.nc")
    # =========================================


    df_spectral_all = ref_to_dataframe(ds_channels=masked_subset_level1_ds["ref_rad_modis"])
    print(f"-----------------------------------------{df_spectral_all.columns}--------------------")
    # -------------------- converting to dataframe ---------------------------------------
    df_level2 = masked_subset_level2_ds.to_dataframe()
    print(f"-----------------------------------------{df_spectral_all.columns}--------------------")

    df_combined = confirm_indices_and_join(df_level2, df_spectral_all)
    print(f"-----------------------------------------{df_combined.columns}--------------------")
    all_index = df_combined.index

    df_combined_cleaned = df_combined.dropna() 

    df_combined_cleaned = df_combined_cleaned.rename(columns={"Nd": "Nd_max"})

    df_modis_retrievals = df_combined_cleaned.iloc[:, :2]
    df_spectral_all = df_combined_cleaned.iloc[:, 3:]
    cleaned_index = df_spectral_all.index

    # channel_relate_clouds_lwp, channel_relate_clouds_nd = select_channels(all_data=all_data,
    #                                                                      num_channels=num_channels,
    #                                                                      type_model=type_model)

    channel_relate_clouds_lwp, channel_relate_clouds_nd = select_channels_new(all_data_lwp=all_data_lwp,
                                                                          all_data_nd=all_data_nd,
                                                                         num_channels=num_channels,
                                                                         type_model=type_model)
 

    df_spectral_lwp = df_spectral_all.loc[:, channel_relate_clouds_lwp]
    df_spectral_nd = df_spectral_all.loc[:, channel_relate_clouds_nd]

    df_spectral_lwp.columns=channel_relate_clouds_lwp
    df_spectral_nd.columns=channel_relate_clouds_nd
    df_spectral_all.columns


    
    # return level1_path
    return ref_rad_18_chan, all_index, cleaned_index, df_spectral_all, df_spectral_lwp, df_spectral_nd, df_modis_retrievals, channel_relate_clouds_lwp, channel_relate_clouds_nd
    # , masked_subset_level1_ds, masked_subset_level2_ds
