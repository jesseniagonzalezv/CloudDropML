import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import numpy as np
import cartopy.mpl.ticker as cticker
import matplotlib.colors as colors
import pandas as pd
import os
import seaborn as sns
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from matplotlib import colors

from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm
import cartopy.feature as cfeature
import glob

from matplotlib.colors import LogNorm
def min_max_mean(variables):
    vmin = np.nanmin(variables)
    vmax = np.nanmax(variables)
    vmean = np.nanmean(variables)
    return vmin, vmax, vmean

def metric_evaluation_nrmse(gt, pred):
    # Ensure gt and pred are numpy arrays 
    gt = np.array(gt)
    pred = np.array(pred)

    indices_no_nan = ~np.isnan(gt) & ~np.isnan(pred)
    filtered_gt = gt[indices_no_nan]
    filtered_pred = pred[indices_no_nan]

    # vmin, vmax, vmean = min_max_mean([filtered_gt, filtered_pred])
    vmin_true, vmax_true, vmean_true = min_max_mean([filtered_gt])
    std_dev_true = np.std(filtered_gt)

    result_rmse = np.sqrt(mean_squared_error(filtered_gt, filtered_pred))
    result_score = r2_score(filtered_gt, filtered_pred)
    result_mae = mean_absolute_error(filtered_gt, filtered_pred)
    result_nrmse = (result_rmse / (vmax_true - vmin_true)) * 100

    percentage_errors = (filtered_gt - filtered_pred) / filtered_gt
    result_rmspe = np.sqrt(np.nanmean(percentage_errors ** 2)) * 100

    # Calcular MAPE
    result_mape = np.mean(np.abs((filtered_gt - filtered_pred) / filtered_gt)) * 100

    return result_rmse, result_score, result_mae, result_mape, result_nrmse, result_rmspe, vmin_true, vmax_true, vmean_true, std_dev_true
    


def histogram_plot2d(metrics, file1_retrievals, file2_ml_predictions, date, name_model, target_name, prediction_name, log_plots="False"):
    """
    ds_model1: ideally RF
    """
    fig, axli = plt.subplots(constrained_layout=True, nrows=2, ncols=1, figsize=(4, 16 / 4 * 2))

    if target_name.lower() == "lwp":
        variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif target_name.lower() == "nd_max":
        variable_target_units = r'$N_{\mathrm{d}}$ ($cm^{-3}$)'
    else:
        variable_target_units = target_name
        
    min_value_threshold = 5
    max_value_threshold = 1000
    
    # Read the datasets
    ds1 = xr.open_dataset(file1_retrievals)  # Expected values
    ds2 = xr.open_dataset(file2_ml_predictions)  # Predicted values
    
    # Align the datasets on the 'index' dimension (only matching indices are kept)
    ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join='inner')
    
    target_df = ds1_aligned.to_dataframe()
    prediction_df = ds2_aligned.to_dataframe()
    
    target_values = target_df[[target_name]]
    # .values.flatten()
    prediction_values = prediction_df[[prediction_name]]

    # print(target_values)
    # Mask: both values in range for both DataFrames
    joint_mask = (
        (target_values[target_name] >= min_value_threshold) &
        (target_values[target_name] <= max_value_threshold) &
        (prediction_values[target_name] >= min_value_threshold) &
        (prediction_values[target_name] <= max_value_threshold)
    )

    target_values = target_values[joint_mask]
    prediction_values = prediction_values[joint_mask]
    
    target_values = target_values[target_name]
    prediction_values = prediction_values[prediction_name]
    
    # .values.flatten()
    mask = ~np.isnan(target_values) & ~np.isnan(prediction_values)
    clean_target = target_values[mask]
    clean_prediction = prediction_values[mask]

    if date!="global":
        year = int(date[1:5])
        doy = int(date[5:])
        date_new_format = datetime(year, 1, 1) + timedelta(days=doy - 1)
        date_info = date_new_format.strftime('%Y-%m-%d')
    else:
        date_info="Mean along 10 days"

    
    if log_plots == "False":
        target_values = clean_target
        prediction_values = clean_prediction 
    if log_plots == "True":
        target_values = np.log(clean_target)
        prediction_values = np.log(clean_prediction) 

    vmin, vmax, vmean = min_max_mean([target_values,
                                      prediction_values])

    result_rmse, result_score, result_mae, result_mape, result_nrmse, result_rmspe, vmin_true, vmax_true, vmean_true, std_dev_true = metric_evaluation_nrmse(
        target_values,
        prediction_values)
    values_metrics_list = []

    for metric_name in metrics:
        if metric_name == 'R2':
            values_metrics = rf"$R^2$: {result_score:.2f}"
            values_metrics_list.append(values_metrics)
        elif metric_name == 'nRMSE':
            values_metrics = rf"nRMSE (%): {result_nRMSE:.2f}"
            values_metrics_list.append(values_metrics)
        elif metric_name == "RMSPE":
            values_metrics = rf"RMSPE (%): {result_rmspe:.2f}"
            values_metrics_list.append(values_metrics)
        elif metric_name == "MAPE":
            values_metrics = rf"MAPE (%): {result_mape:.2f}"
            values_metrics_list.append(values_metrics)
        elif metric_name == "RMSE":
            values_metrics = rf"RMSE: {result_rmse:.2f}"
            values_metrics_list.append(values_metrics)
        elif metric_name == "MAE":
            values_metrics = rf"MAE: {result_mae:.2f}"
            values_metrics_list.append(values_metrics)

    # Join the list into a single string separated by commas
    metrics_string = ", ".join(values_metrics_list)

    titles_font_size = 17
    # subtitles_font_size = 16.5
    subtitles_font_size = 16
    labels_font_size = 16
    # axes_font_size = 13
    axes_font_size = 14

    ax_idx = 0
    print("mean------------------------", vmean)
    hist_values, _ = np.histogram(target_values, bins=30, density=True)
    max_hist_density = max(hist_values)
    y_max = 2*max_hist_density  # lwp
    sns.kdeplot(target_values, ax=axli[ax_idx], color='blue', label="MODIS retrievals", linewidth=2, bw_adjust=0.3)   # NaNs are ignored automatically
    sns.kdeplot(prediction_values, ax=axli[ax_idx], color='brown', label="MODIS spectral + ML", linewidth=2,
                bw_adjust=0.3)
    axli[ax_idx].set_xlim(0, None)  # Set the lower x-limit to 0
    axli[ax_idx].set_ylim(0, y_max)
    # =========== end test limite superior =============    
    axli[ax_idx].set_ylabel('Density', fontsize=labels_font_size)
    axli[ax_idx].set_xlabel('Value', fontsize=labels_font_size)
    axli[ax_idx].legend(fontsize=axes_font_size-2)
    axli[ax_idx].tick_params(labelsize=axes_font_size)
    # axli[ax_idx].set_title(f"{variable_target_units}", fontsize=titles_font_size)
    axli[ax_idx].set_title(f"{date_info}\n{variable_target_units}", fontsize=titles_font_size)


    ax_idx = 1
    axli[ax_idx].set_title(metrics_string, fontsize=subtitles_font_size)
    # hb1 = axli[ax_idx].hexbin(target_values, prediction_values,
                              # gridsize=300, cmap='viridis', mincnt=10, norm=colors.LogNorm())

    hb1 = axli[ax_idx].hexbin(target_values, prediction_values,
                              gridsize=500, cmap='viridis', mincnt=20, norm=colors.LogNorm())


    axli[ax_idx].set_xlabel('MODIS retrievals', fontsize=labels_font_size)
    axli[ax_idx].set_ylabel('MODIS spectral + ML', fontsize=labels_font_size)
    axli[ax_idx].tick_params(labelsize=axes_font_size)
    axli[ax_idx].set_xlim(vmin, vmax)
    axli[ax_idx].set_ylim(vmin, vmax)
    axli[ax_idx].set_aspect('equal')
    ticks = np.linspace(vmin, vmax, 4)
    axli[ax_idx].set_xticks(ticks)
    axli[ax_idx].set_yticks(ticks)

    line = axli[ax_idx].plot([vmin_true, vmax_true], [vmin_true, vmax_true], color='red')  # Perfect prediction line

    cb = plt.colorbar(hb1, ax=axli[ax_idx], location='bottom', )
    cb.ax.tick_params(labelsize=axes_font_size)

    # plt.title(f"{variable_target_units} - {title}")
    path_file = f"{output_folder}/{date}_{variable_target}_hist_2dplot.png"
    plt.savefig(path_file, bbox_inches='tight')
    plt.close()




def plot_global_from_df(files, title, date, variable_target, output_folder):       
    vmin=1e0
    vmax=1e3
    if variable_target == "lwp" or variable_target == 'LWP':
        variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif variable_target == "Nd_max" or variable_target == r'$N_{d,\ max}$':
        variable_target_units = r' $N_{\mathrm{d}}$ ($cm^{-3}$)'

    norm = colors.LogNorm(vmin=vmin, vmax=vmax)            
    
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    for file in files:
        # ds = xr.open_dataset(file)
        df_xarray = xr.open_dataset(file)
        df = df_xarray.to_dataframe()
        print(df)
        # ===================== filter -60 to 60 ==============
        df = df[(df['lat'] >= -60) & (df['lat'] <= 60)]  # Filter by latitude
        print(df)        
        # ===================== end filter -60 to 60 ==============
        ds = df.set_index(['lat', 'lon'], inplace=False).to_xarray()
        
        var = ds[variable_target]  # Replace with actual variable
        lat = ds['lat']
        lon = ds['lon']
        
        plt.pcolormesh(lon, lat, var, shading='auto', transform=ccrs.PlateCarree(), norm=norm, cmap='jet')
        # plt.pcolormesh(lon, lat, var, shading='auto', transform=ccrs.PlateCarree(), norm=norm, cmap='Spectral_r')


    plt.colorbar(location='right', fraction=0.006, pad=0.08)
    plt.title(f"{variable_target_units} - {title}")
    path_file = f"{output_folder}/{date}_{variable_target}_{title}_global_last_point.png"
    plt.savefig(path_file, bbox_inches='tight')
    plt.close()
    # plt.show()



# def no_sirve_combine_average(files, title, variable_target, output_folder):
#     vmin = 1e0
#     vmax = 1e3
#     norm = colors.LogNorm(vmin=vmin, vmax=vmax)

#     # Set variable units for plot title
#     if variable_target.lower() == "lwp":
#         variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
#     elif variable_target.lower() == "nd_max":
#         variable_target_units = r'$N_{\mathrm{d}}$ ($cm^{-3}$)'
#     else:
#         variable_target_units = variable_target

#     # Load multiple NetCDFs as a virtual stack along 'file' dimension
#     ds = xr.open_mfdataset(
#         files,
#         combine='nested',
#         concat_dim='file',
#         parallel=True
#     )

#     # Compute mean across files, ignoring NaNs
#     avg = ds[variable_target].mean(dim='file', skipna=True)

#     # Save averaged result to NetCDF
#     output_nc = f"{output_folder}/averaged_{variable_target}_{title}.nc"
#     avg.to_dataset(name=variable_target).to_netcdf(output_nc, mode='w')

#     # Plot
#     plt.figure(figsize=(12, 6))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE)
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
#     gl.xformatter = cticker.LongitudeFormatter()
#     gl.yformatter = cticker.LatitudeFormatter()

#     pc = ax.pcolormesh(avg['lon'], avg['lat'], avg.values, shading='auto', norm=norm, cmap='jet')
#     plt.colorbar(pc, label=variable_target_units, location='right', fraction=0.006, pad=0.08)
#     plt.title(f"{variable_target_units} - {title} (Mean)")

#     # output_plot = os.path.join(output_folder, f"{variable_target}_{title}_global_mean.png")
#     output_plot = f"{output_folder}/{variable_target}_{title}_global_mean.png"
#     plt.savefig(output_plot, bbox_inches='tight')
#     plt.close()


# def plot_ds(ds_out, variable_target, title, output_folder): 
#     # Plotting
#     plt.figure(figsize=(12, 6))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE)
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
#     gl.xformatter = cticker.LongitudeFormatter()
#     gl.yformatter = cticker.LatitudeFormatter()

#     pc = ax.pcolormesh(lon_grid, lat_grid, value_grid, shading='auto', norm=norm, cmap='jet')
#     plt.colorbar(pc, label=variable_target_units, location='right', fraction=0.006, pad=0.08)

#     path_file = os.path.join(output_folder, f"{variable_target}_{title}_global_mean.png")
#     plt.savefig(path_file, bbox_inches='tight')
#     plt.close()


def combine_average(files, title, variable_target, date, output_folder):
    vmin = 1e0
    vmax = 1e3
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    if variable_target.lower() == "lwp":
        variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif variable_target.lower() == "nd_max":
        variable_target_units = r'$N_{\mathrm{d}}$ ($cm^{-3}$)'
    else:
        variable_target_units = variable_target

    records = []

    for f in files:
        ds = xr.open_dataset(f)
        if variable_target not in ds:
            continue  # skip if var not in this file

        # var = ds[variable_target].squeeze()
        # lat = ds['lat'].values.flatten()
        # lon = ds['lon'].values.flatten()
        # val = var.values.flatten()

        # # Skip all-NaN cases
        # valid = ~pd.isna(val)
        # df = pd.DataFrame({
        #     'lat': lat[valid],
        #     'lon': lon[valid],
        #     'value': val[valid]
        # })

        df = ds[[variable_target, "lat", "lon"]].to_dataframe()
        df = df[(df['lat'] >= -60) & (df['lat'] <= 60)]  # Filter by latitude
        
        df2 = df.set_index(['lat', 'lon'], inplace=False)
        df3 = df2.dropna(inplace=False)
        df3 = df3.reset_index()

        records.append(df3)

    combined = pd.concat(records, ignore_index=True)

    # Round to reduce float mismatches
    combined['lat'] = combined['lat'].round(3)
    combined['lon'] = combined['lon'].round(3)

    # Group by location and average
    # averaged = combined.groupby(['lat', 'lon'], as_index=False).mean()
    # averaged = combined.groupby(['lat', 'lon'], as_index=False).mean(numeric_only=True)
    averaged = combined.groupby(['lat', 'lon'], as_index=False).mean(numeric_only=True)
    averaged_cleaned = averaged.dropna(inplace=False)

    ds_out = xr.Dataset.from_dataframe(averaged_cleaned)

    # # Pivot into grid for plotting
    # pivot = averaged.pivot(index='lat', columns='lon', values='value')
    # lat_grid = pivot.index.values
    # lon_grid = pivot.columns.values
    # value_grid = pivot.values

    # # Save to NetCDF
    # ds_out = xr.Dataset(
    #     {variable_target: (['lat', 'lon'], value_grid)},
    #     coords={'lat': lat_grid, 'lon': lon_grid}
    # )
    output_nc = os.path.join(output_folder, f'averaged_{date}_{variable_target}_{title}.nc')
    # ds_out.to_netcdf(output_nc, mode='w')
    
    if os.path.exists(output_nc):
        os.remove(output_nc)
    ds_out.to_netcdf(output_nc, mode='w')
    ds_out.close()


# def nor_sirvecombine_average(files, title, variable_target, output_folder):
#     vmin = 1e0
#     vmax = 1e3
#     norm = colors.LogNorm(vmin=vmin, vmax=vmax)

#     # Optional: readable label
#     if variable_target.lower() == "lwp":
#         variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
#     elif variable_target.lower() == "nd_max":
#         variable_target_units = r'$N_{\mathrm{d}}$ ($cm^{-3}$)'
#     else:
#         variable_target_units = variable_target

#     records = []

#     for f in files:
#         ds = xr.open_dataset(f)

#         if variable_target not in ds:
#             continue  # skip file if variable is missing

#         var = ds[variable_target].squeeze()

#         # Get 2D lat/lon grid
#         lat1d = ds['lat'].values
#         lon1d = ds['lon'].values
#         lon2d, lat2d = np.meshgrid(lon1d, lat1d)

#         # Flatten all values
#         val = var.values.flatten()
#         lat_flat = lat2d.flatten()
#         lon_flat = lon2d.flatten()

#         # Keep only valid (non-NaN) entries
#         valid = ~np.isnan(val)
#         if not np.any(valid):
#             continue

#         df = pd.DataFrame({
#             'lat': lat_flat[valid],
#             'lon': lon_flat[valid],
#             'value': val[valid]
#         })
#         records.append(df)

#     if not records:
#         raise ValueError("No valid data found in input files.")

#     # Combine all entries
#     combined = pd.concat(records, ignore_index=True)

#     # Round to reduce floating-point mismatches in lat/lon
#     combined['lat'] = combined['lat'].round(3)
#     combined['lon'] = combined['lon'].round(3)

#     # Average by grid point
#     averaged = combined.groupby(['lat', 'lon'], as_index=False).mean(numeric_only=True)

#     # Pivot to grid for output
#     pivot = averaged.pivot(index='lat', columns='lon', values='value')
#     lat_grid = pivot.index.values
#     lon_grid = pivot.columns.values
#     value_grid = pivot.values

#     # Save to NetCDF
#     ds_out = xr.Dataset(
#         {variable_target: (['lat', 'lon'], value_grid)},
#         coords={'lat': lat_grid, 'lon': lon_grid}
#     )

#     output_nc = os.path.join(output_folder, f'averaged_{variable_target}_{title}.nc')
#     ds_out.to_netcdf(output_nc, 'w')
#     ds_out.close()


# def old_combine_average(files, title, variable_target, output_folder):
#     vmin=1e0
#     vmax=1e3
#     norm = colors.LogNorm(vmin=vmin, vmax=vmax)            
#     data_points = []
    
#     if variable_target == "lwp" or variable_target == 'LWP':
#         variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
#     elif variable_target == "Nd_max" or variable_target == r'$N_{d,\ max}$':
#         variable_target_units = r' $N_{\mathrm{d}}$ ($cm^{-3}$)'
    
#     for f in files[:1]:
#     # for f in files:
#         ds = xr.open_dataset(f)
#         var = ds[variable_target]  # e.g., temperature
#         lat = ds['lat']
#         lon = ds['lon']
    
#         # Handle 1D or flattened data (e.g., 'point' dim)
#         var = var.squeeze()
#         df = pd.DataFrame({
#             'lat': lat.values.flatten(),
#             'lon': lon.values.flatten(),
#             'value': var.values.flatten()
#         })
    
#         data_points.append(df)
    
#     # Combine all into one DataFrame
#     combined = pd.concat(data_points, ignore_index=True)
    
#     # Round coordinates to reduce floating-point mismatch (optional but helps)
#     combined['lat'] = combined['lat'].round(3)
#     combined['lon'] = combined['lon'].round(3)
    
#     # Group by lat/lon and compute mean
#     averaged = combined.groupby(['lat', 'lon'], as_index=False).mean()
    
#     # # Convert back to grid for plotting (optional)
#     # pivot = averaged.pivot(index='lat', columns='lon', values='value')
#     # lat_grid = pivot.index.values
#     # lon_grid = pivot.columns.values
#     # value_grid = pivot.values


#     # Create xarray Dataset from the 2D gridded data
#     ds_out = xr.Dataset(
#         {
#             f'{variable_target}': (['lat', 'lon'], value_grid)
#         },
#         coords={
#             'lat': lat_grid,
#             'lon': lon_grid
#         }
#     )
    
#     ds_out.to_netcdf(f'{output_folder}/averaged_{variable_target}_{title}.nc', mode='w')

#     plt.figure(figsize=(12, 6))
#     ax = plt.axes(projection=ccrs.PlateCarree())

#     ax.add_feature(cfeature.COASTLINE)
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)    
#     gl.xformatter = cticker.LongitudeFormatter()
#     gl.yformatter = cticker.LatitudeFormatter()

#     pc = ax.pcolormesh(lon_grid, lat_grid, value_grid, shading='auto', norm=norm, cmap='jet')  #nearest
#     # plt.colorbar(pc, label='Averaged Value')
#     plt.colorbar(pc, location='right', fraction=0.006, pad=0.08)

#     plt.title(f"{variable_target_units} - {title} (Mean)")
#     path_file = f"{output_folder}/{variable_target}_{title}_global_mean.png"
#     plt.savefig(path_file, bbox_inches='tight')
#     plt.close()    

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature





import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import ticker

def plot_comparison(file1_retrievals, file2_ml_predictions, date, target_variable='lwp', output_folder='', sample_percentage=0.01):
    # vmin = 1e0
    vmin = 5
    vmax = 1e3
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)            
    # cmap = plt.get_cmap('Spectral_r', 75)  # Discretize to 65 colors
    log_ticks = [10, 20, 50, 100, 200, 500, 1000]

    
    # Optional: readable label
    if variable_target.lower() == "lwp":
        variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif variable_target.lower() == "nd_max":
        variable_target_units = r'$N_{\mathrm{d}}$ ($cm^{-3}$)'
    else:
        variable_target_units = variable_target


    # Read the datasets
    ds1 = xr.open_dataset(file1_retrievals)  # Expected values
    ds2 = xr.open_dataset(file2_ml_predictions)  # Predicted values
    
    # Align the datasets on the 'index' dimension (only matching indices are kept)
    ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join='inner')
    
    # Convert to DataFrames for easier handling
    df1 = ds1_aligned.to_dataframe()
    # .reset_index()
    df2 = ds2_aligned.to_dataframe()
    # .reset_index()
    
    # Ensure that the DataFrames have the same shape
    if df1.shape != df2.shape:
        raise ValueError("The datasets have different shapes after alignment. Check the alignment process.")
    
    # Sample the data (default is 1% of the dataset size)
    sample_size = int(len(df1) * sample_percentage)
    df1_sampled = df1.sample(n=sample_size, random_state=42)
    df2_sampled = df2.sample(n=sample_size, random_state=42)

    if date!="global":
        year = int(date[1:5])
        doy = int(date[5:])
        date_new_format = datetime(year, 1, 1) + timedelta(days=doy - 1)
        date_info = date_new_format.strftime('%Y-%m-%d')

    else:
        date_info="Mean along 10 days"
    
    title = ["MODIS retrievals",  "MODIS spectral + ML"]
    
    # Create the plot
    for i, df_sampled in enumerate([df1_sampled, df2_sampled]):
        plt.figure(figsize=(8, 5.5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)  # Add coastlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()
    
        # Scatter plot comparing LWP values from both datasets
        # scatter = ax.scatter(df_sampled['lon'], df_sampled['lat'], c=df_sampled[target_variable], norm=norm, cmap='jet', s=1, label=title[i])
        scatter = ax.scatter(df_sampled['lon'], df_sampled['lat'], c=df_sampled[target_variable], norm=norm, cmap='Spectral_r', s=1, label=title[i])


        # Add colorbar for the scatter plot
        # plt.colorbar(scatter, ax=ax, label=f'{target_variable.upper()}')
        # plt.colorbar(scatter, location='right', fraction=0.017, pad=0.06)

        cbar = plt.colorbar(scatter, location='right', fraction=0.017, pad=0.08)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([str(t) for t in log_ticks])

        # Set the title and save the plot
        plt.title(f'{variable_target_units} - {title[i]} - {date_info}')
        path_file = f"{output_folder}/{date}_{target_variable}_{title[i]}_global_mean_scatter.png"
        plt.savefig(path_file, bbox_inches='tight')
        plt.close()

    # ------------------------- diferences
    threshold = 30  
    
    mask = df1_sampled[target_variable] > threshold
    
    percentage_error = ((df2_sampled[mask] - df1_sampled[mask]) * 100) / df1_sampled[mask]
    
    lon_masked = df1_sampled['lon'][mask]
    
    
    # percentage_error = ((df2_sampled - df1_sampled) * 100) / df1_sampled
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)

    plt.figure(figsize=(8, 5.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)  # Add coastlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()


    hb = ax.scatter(lon_masked, lat_masked, C=percentage_error[target_variable], norm=norm, cmap='coolwarm_r', s=1)

    cbar = plt.colorbar(hb, location='right', fraction=0.017, pad=0.08)
    formatter = FuncFormatter(lambda x, pos: f'{x:.0f}%')
    cbar.ax.yaxis.set_major_formatter(formatter)
    plt.title(f"Percentage error - {date_info}")

    path_file = f"{output_folder}/{date}_{target_variable}_percent_error_global_mean_scatter.png"
    plt.savefig(path_file, bbox_inches='tight')
    plt.close()



def plot_comparison_hexbin(file1_retrievals, file2_ml_predictions, date, target_variable='lwp', output_folder='', sample_percentage=0.01):
    # vmin = 1e0
    vmin = 5
    vmax = 1e3
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)            
    log_ticks = [10, 20, 50, 100, 200, 500, 1000]

    # cmap = plt.get_cmap('Spectral_r', 75)  # Discretize to 65 colors

    if variable_target.lower() == "lwp":
        variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif variable_target.lower() == "nd_max":
        variable_target_units = r'$N_{\mathrm{d}}$ ($cm^{-3}$)'
    else:
        variable_target_units = variable_target


    # Read the datasets
    ds1 = xr.open_dataset(file1_retrievals)  # Expected values
    ds2 = xr.open_dataset(file2_ml_predictions)  # Predicted values
    
    # Align the datasets on the 'index' dimension (only matching indices are kept)
    ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join='inner')
    
    # Convert to DataFrames for easier handling
    df1 = ds1_aligned.to_dataframe()
    # .reset_index()
    df2 = ds2_aligned.to_dataframe()
    # .reset_index()
    
    # Ensure that the DataFrames have the same shape
    if df1.shape != df2.shape:
        raise ValueError("The datasets have different shapes after alignment. Check the alignment process.")
    
    # Sample the data (default is 1% of the dataset size)
    sample_size = int(len(df1) * sample_percentage)
    df1_sampled = df1.sample(n=sample_size, random_state=42)
    df2_sampled = df2.sample(n=sample_size, random_state=42)

    min_value_threshold = 5
    max_value_threshold = 1000
    
    # Mask: both values in range for both DataFrames
    joint_mask = (
        (df1_sampled[variable_target] >= min_value_threshold) &
        (df1_sampled[variable_target] <= max_value_threshold) &
        (df2_sampled[variable_target] >= min_value_threshold) &
        (df2_sampled[variable_target] <= max_value_threshold)
    )
    
    df1_sampled = df1_sampled[joint_mask]
    df2_sampled = df2_sampled[joint_mask]

    if date!="global":
        year = int(date[1:5])
        doy = int(date[5:])
        date_new_format = datetime(year, 1, 1) + timedelta(days=doy - 1)
        date_info = date_new_format.strftime('%Y-%m-%d')

    else:
        date_info="Mean along 10 days"
    
    title = ["MODIS retrievals",  "MODIS spectral + ML"]
    
    for i, df_sampled in enumerate([df1_sampled, df2_sampled]):
        plt.figure(figsize=(8, 5.5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)  # Add coastlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()
    
        # Scatter plot comparing LWP values from both datasets
        # hb = ax.hexbin(df_sampled['lon'], df_sampled['lat'], C=df_sampled[target_variable], gridsize=1000, norm=norm, cmap='jet', mincnt=1)
        hb = ax.hexbin(df_sampled['lon'], df_sampled['lat'], C=df_sampled[target_variable], gridsize=1000, norm=norm, cmap='Spectral_r', mincnt=1)

        # Add colorbar for the scatter plot
        # plt.colorbar(hb, ax=ax)
        # plt.colorbar(hb, location='right', fraction=0.017, pad=0.08)

        cbar = plt.colorbar(hb, location='right', fraction=0.017, pad=0.08)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([str(t) for t in log_ticks])

        # , label=f'{target_variable.upper()}')


        plt.title(f'{variable_target_units} - {title[i]} - {date_info}')
            
        path_file = f"{output_folder}/{date}_{target_variable}_{title[i]}_global_mean_hexbin.png"
        plt.savefig(path_file, bbox_inches='tight')
        plt.close()

    # ------------------------- diferences
    threshold = 30  
    
    mask = df1_sampled[target_variable] > threshold
    
    percentage_error = ((df2_sampled[mask] - df1_sampled[mask]) * 100) / df1_sampled[mask]
    
    lon_masked = df1_sampled['lon'][mask]
    lat_masked = df1_sampled['lat'][mask]
    
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    
    plt.figure(figsize=(8, 5.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    
    # coordenadas ya filtradas
    hb = ax.hexbin(
        lon_masked, lat_masked, C=percentage_error[target_variable],
        gridsize=2000, norm=norm, cmap='coolwarm_r', mincnt=1
    )
    
    cbar = plt.colorbar(hb, location='right', fraction=0.017, pad=0.08)
    formatter = FuncFormatter(lambda x, pos: f'{x:.0f}%')
    cbar.ax.yaxis.set_major_formatter(formatter)
    plt.title(f"Percentage error - {date_info}")


    # percentage_error = ((df2_sampled - df1_sampled) * 100) / df1_sampled
    # norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    # plt.figure(figsize=(8, 5.5))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.add_feature(cfeature.COASTLINE)  # Add coastlines
    # gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    # gl.xformatter = cticker.LongitudeFormatter()
    # gl.yformatter = cticker.LatitudeFormatter()
                   
    # hb = ax.hexbin(df2_sampled['lon'], df2_sampled['lat'], C=percentage_error[target_variable], gridsize=1000, norm=norm, cmap='RdBu', mincnt=1)
    # cbar = plt.colorbar(hb, location='right', fraction=0.017, pad=0.08)
    # formatter = FuncFormatter(lambda x, pos: f'{x:.0f}%')
    # cbar.ax.yaxis.set_major_formatter(formatter)
    # plt.title("Percentage error")

    path_file = f"{output_folder}/{date}_{target_variable}_percent_error_global_mean_hexbin.png"
    plt.savefig(path_file, bbox_inches='tight')
    plt.close()



def histogram_method_comparison_modis(df_modis_retrievals, df_modis_ml, result_folder, date, clip_min=0, clip_maximun=2000):
    clip_range = (clip_min, clip_maximun)

    custom_palette = [
                            "#006400",  # verde
                            "#A55194",  # dusty purple
                        ]
    sns.set_palette(custom_palette)

    for variable_target in ["Nd_max", "LWP"]:
        plt.figure(figsize=(7, 5)) 
        if variable_target=="Nd_max":
            # KDE plot for N_d
            sns.kdeplot(data=df_modis_retrievals['Nd_max'], fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
            sns.kdeplot(data=df_modis_ml['Nd_max'], fill=False, clip=clip_range, common_norm=False, label='MODIS spectral+ ML')
            plt.xlabel(r'$N_{\mathrm{d}}$ ($cm^{-3}$)', fontsize=14)
        # plt.title('KDE of Nd across different data sources', fontsize=15)
    
        elif variable_target=="LWP":
            # KDE plot for LWP
            sns.kdeplot(data=df_modis_retrievals['lwp'], fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
            sns.kdeplot(data=df_modis_ml['lwp'], fill=False, clip=clip_range, common_norm=False, label='MODIS spectral + ML')
            plt.xlabel(r'$L$ ($\mathrm{g\,m}^{-2}$)', fontsize=14)
    
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
 
        # Adjust layout to prevent overlap
        plt.tight_layout()
        path_file = f"{result_folder}/{date}_{variable_target}_comparison_modis_kde_global.png"

        plt.savefig(path_file, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare methods')
    arg = parser.add_argument
    arg('--type_aggregation', type=str, default="/work/", help='last_point, mean')
    arg('--output_nc', type=str, default="/work/", help='"Folder where there are the nc files.')
    arg('--output_folder', type=str, default="/work/global_data", help='"output folder to put the plots.')

    args = parser.parse_args()
    
    type_aggregation = args.type_aggregation
    output_nc = args.output_nc
    output_folder = args.output_folder
    
    # # type_aggregation = "last_point"
    # # type_aggregation = "mean"
    # type_aggregation = "hist_mean"
    # output_folder='/scratch/b/b381362/copy_hist/compare_mean_normal'

    # output_nc='/scratch/b/b381362/copy_hist/'
    # # output_folder='/scratch/b/b381362/copy_hist/'

        

    if type_aggregation=="mean_all_dates":
        # --------------------- create the nc average file -------------------------------
        for variable_target in ["lwp", "Nd_max"]:
            #================== all files
            file_list1 = sorted(glob.glob(f"{output_folder}/averaged_A*_{variable_target}_MODIS spectral + ML*.nc"))
            files_level1 = file_list1
            # files_level2 = file_list[:2]
            
            file_list2 = sorted(glob.glob(f"{output_folder}/averaged_A*_{variable_target}_MODIS retrievals.nc"))
            files_level2 = file_list2
            # files_level1 = file_list[:2]
            print(len(files_level1), len(files_level2))
            
            for f1, f2 in zip(files_level1, files_level2):
                print(f"=: {os.path.basename(f1)} \n   {os.path.basename(f2)}")    
                
            combine_average(files=files_level1, 
                        title="MODIS spectral + ML", 
                        variable_target=variable_target,
                        date="global",
                        output_folder=output_folder)       
            
            combine_average(files=files_level2, 
                        title="MODIS retrievals", 
                        variable_target=variable_target,
                        date="global",
                        output_folder=output_folder)    

    elif type_aggregation=="plot_averaged_global_hexbin":
        for variable_target in ["lwp", "Nd_max"]:
            file_level1 = f"{output_folder}/averaged_global_{variable_target}_MODIS spectral + ML.nc"
            file_level2 = f"{output_folder}/averaged_global_{variable_target}_MODIS retrievals.nc"

            plot_comparison_hexbin(file1_retrievals=file_level2, 
                                    file2_ml_predictions=file_level1, 
                                    target_variable=variable_target, 
                                    sample_percentage=1, #0.08, #0.01,
                                    date="global",
                                    output_folder=output_folder)

    elif type_aggregation=="plot_averaged_global_scatter":
        for variable_target in ["lwp", "Nd_max"]:
            file_level1 = f"{output_folder}/averaged_global_{variable_target}_MODIS spectral + ML.nc"
            file_level2 = f"{output_folder}/averaged_global_{variable_target}_MODIS retrievals.nc"

            plot_comparison(file1_retrievals=file_level2, 
                                    file2_ml_predictions=file_level1, 
                                    target_variable=variable_target, 
                                    sample_percentage=1, #0.08, #0.01,
                                    date="global",
                                    output_folder=output_folder)

    elif type_aggregation=="plot_averaged_global_hist":
        for variable_target in ["lwp", "Nd_max"]:
            file_level1 = f"{output_folder}/averaged_global_{variable_target}_MODIS spectral + ML.nc"
            file_level2 = f"{output_folder}/averaged_global_{variable_target}_MODIS retrievals.nc"

            histogram_plot2d(file1_retrievals=file_level2,
                             file2_ml_predictions=file_level1,
                             metrics=['R2', 'MAE'],
                             name_model="NN",
                             target_name=f'{variable_target}',
                             prediction_name=f'{variable_target}',
                             date="global",
                             log_plots="False")
        
    # ================================== by day
    # -------------------------level 1 ------------------
    # Adjust paths and patterns as needed
    file_pattern_level1 = os.path.join(output_nc, "df_level1_ml_predictions_*.nc")
    file_list_level1 = sorted(glob.glob(file_pattern_level1))
   
    # Group files by date (assuming date format is AYYYYDDD in the filename)
    files_by_date_level1 = defaultdict(list)
    
    for f in file_list_level1:
        basename = os.path.basename(f)
        # Extract date pattern, e.g., A2010151
        date_part = basename.split("_")[4]  
        files_by_date_level1[date_part].append(f)
    
    # Step 3: Build `date_file_all` and `dates`
    date_file_all_level1 = list(files_by_date_level1.values())
    dates_all_level1 = list(files_by_date_level1.keys())
    
    # -------------------------level 2 ------------------
    # Adjust paths and patterns as needed
    file_pattern_level2 = os.path.join(output_nc, "df_level2_modis_*.nc")
    file_list_level2 = sorted(glob.glob(file_pattern_level2))
   
    # Group files by date (assuming date format is AYYYYDDD in the filename)
    files_by_date_level2 = defaultdict(list)
    
    for f in file_list_level2:
        basename = os.path.basename(f)
        # Extract date pattern, e.g., A2010151
        date_part = basename.split("_")[3]  # Adjust index if needed
        files_by_date_level2[date_part].append(f)
    
    date_file_all_level2 = list(files_by_date_level2.values())
    dates_all_level2 = list(files_by_date_level2.keys())
    
    #====
    for files_level1, date_level1, files_level2, date_level2 in zip(date_file_all_level1, dates_all_level1, date_file_all_level2, dates_all_level2):
    
    # for files_level1, date_level1, files_level2, date_level2 in zip([date_file_all_level1[3]], [dates_all_level1[3]], [date_file_all_level2[3]], [dates_all_level2[3]]):

        
        if type_aggregation=="last_point":
            for variable_target in ["lwp", "Nd_max"]:
                plot_global_from_df(files=files_level1, 
                            title="MODIS spectral + ML", 
                            variable_target=variable_target, 
                            date=date_level1,
                            output_folder=output_folder)       
                
                plot_global_from_df(files=files_level2, 
                            title="MODIS retrievals", 
                            variable_target=variable_target,
                            date=date_level2,
                            output_folder=output_folder)       
    
        elif type_aggregation=="mean_by_date":
            # --------------------- create the nc average file -------------------------------
            
            for variable_target in ["lwp", "Nd_max"]:
                combine_average(files=files_level1, 
                            title="MODIS spectral + ML", 
                            variable_target=variable_target,
                            date=date_level1,
                            output_folder=output_folder)       
                
                combine_average(files=files_level2, 
                            title="MODIS retrievals", 
                            variable_target=variable_target,
                            date=date_level2,
                            output_folder=output_folder)    
                
            
        elif type_aggregation=="plot_mean_scatter":
            
            for variable_target in ["lwp", "Nd_max"]:
                file2 = f"{output_folder}/averaged_{date_level1}_{variable_target}_MODIS spectral + ML.nc"
                file1 = f"{output_folder}/averaged_{date_level2}_{variable_target}_MODIS retrievals.nc"
    
                # -------------------------- plots scatter -------------------------------
                if date_level1==date_level2:
                    plot_comparison(file1_retrievals=file1, 
                                file2_ml_predictions=file2, 
                                target_variable=variable_target, 
                                sample_percentage=1, 
                                date=date_level1,
                                output_folder=output_folder)
                else:
                    print("check dates")
    
                
        elif type_aggregation=="plot_hexbin":
            for variable_target in ["lwp", "Nd_max"]:
                file2 = f"{output_folder}/averaged_{date_level1}_{variable_target}_MODIS spectral + ML.nc"
                file1 = f"{output_folder}/averaged_{date_level2}_{variable_target}_MODIS retrievals.nc"
                if date_level1==date_level2:
    
                    plot_comparison_hexbin(file1_retrievals=file1, 
                                            file2_ml_predictions=file2, 
                                            target_variable=variable_target, 
                                            sample_percentage=1, #0.08, #0.01,
                                            date=date_level1,
                                            output_folder=output_folder)
                else:
                    print("check dates")

        
        elif type_aggregation=="plot_hist_plot2d_date":
            for variable_target in ["lwp", "Nd_max"]:
                file1 = f"{output_folder}/averaged_{date_level1}_{variable_target}_MODIS retrievals.nc"
                file2 = f"{output_folder}/averaged_{date_level1}_{variable_target}_MODIS spectral + ML.nc"
    
                histogram_plot2d(file1_retrievals=file1,
                                 file2_ml_predictions=file2,
                                 metrics=['R2', 'MAE'],
                                 name_model="NN",
                                 target_name=f'{variable_target}',
                                 prediction_name=f'{variable_target}',
                                 date=date_level2,
                                 log_plots="False")

        
        
        elif type_aggregation=="hist_mean" or type_aggregation=="plot_mean":
            if date_level1==date_level2:
                #------------------ lwp
                ds_modis_level1_lwp=xr.open_dataset(f"{output_folder}/averaged_{date_level1}_lwp_MODIS spectral + ML.nc")
                ds_modis_level2_lwp=xr.open_dataset(f"{output_folder}/averaged_{date_level2}_lwp_MODIS retrievals.nc")
            
                ds_modis_level2_lwp = ds_modis_level2_lwp[['lat', 'lon', 'lwp']]
                df_modis_level2_lwp = ds_modis_level2_lwp.to_dataframe()
            
                
                ds_modis_level1_lwp = ds_modis_level1_lwp[['lat', 'lon', 'lwp']]
                df_modis_level1_lwp = ds_modis_level1_lwp.to_dataframe()
                
                #------------------ ndmax
                ds_modis_level1_nd=xr.open_dataset(f"{output_folder}/averaged_{date_level1}_Nd_max_MODIS spectral + ML.nc")
                ds_modis_level2_nd=xr.open_dataset(f"{output_folder}/averaged_{date_level2}_Nd_max_MODIS retrievals.nc")
            
                ds_modis_level2_nd = ds_modis_level2_nd[['lat', 'lon', 'Nd_max']]
                df_modis_level2_nd = ds_modis_level2_nd.to_dataframe()
                
                ds_modis_level1_nd = ds_modis_level1_nd[['lat', 'lon', 'Nd_max']]
                df_modis_level1_nd = ds_modis_level1_nd.to_dataframe()
            
                # ---------------
                df_concatenate_level2 = pd.DataFrame({
                        'Nd_max': df_modis_level2_nd["Nd_max"],
                        'lwp': df_modis_level2_lwp["lwp"]
                    })
            
                df_concatenate_ml_level1 = pd.DataFrame({
                        'Nd_max': df_modis_level1_nd["Nd_max"],
                        'lwp': df_modis_level1_lwp["lwp"]
                    })
            
            
                histogram_method_comparison_modis(df_modis_retrievals=df_concatenate_level2, 
                                            df_modis_ml=df_concatenate_ml_level1, 
                                            clip_min=5, 
                                            clip_maximun=1000,
                                                  date=date_level1,
                                           result_folder=output_folder)
        
            else:
                    print("check dates")



     # no used next: 
     # for variable_target in ["lwp", "Nd_max"]:
        #     title="MODIS spectral + ML"
        #     ds_mean=xr.open_dataset(f"{output_folder}/averaged_{variable_target}_{title}.nc")
        #     plot_ds(ds_out=ds_mean, 
        #             variable_target=variable_target, 
        #             title=title,
        #             output_folder=output_folder)

            
        #     title="MODIS retrievals"
        #     ds_mean=xr.open_dataset(f"{output_folder}/averaged_{variable_target}_{title}.nc")
        #     plot_ds(ds_out=ds_mean, 
        #             variable_target=variable_target, 
        #             title=title,
        #             output_folder=output_folder)


    
