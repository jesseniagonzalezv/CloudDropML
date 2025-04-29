import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import seaborn as sns
import matplotlib.colors as colors
import pandas as pd

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm

def min_max_mean(variables):
    vmin = np.nanmin(variables)
    vmax = np.nanmax(variables)
    vmean = np.nanmean(variables)
    return vmin, vmax, vmean


def metric_evaluation(gt, pred):
    """
    Compute the SSIM, RMSE, R2 score and Mean Absolute Error (MAE) between two variables.

    Args:
        gt -- First variable for comparison (array)
        pred -- Second variable for comparison (array)

    Returns:
        result_ssim -- Structural Similarity Index (SSIM) between var1 and var2
        result_rmse -- Root Mean Square Error (RMSE) between var1 and var2
        result_score -- R2 Score between var1 and var2
        result_mae -- Mean Absolute Error (MAE) between var1 and var2
    """
    # Ensure gt and pred are numpy arrays 
    gt = np.array(gt)
    pred = np.array(pred)

    # Create a mask to ignore rows where the ground truth is NaN
    mask_gt_not_nan = ~np.isnan(gt)
    filtered_gt = gt[mask_gt_not_nan]
    filtered_pred = pred[mask_gt_not_nan]

    result_ssim = ssim(filtered_gt, filtered_pred)
    result_rmse = np.sqrt(mean_squared_error(filtered_gt, filtered_pred))
    result_score = r2_score(filtered_gt, filtered_pred)
    result_mae = mean_absolute_error(filtered_gt, filtered_pred)

    return result_ssim, result_rmse, result_score, result_mae


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


def histogram_plot2d(metrics, ds_model1, name_model, target_name, prediction_name, axes, plt, title_units, log_plots="False"):
    """
    ds_model1: ideally RF
    """

    axli = axes
    target_values = ds_model1[target_name].values.flatten()
    prediction_values = ds_model1[prediction_name].values.flatten()
    mask = ~np.isnan(target_values) & ~np.isnan(prediction_values)
    clean_target = target_values[mask]
    clean_prediction = prediction_values[mask]
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
    # Calcular el máximo de la densidad en el histograma
    max_hist_density = max(hist_values)
    y_max = 2*max_hist_density  # lwp

    sns.kdeplot(target_values, ax=axli[ax_idx], color='blue', label='Target', linewidth=2, bw_adjust=0.3)   # NaNs are ignored automatically
    sns.kdeplot(prediction_values, ax=axli[ax_idx], color='brown', label=f'{name_model} Prediction', linewidth=2,
                bw_adjust=0.3)
    axli[ax_idx].set_xlim(0, None)  # Set the lower x-limit to 0


    axli[ax_idx].set_ylim(0, y_max)
    # =========== end test limite superior =============    

    axli[ax_idx].set_ylabel('Density', fontsize=labels_font_size)
    axli[ax_idx].set_xlabel('Value', fontsize=labels_font_size)
    axli[ax_idx].legend(fontsize=axes_font_size)
    axli[ax_idx].tick_params(labelsize=axes_font_size)
    axli[ax_idx].set_title(f"{title_units}", fontsize=titles_font_size)

    ax_idx = 1

    axli[ax_idx].set_title(metrics_string, fontsize=subtitles_font_size)
    hb1 = axli[ax_idx].hexbin(target_values, prediction_values,
                              gridsize=50, cmap='viridis', mincnt=1, norm=colors.LogNorm())

    axli[ax_idx].set_xlabel('Target', fontsize=labels_font_size)
    axli[ax_idx].set_ylabel(f'{name_model} Prediction', fontsize=labels_font_size)
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



def comparison_target_predict(i, ds_model1, name_model, target_name, prediction_name,
                              axes, fig, title_units, cmap_img='Spectral'):
    """
    ds_model1: ideally NN
    """

    fig = fig
    axli = axes

    _, result_rmse_m1, result_score_m1, _ = metric_evaluation(ds_model1[target_name][i].values.flatten(),
                                                              ds_model1[prediction_name][i].values.flatten())

    vmin, vmax, vmean = min_max_mean([ds_model1[target_name], ds_model1[prediction_name]])
    vmax_99 = np.nanpercentile(ds_model1[target_name][i].values.flatten(), 99)

    print(f"\n--------------Maximun range used: {vmax_99} -------------------------- ")

    titles_font_size = 23
    axes_font_size = 19.5

    cmap1 = cm.get_cmap(cmap_img)
    cmap1.set_bad(color='lightgray')  # NaN light gray color 

    cmap2 = cm.get_cmap('RdBu')
    cmap2.set_bad(color='lightgray')  # NaN light gray color 


    ax_idx = 0
    im = ds_model1[target_name][i].plot(ax=axli[ax_idx], cmap=cmap1, vmin=vmin, vmax=vmax_99, add_colorbar=False)

    im.set_rasterized(True)  

    axli[ax_idx].add_feature(ccrs.cartopy.feature.BORDERS, color='black', linewidth=1)
    axli[ax_idx].add_feature(ccrs.cartopy.feature.COASTLINE, color='black', linewidth=1)

    # ------------------- X axis ------------------------------
    gl = axli[ax_idx].gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gl.xlabel_style = {'size': axes_font_size, 'color': 'black'}
    gl.ylabel_style = {'size': axes_font_size, 'color': 'black'}
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    
    axli[ax_idx].set_xlabel('')
    axli[ax_idx].set_ylabel(title_units, fontsize=titles_font_size)
    axli[ax_idx].set_title("ICON-LES", fontsize=titles_font_size)

    
    ax_idx = 1

    im = ds_model1[prediction_name][i].plot(ax=axli[ax_idx], cmap=cmap1, vmin=vmin, vmax=vmax_99,
                                            add_colorbar=False)  # cbar_kwargs={"label":"model1_RF / K"})
    
    im.set_rasterized(True)  

    axli[ax_idx].add_feature(ccrs.cartopy.feature.BORDERS, color='black', linewidth=1)
    axli[ax_idx].add_feature(ccrs.cartopy.feature.COASTLINE, color='black', linewidth=1)
    # ------------------- X axis ------------------------------
    gl = axli[ax_idx].gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gl.xlabel_style = {'size': axes_font_size, 'color': 'black'}
    gl.ylabel_style = {'size': axes_font_size, 'color': 'black'}
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()


    axli[ax_idx].set_xlabel('')
    axli[ax_idx].set_ylabel('')
    axli[ax_idx].set_title(f"{name_model} Prediction", fontsize=titles_font_size)

    cbar = fig.colorbar(im, ax=axli[ax_idx], location='right', fraction=0.037, pad=0.05)
    cbar.ax.tick_params(labelsize=axes_font_size - 4)

    ax_idx = 2
    percentage_diff = 100 * ((ds_model1[target_name][i] - ds_model1[prediction_name][i]) / (ds_model1[target_name][i]))
    
    min_target = ds_model1[target_name][i].min().item()
    max_target = ds_model1[target_name][i].max().item()
    min_prediction = ds_model1[prediction_name][i].min().item()
    max_prediction = ds_model1[prediction_name][i].max().item()

    print(f"Min Target: {min_target}, Max Target: {max_target}")
    print(f"Min Prediction: {min_prediction}, Max Prediction: {max_prediction}")


    # Normalize around zero
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    #                                norm=norm, shading='auto')
    plot = axli[ax_idx].pcolormesh(percentage_diff['lon'], percentage_diff['lat'], percentage_diff, cmap=cmap2,
                                   vmin=-40, vmax=40, shading='auto')
    plot.set_rasterized(True)  

    cbar2 = fig.colorbar(plot, ax=axli[ax_idx], location='right', fraction=0.037, pad=0.05)

    axli[ax_idx].add_feature(ccrs.cartopy.feature.BORDERS, color='black', linewidth=1)
    axli[ax_idx].add_feature(ccrs.cartopy.feature.COASTLINE, color='black', linewidth=1)

  
    gl = axli[ax_idx].gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gl.xlabel_style = {'size': axes_font_size, 'color': 'black'}
    gl.ylabel_style = {'size': axes_font_size, 'color': 'black'}
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    formatter = FuncFormatter(lambda x, pos: f'{x:.0f}%')
    cbar2.ax.xaxis.set_major_formatter(formatter)
    cbar2.ax.tick_params(labelsize=axes_font_size - 4)
    axli[ax_idx].set_title("\n % Difference", fontsize=titles_font_size)



def old_comparison_target_predict(i, ds_model1, name_model, target_name, prediction_name,
                              axes, fig, title_units, cmap_img='Spectral'):
    """
    ds_model1: ideally NN
    """

    fig = fig
    axli = axes

    _, result_rmse_m1, result_score_m1, _ = metric_evaluation(ds_model1[target_name][i].values.flatten(),
                                                              ds_model1[prediction_name][i].values.flatten())

    vmin, vmax, vmean = min_max_mean([ds_model1[target_name], ds_model1[prediction_name]])

    vmax_99 = np.nanpercentile(ds_model1[target_name][i].values.flatten(), 99)

    print(f"\n--------------Maximun range used: {vmax_99} -------------------------- ")

    titles_font_size = 23
    axes_font_size = 19.5

    cmap1 = cm.get_cmap(cmap_img)
    cmap1.set_bad(color='lightgray')  # NaN light gray color 

    cmap2 = cm.get_cmap('RdBu')
    cmap2.set_bad(color='lightgray')  # NaN light gray color 


    ax_idx = 0
    im = ds_model1[target_name][i].plot(ax=axli[ax_idx], cmap=cmap1, vmin=vmin, vmax=vmax_99, add_colorbar=False)
    im.set_rasterized(True)  
    axli[ax_idx].add_feature(ccrs.cartopy.feature.BORDERS, color='black', linewidth=1)
    axli[ax_idx].add_feature(ccrs.cartopy.feature.COASTLINE, color='black', linewidth=1)

    axli[ax_idx].set_xticks([5., 8., 11., 14.], crs=ccrs.PlateCarree())
    axli[ax_idx].set_xticklabels([5., 8., 11., 14.], color='black', fontsize=axes_font_size - 1)
    lon_formatter = cticker.LongitudeFormatter()
    axli[ax_idx].xaxis.set_major_formatter(lon_formatter)  # it gives the degre units E, N

    # ------------------- y axis ------------------------------
    axli[ax_idx].set_yticks([48., 50., 52., 54.], crs=ccrs.PlateCarree())
    axli[ax_idx].set_yticklabels([48., 50., 52., 54.], color='black', fontsize=axes_font_size)  # NN, CNN
    lat_formatter = cticker.LatitudeFormatter()
    axli[ax_idx].yaxis.set_major_formatter(lat_formatter)
    # -------------------
    
    axli[ax_idx].set_xlabel('')
    axli[ax_idx].set_ylabel(title_units, fontsize=titles_font_size)
    axli[ax_idx].set_title("ICON-LES", fontsize=titles_font_size)

    
    ax_idx = 1

    im = ds_model1[prediction_name][i].plot(ax=axli[ax_idx], cmap=cmap1, vmin=vmin, vmax=vmax_99,
                                            add_colorbar=False)  # cbar_kwargs={"label":"model1_RF / K"})
    im.set_rasterized(True)  
    axli[ax_idx].add_feature(ccrs.cartopy.feature.BORDERS, color='black', linewidth=1)
    axli[ax_idx].add_feature(ccrs.cartopy.feature.COASTLINE, color='black', linewidth=1)
    # ------------------- X axis ------------------------------
    axli[ax_idx].set_xticks([5., 8., 11., 14.], crs=ccrs.PlateCarree())
    axli[ax_idx].set_xticklabels([5., 8., 11., 14.], color='black', fontsize=axes_font_size - 1)
    lon_formatter = cticker.LongitudeFormatter()
    axli[ax_idx].xaxis.set_major_formatter(lon_formatter)  # it gives the degre units E, N
    axli[ax_idx].set_xlabel('')
    axli[ax_idx].set_ylabel('')
    axli[ax_idx].set_title(f"{name_model} Prediction", fontsize=titles_font_size)

    cbar = fig.colorbar(im, ax=axli[ax_idx], location='right', fraction=0.037, pad=0.05)
    cbar.ax.tick_params(labelsize=axes_font_size - 4)

    ax_idx = 2
    percentage_diff = 100 * ((ds_model1[target_name][i] - ds_model1[prediction_name][i]) / (ds_model1[target_name][i]))
    
    min_target = ds_model1[target_name][i].min().item()
    max_target = ds_model1[target_name][i].max().item()
    min_prediction = ds_model1[prediction_name][i].min().item()
    max_prediction = ds_model1[prediction_name][i].max().item()

    print(f"Min Target: {min_target}, Max Target: {max_target}")
    print(f"Min Prediction: {min_prediction}, Max Prediction: {max_prediction}")

    # Normalize around zero
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    plot = axli[ax_idx].pcolormesh(percentage_diff['lon'], percentage_diff['lat'], percentage_diff, cmap=cmap2,
                                   vmin=-40, vmax=40, shading='auto')
    plot.set_rasterized(True)  

    cbar2 = fig.colorbar(plot, ax=axli[ax_idx], location='right', fraction=0.037, pad=0.05)

    axli[ax_idx].add_feature(ccrs.cartopy.feature.BORDERS, color='black', linewidth=1)
    axli[ax_idx].add_feature(ccrs.cartopy.feature.COASTLINE, color='black', linewidth=1)

    axli[ax_idx].set_xticks([5., 8., 11., 14.], crs=ccrs.PlateCarree())
    axli[ax_idx].set_xticklabels([5., 8., 11., 14.], color='black', fontsize=axes_font_size - 1)
    lon_formatter = cticker.LongitudeFormatter()
    axli[ax_idx].xaxis.set_major_formatter(lon_formatter)  # it gives the degre units E, N

    # # ------------------- y axis ------------------------------
    formatter = FuncFormatter(lambda x, pos: f'{x:.0f}%')
    cbar2.ax.xaxis.set_major_formatter(formatter)
    cbar2.ax.tick_params(labelsize=axes_font_size - 4)
    axli[ax_idx].set_title("\n % Difference", fontsize=titles_font_size)


# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path_output', type=str, default="/work/bb1036/b381362/", help='for saving the results')
    arg('--path_pred_target_nc', type=str, default="/work/bb1036/b381362/output/file.nc",
        help='path where is the netcdfile')
    arg('--variable_target', type=str, default="lwp", help='"Re, COT, lwp, Nd_max" List of the variables to use.')
    arg('--metrics', nargs='+', type=str, default=['R2', 'RMSE'],
        help='List of the metrics to evaluate.')

    args = parser.parse_args()
    path_pred_target_nc = args.path_pred_target_nc
    path_output = args.path_output
    variable_target = args.variable_target
    metrics = args.metrics

    name_model = path_pred_target_nc.split("/")[-1].split("_")[5]
    file_name_part = path_pred_target_nc.split("/")[-1]

    pred_target_ds = xr.open_dataset(path_pred_target_nc)
    unique_name = file_name_part.replace('prediction_target_', '').rstrip('.nc')

    # Print all arguments
    print("Arguments received:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(repr(variable_target))  
    if variable_target == "lwp" or variable_target == 'LWP':
        variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif variable_target == "Nd_max" or variable_target == r'$N_{d,\ max}$':
        variable_target_units = r' $N_{\mathrm{d},\max}$ ($cm^{-3}$)'
    elif variable_target == 'Reff_cloud_top' or variable_target == r'$r_e$':
        variable_target_units = r'$r_e$ ($μm$)'
    elif variable_target == 'cot' or variable_target == 'COT':
        variable_target_units = 'COT'
    else:
        print(f"Variable target no encontrada:{variable_target}")

    num_times = list(range(len(pred_target_ds.time)))
    list_names = num_times + ["all"]

    for log_plots in ["True", "False"]:

        for sample_test in list_names:
            print(f"sample_test:{sample_test}")

            # --------------------------- Crear histogram_plot2d
            fig, axs = plt.subplots(constrained_layout=True, nrows=2, ncols=1, figsize=(4, 16 / 4 * 2))

            if sample_test == "all":
                histogram_plot2d(ds_model1=pred_target_ds,
                                 metrics=metrics,
                                 name_model=name_model,
                                 target_name=f'target_{variable_target}',
                                 prediction_name=f'prediction_{variable_target}',
                                 axes=axs,
                                 plt=plt,
                                 title_units=variable_target_units,
                                 log_plots=log_plots)
                if log_plots == "True":
                    name_file = f'{path_output}/log_density_2d_{unique_name}_all_test.pdf'
                elif log_plots == "False":
                    name_file = f'{path_output}/density_2d_{unique_name}_all_test.pdf'

                fig.savefig(name_file)
                plt.close()

                print(f"-------------------------------- Image save in {name_file}")

            else:
                time_data = pd.to_datetime(pred_target_ds.isel(time=sample_test).time.values)
                date_time_plot = time_data.strftime('%Y%m%dT%H%M')

                histogram_plot2d(ds_model1=pred_target_ds.isel(time=sample_test),
                                 metrics=metrics,
                                 name_model=name_model,
                                 target_name=f'target_{variable_target}',
                                 prediction_name=f'prediction_{variable_target}',
                                 axes=axs,
                                 plt=plt,
                                 title_units=variable_target_units,
                                 log_plots=log_plots)

                if log_plots == "True":
                    name_file = f'{path_output}/log_density_2d_{unique_name}_{date_time_plot}.pdf'
                elif log_plots == "False":
                    name_file = f'{path_output}/density_2d_{unique_name}_{date_time_plot}.pdf'

                fig.savefig(name_file)
                plt.close()
                print(f"-------------------------------- Image save in {name_file}")

            if sample_test != "all" and log_plots == "False":
                time_data = pd.to_datetime(pred_target_ds.isel(time=sample_test).time.values)
                date_time_plot = time_data.strftime('%Y%m%dT%H%M')

                fig, axs = plt.subplots(constrained_layout=True, subplot_kw={'projection': ccrs.PlateCarree()}, nrows=1,
                                        ncols=3, figsize=(18 / 4 * 3, 3.8))  # Rf

                comparison_target_predict(i=sample_test,
                                          ds_model1=pred_target_ds,
                                          name_model=name_model,
                                          target_name=f'target_{variable_target}',
                                          prediction_name=f'prediction_{variable_target}',
                                          axes=axs,
                                          fig=fig,
                                          title_units=variable_target_units,
                                          # cmap='Blues')
                                          cmap_img="plasma")

                name_file = f'{path_output}/distribution_{unique_name}_{date_time_plot}.pdf'
                fig.savefig(name_file, dpi=100)

                plt.close()
                # plt.show()
                print(f"-------------------------------- Image save in {name_file} \n --------------------------------")


if __name__ == '__main__':
    main()
