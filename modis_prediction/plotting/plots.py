
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker


from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
# Define a formatter to convert the log values back to the original scale.
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter, FixedLocator


def plot_geospatial_comparison(variable_target, datasets, result_folder, titles=None, cmap="jet", vmin=1e0, vmax=1e3, index=None):
    """
    Plots 4 horizontal geospatial subplots with shared y-axis and a single vertical colorbar.

    Parameters:
    - variable_name: str, variable to plot
    - datasets: list of xarray datasets
    - titles: list of subplot titles
    - cmap: str, colormap name
    - vmin, vmax: float, for LogNorm scaling
    - units: str, units to display in colorbar label
    """
    if variable_target == "lwp" or variable_target == 'LWP':
        variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif variable_target == "Nd_max" or variable_target == r'$N_{d,\ max}$':
        # variable_target_units = r' $N_{\mathrm{d},\max}$ ($cm^{-3}$)'
        variable_target_units = r' $N_{\mathrm{d}}$ ($cm^{-3}$)'

    titles_font_size = 23
    axes_font_size = 19.5
    plots_num = len(datasets)
    fig, axes = plt.subplots(
        1, plots_num, figsize=(20/3.5*plots_num, 9),
        subplot_kw={'projection': ccrs.PlateCarree()},
        sharey=True,
        constrained_layout=True  # this avoids overlapping
    )
    # fig.subplots_adjust(wspace=0.05, right=0.88)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    for i, (ds, ax) in enumerate(zip(datasets, axes)):
        var = ds[variable_target]
        im = var.plot.pcolormesh(
            ax=ax,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            add_colorbar=False
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {'size': axes_font_size}
        gl.ylabel_style = {'size': axes_font_size}
        gl.xlocator = mticker.MaxNLocator(4)
        gl.ylocator = mticker.MaxNLocator(4)

        if i > 0:
            gl.left_labels = False
        # else:
        #     # Solo para el primer subplot (más a la izquierda)
        #     ax.set_ylabel(f"{variable_name} [{units}]", fontsize=12)
        elif i == 0:
            ax.text(
                -0.28, 0.5, f"{variable_target_units}",
                va='center', ha='center',
                rotation='vertical',
                transform=ax.transAxes,
                fontsize=titles_font_size
            )

        if titles:
            ax.set_title(titles[i], fontsize=titles_font_size)

    cbar = fig.colorbar(im, ax=axes, location='right', fraction=0.006, pad=0.02)
    cbar.ax.tick_params(labelsize=axes_font_size-2)
    # path_file = f"{result_folder}/{variable_target}_methods_comparison_distribution.pdf"
    # fig.savefig(path_file, bbox_inches='tight', dpi=100) # con pdf interpolate se tendria q rasterized for now not needed
    path_file = f"{result_folder}/{index}_{variable_target}_methods_comparison_distribution.png"
    plt.savefig(path_file, bbox_inches='tight', dpi=300)
    plt.close()  
    # plt.show()


def histogram_method_comparison_modis(df_modis_retrievals, df_modis_ml, result_folder, clip_min=0, clip_maximun=2000, index="all"):
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
            sns.kdeplot(data=df_modis_retrievals[r'$N_{d,\ MODIS L2}$'], fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
            sns.kdeplot(data=df_modis_ml[r'$N_{d,\ MODIS L1}$'], fill=False, clip=clip_range, common_norm=False, label='MODIS spectral+ ML')
            plt.xlabel(r'$N_{\mathrm{d}}$ ($cm^{-3}$)', fontsize=14)
        # plt.title('KDE of Nd across different data sources', fontsize=15)
    
        elif variable_target=="LWP":
            # KDE plot for LWP
            sns.kdeplot(data=df_modis_retrievals[r'LWP_{MODIS L2}'], fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
            sns.kdeplot(data=df_modis_ml[r'LWP_{MODIS L1}'], fill=False, clip=clip_range, common_norm=False, label='MODIS spectral + ML')
            plt.xlabel(r'$L$ ($\mathrm{g\,m}^{-2}$)', fontsize=14)
    
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
 
        # Adjust layout to prevent overlap
        plt.tight_layout()
        if index=="all":
            path_file = f"{result_folder}/{variable_target}_comparison_modis_kde_all.png"
        elif index!="all":
            path_file = f"{result_folder}/{variable_target}_comparison_modis_kde_{index}.png"

        plt.savefig(path_file, bbox_inches="tight")
        plt.close()
    

def histogram_method_comparison(df_icon_les, df_modis_retrievals, df_ml_rttov, df_modis_ml, result_folder, clip_min=0, clip_maximun=2000):
    clip_range = (clip_min, clip_maximun)

    for variable_target in ["Nd_max", "LWP"]:
        plt.figure(figsize=(7, 5)) 

        # custom_palette = ["#d62728", "#E17C05", "#1f77b4", "#2ca02c"]
        # custom_palette = [
        #                     "red",  #
        #                     # "#D4A017",  # mostaza
        #                     "#006400",  # 
        #                     "#8c564b",  #  green
        #                     "#A55194",  # dusty purple
        #                 ]
        custom_palette = [
                            "#D55E00",  # 
                            "#006400",  # verde
                            "#1f77b9",  # 
                            "#A55194",  # purple
                        ]

        sns.set_palette(custom_palette)

        if variable_target=="Nd_max":
            # KDE plot for N_d
            sns.kdeplot(data=df_icon_les[r'$N_{d,\ ICON-LES}$'], fill=False, clip=clip_range, common_norm=False, label='ICON-LES')
            sns.kdeplot(data=df_modis_retrievals[r'$N_{d,\ MODIS L2}$'], fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
            sns.kdeplot(data=df_ml_rttov[r'$N_{d,\ MODIS simulated}$'], fill=False, clip=clip_range, common_norm=False, label='MODIS simulation + ML')
            sns.kdeplot(data=df_modis_ml[r'$N_{d,\ MODIS L1}$'], fill=False, clip=clip_range, common_norm=False, label='MODIS spectral + ML')
            plt.xlabel(r'$N_{\mathrm{d}}$ ($cm^{-3}$)', fontsize=14)
        # plt.title('KDE of Nd across different data sources', fontsize=15)


        elif variable_target=="LWP":
            # KDE plot for LWP
            sns.kdeplot(data=df_icon_les[r'LWP_{ICON-LES}'], fill=False, clip=clip_range, common_norm=False, label='ICON-LES')
            sns.kdeplot(data=df_modis_retrievals[r'LWP_{MODIS L2}'], fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
            sns.kdeplot(data=df_ml_rttov[r'LWP_{MODIS simulated}'], fill=False, clip=clip_range, common_norm=False, label='MODIS simulation + ML')
            sns.kdeplot(data=df_modis_ml[r'LWP_{MODIS L1}'], fill=False, clip=clip_range, common_norm=False, label='MODIS spectral + ML')
            plt.xlabel(r'$L$ ($\mathrm{g\,m}^{-2}$)', fontsize=14)
    
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
 
        # Adjust layout to prevent overlap
        plt.tight_layout()
        path_file = f"{result_folder}/{variable_target}_methods_comparison_kde.png"
        plt.savefig(path_file, bbox_inches="tight")
        plt.close()


def log_histogram_method_comparison(df_icon_les, df_modis_retrievals, df_ml_rttov, df_modis_ml, result_folder, clip_min=0, clip_maximun=2000):
    clip_range = (clip_min, clip_maximun)

    for variable_target in ["Nd_max", "LWP"]:
        plt.figure(figsize=(7, 5)) 

        # custom_palette = ["#d62728", "#E17C05", "#1f77b4", "#2ca02c"]
        # custom_palette = [
        #                     "red",  #
        #                     # "#D4A017",  # mostaza
        #                     "#006400",  # 
        #                     "#8c564b",  #  green
        #                     "#A55194",  # dusty purple
        #                 ]
        custom_palette = [
                            "#D55E00",  # 
                            "#006400",  # verde
                            "#1f77b9",  # 
                            "#A55194",  # purple
                        ]

        sns.set_palette(custom_palette)

        if variable_target=="Nd_max":
            # KDE plot for N_d
            sns.kdeplot(data=np.log(df_icon_les[r'$N_{d,\ ICON-LES}$']), fill=False, clip=clip_range, common_norm=False, label='ICON-LES')
            sns.kdeplot(data=np.log(df_modis_retrievals[r'$N_{d,\ MODIS L2}$']), fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
            sns.kdeplot(data=np.log(df_ml_rttov[r'$N_{d,\ MODIS simulated}$']), fill=False, clip=clip_range, common_norm=False, label='MODIS simulation + ML')
            sns.kdeplot(data=np.log(df_modis_ml[r'$N_{d,\ MODIS L1}$']), fill=False, clip=clip_range, common_norm=False, label='MODIS spectral + ML')
            plt.xlabel(r'$N_{\mathrm{d}}$ ($cm^{-3}$)', fontsize=14)
        # plt.title('KDE of Nd across different data sources', fontsize=15)


        elif variable_target=="LWP":
            # KDE plot for LWP
            sns.kdeplot(data=np.log(df_icon_les[r'LWP_{ICON-LES}']), fill=False, clip=clip_range, common_norm=False, label='ICON-LES')
            sns.kdeplot(data=np.log(df_modis_retrievals[r'LWP_{MODIS L2}']), fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
            sns.kdeplot(data=np.log(df_ml_rttov[r'LWP_{MODIS simulated}']), fill=False, clip=clip_range, common_norm=False, label='MODIS simulation + ML')
            sns.kdeplot(data=np.log(df_modis_ml[r'LWP_{MODIS L1}']), fill=False, clip=clip_range, common_norm=False, label='MODIS spectral + ML')
            plt.xlabel(r'$L$ ($\mathrm{g\,m}^{-2}$)', fontsize=14)
    
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
 
        # Adjust layout to prevent overlap
        plt.tight_layout()
        path_file = f"{result_folder}/log_{variable_target}_methods_comparison_kde.png"
        plt.savefig(path_file, bbox_inches="tight")
        plt.close()

def histogram_method_comparison_old(df_icon_les, df_modis_retrievals, df_ml_rttov, df_modis_ml, result_folder, clip_min=0, clip_maximun=2000):
    clip_range = (clip_min, clip_maximun)

    # Set up the figure and axes
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column of plots
    
    # KDE plot for N_d
    sns.kdeplot(data=df_icon_les[r'$N_{d,\ ICON-LES}$'], ax=ax[0], fill=False, clip=clip_range, common_norm=False, label='ICON-LES')
    sns.kdeplot(data=df_modis_retrievals[r'$N_{d,\ MODIS L2}$'], ax=ax[0], fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
    sns.kdeplot(data=df_ml_rttov[r'$N_{d,\ MODIS simulated}$'], ax=ax[0], fill=False, clip=clip_range, common_norm=False, label='MODIS simulation + ML')
    sns.kdeplot(data=df_modis_ml[r'$N_{d,\ MODIS L1}$'], ax=ax[0], fill=False, clip=clip_range, common_norm=False, label='MODIS + ML')
    
    ax[0].set_xlabel(r'$N_{d,\ max}$ ($cm^{-3}$)', fontsize=14)
    ax[0].set_ylabel('Density', fontsize=14)
    ax[0].set_title('KDE of Nd across different data sources', fontsize=15)
    ax[0].legend(fontsize=14)
    
    # KDE plot for LWP
    sns.kdeplot(data=df_icon_les[r'LWP_{ICON-LES}'], ax=ax[1], fill=False, clip=clip_range, common_norm=False, label='ICON-LES')
    sns.kdeplot(data=df_modis_retrievals[r'LWP_{MODIS L2}'], ax=ax[1], fill=False, clip=clip_range, common_norm=False, label='MODIS retrievals')
    sns.kdeplot(data=df_ml_rttov[r'LWP_{MODIS simulated}'], ax=ax[1], fill=False, clip=clip_range, common_norm=False, label='MODIS simulation + ML')
    sns.kdeplot(data=df_modis_ml[r'LWP_{MODIS L1}'], ax=ax[1], fill=False, clip=clip_range, common_norm=False, label='MODIS + ML')
    
    ax[1].set_xlabel('LWP ($gm^{-2}$)', fontsize=14)
    ax[1].set_ylabel('Density', fontsize=14)
    ax[1].set_title('KDE of LWP across different data sources', fontsize=15)
    ax[1].legend(fontsize=14)
    plt.tight_layout()
    path_file = f"{result_folder}/lwp_nd_methods_comparison_kde.png"
    plt.savefig(path_file, bbox_inches="tight")
    
    # plt.show()

def plot_pcolormesh(longitude, latitude, variable, name_var, level_name):
    """
    Input:
        -- variable ()
        -- label_name (str): can be "lwp", "Nd_max" or "channel_number: value"
    """
    plt.figure(figsize=(6, 5))
    # plt.pcolormesh(longitude, latitude, variable, shading='auto', cmap='viridis')
    plt.pcolormesh(variable, shading='auto', cmap='viridis')

    if ':' in name_var:
        # Extract the number from the variable_name
        channel_number = int(label_name.split(':')[0])
    elif ':' in name_var:
        channel_number = int(label_name.split(':')[0])
        
    if name_var == 'lwp':
        label_name = f'LWP ($gm^{-2}$)'
    elif name_var == 'Nd_max':
        label_name = r'$N_{d,\ max}$ ($cm^{-3}$)'
    elif name_var == 'Nd':
        label_name = r'$N_{d}$ ($cm^{-3}$)'
        
    elif (1 <= channel_number <= 19) or channel_number == 26:
        label_name = f'Reflectance for Channel {label_name}'
    elif (20 <= channel_number <= 25) or (27 <= channel_number <= 36):
        label_name = f'Radiances (W/$m^2$/sr/µm) for Channel {label_name}'
            
    plt.colorbar(label=label_name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'MODIS (Level {level_name})')
    # plt.show()
    plt.close()


def plot_pcolormesh_no_lat_lon(variable, name_var, level_name):
    """
    Input:
        -- variable ()
        -- label_name (str): can be "lwp", "Nd_max" or "channel_number: value"
    """
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(variable, shading='auto', cmap='viridis')

    if ':' in name_var:
        channel_number = int(label_name.split(':')[0])
    elif ':' in name_var:
        channel_number = int(label_name.split(':')[0])
        
    if name_var == 'lwp':
        label_name = f'LWP ($gm^{-2}$)'
    elif name_var == 'Nd_max':
        label_name = r'$N_{d,\ max}$ ($cm^{-3}$)'
        
    elif (1 <= channel_number <= 19) or channel_number == 26:
        label_name = f'Reflectance for Channel {label_name}'
    elif (20 <= channel_number <= 25) or (27 <= channel_number <= 36):
        label_name = f'Radiances (W/$m^2$/sr/µm) for Channel {label_name}'
            
    plt.colorbar(label=label_name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'MODIS (Level {level_name})')
    plt.close()
    # plt.show()


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def get_joint_histogram(nx, ny, xmin, xmax, ymin, ymax, x, y):
    jh = np.empty([ny, nx])
    y_median = np.empty([nx])
    x_bins = np.linspace(xmin, xmax, nx + 1)
    y_bins = np.linspace(ymin, ymax, ny + 1)
    x_mid = x_bins[:-1] + (x_bins[1] - x_bins[0]) / 2
    labels = ['Bin {}'.format(i) for i in range(1, nx + 1)]
    x_bin = pd.cut(x, x_bins, labels=labels)
    i = 0
    for bin in labels:
        y_median[i] = np.median(y[x_bin == bin]) if np.any(x_bin == bin) else np.nan
        i += 1
    return jh, x_mid, y_median


def get_PDF_bin_range(x, x_bins):
    dict_PDF = {}
    hist, bin_edges = np.histogram(x, bins=x_bins)
    dict_PDF['x'] = bin_edges[0:len(x_bins) - 1] + (bin_edges[1] - bin_edges[0]) / 2  # mid value
    dict_PDF['pdf'] = hist / sum(hist) * 100
    return dict_PDF


def get_values_joint_hist(xedges, yedges):
    xedges_mid = xedges[:-1] + (xedges[1] - xedges[0]) / 2
    yedges_mid = yedges[:-1] + (yedges[1] - yedges[0]) / 2
    print("========print xedges_mid", xedges_mid)
    print("========print yedges_mid", yedges_mid)
    return (xedges_mid, yedges_mid)


def generate_dynamic_ticks(data, num_ticks=7):
    """
    Generate tick marks based on the quartiles and range of the data.
    Args:
        data (pd.Series or np.array): The data to base the ticks on.
        num_ticks (int): The desired number of tick marks.
    Returns:
        ticks (np.array): An array of tick marks.
    """
    min_val, max_val = np.log(np.min(data)), np.log(np.max(data))

    # Generate ticks linearly spaced within the log-transformed range
    ticks = np.linspace(min_val, max_val, num_ticks)

    return np.exp(ticks)


# Format ticks to show the original scale
def loge_to_original(x, pos):
    return f"{np.exp(x):.0f}"  # Convert log-e value to original scale and format


def plot_joint_histogram(df, x_col, y_col, figure_size=(10, 8), num_ticks=5, threshold="filterlog",
                         min_value_threshold=1e-4):
    """
    Plot a joint histogram showing the relationship between two log-transformed variables.

    Args:
    - df: DataFrame containing the data.
    - x_col: The column name for the x-axis variable.
    - y_col: The column name for the y-axis variable.
    - figure_size: Tuple specifying the size of the figure (width, height).
    - num_ticks: Number of ticks to display on the x and y axes.
    - min_value_threshold: Minimum value threshold to filter out near-zero values.

    Returns:
    - fig: The matplotlib figure object.
    - ax: The matplotlib axes object.
    """

    # Ensure data is strictly positive before applying log transformation
    df_filtered = df.loc[(df[x_col] > min_value_threshold) & (df[y_col] > min_value_threshold)].copy()
    # df_filtered = df.copy()

    # if threshold == "range3-870":
    if threshold == "range7-870":
        # Filter and set limits 
        # xmin, xmax = 3, 870
        # ymin, ymax = 3, 800
        # xmin, xmax = 7, 870
        # ymin, ymax = 7, 800

        xmin, xmax = 5, 870
        ymin, ymax = 5, 800
        
        df_filtered = df_filtered.loc[
            (df_filtered[f'{x_col}'] > xmin) &
            (df_filtered[f'{x_col}'] < xmax) &
            (df_filtered[f'{y_col}'] > ymin) &
            (df_filtered[f'{y_col}'] < ymax)
            ].copy()
        
        df_filtered[f'Log_{x_col}'] = np.log(df_filtered[x_col])
        df_filtered[f'Log_{y_col}'] = np.log(df_filtered[y_col])
        
        xmin = np.min(df_filtered[f'Log_{x_col}'])
        xmax = np.max(df_filtered[f'Log_{x_col}'])
        ymin = np.min(df_filtered[f'Log_{y_col}'])
        ymax = np.max(df_filtered[f'Log_{y_col}'])
        
    if threshold == "nofilter":
        xmin = np.min(df_filtered[f'{x_col}'])
        xmax = np.max(df_filtered[f'{x_col}'])
        ymin = np.min(df_filtered[f'{y_col}'])
        ymax = np.max(df_filtered[f'{y_col}'])
        # print("antes", xmin, xmax, ymin, ymax)
    
        # Apply log transformation and create new columns in the DataFrame
        df_filtered[f'Log_{x_col}'] = np.log(df_filtered[x_col])
        df_filtered[f'Log_{y_col}'] = np.log(df_filtered[y_col])
        
        xmin = np.min(df_filtered[f'Log_{x_col}'])
        xmax = np.max(df_filtered[f'Log_{x_col}'])
        ymin = np.min(df_filtered[f'Log_{y_col}'])
        ymax = np.max(df_filtered[f'Log_{y_col}'])


    elif threshold == "filter":
        # Filter and set limits based on quantiles
        xmin, xmax = df_filtered[f'{x_col}'].quantile([0.01, 0.99])
        ymin, ymax = df_filtered[f'{y_col}'].quantile([0.01, 0.99])

        # a =(df_filtered[x_col]> xmax).sum()
        # print(f"higher that {xmax}: {a} ({a*100/len(df_filtered)})")
        # b =(df_filtered[y_col]> ymax).sum()
        # print(f"higher that {ymax}: {b} ({b*100/len(df_filtered)})")
        # print("despues", xmin, xmax, ymin, ymax)

        df_filtered = df_filtered.loc[
            (df_filtered[f'{x_col}'] > xmin) &
            (df_filtered[f'{x_col}'] < xmax) &
            (df_filtered[f'{y_col}'] > ymin) &
            (df_filtered[f'{y_col}'] < ymax)
            ].copy()

        df_filtered[f'Log_{x_col}'] = np.log(df_filtered[x_col])
        df_filtered[f'Log_{y_col}'] = np.log(df_filtered[y_col])
        
        xmin = np.min(df_filtered[f'Log_{x_col}'])
        xmax = np.max(df_filtered[f'Log_{x_col}'])
        ymin = np.min(df_filtered[f'Log_{y_col}'])
        ymax = np.max(df_filtered[f'Log_{y_col}'])

    elif threshold == "filterlog":
        df_filtered[f'Log_{x_col}'] = np.log(df_filtered[x_col])
        df_filtered[f'Log_{y_col}'] = np.log(df_filtered[y_col])

        # Filter and set limits based on quantiles
        xmin, xmax = df_filtered[f'Log_{x_col}'].quantile([0.01, 0.99])
        ymin, ymax = df_filtered[f'Log_{y_col}'].quantile([0.01, 0.99])

        df_filtered = df_filtered.loc[
            (df_filtered[f'Log_{x_col}'] > xmin) &
            (df_filtered[f'Log_{x_col}'] < xmax) &
            (df_filtered[f'Log_{y_col}'] > ymin) &
            (df_filtered[f'Log_{y_col}'] < ymax)
            ].copy()

    # Create joint plot
    joint_plot = sns.jointplot(
        x=f'Log_{x_col}',
        y=f'Log_{y_col}',
        data=df_filtered,
        kind='hex',
        # gridsize=100,
        gridsize=15, # test to compare with conditional
        cmap='viridis',
        height=figure_size[1]  # Height is controlled here, width is managed by the aspect ratio
    )

    fig = joint_plot.fig
    fig.set_size_inches(*figure_size)  # Set the figure size

    # Adjust axis limits to focus on relevant data range, min, max are in log (log axis limits)
    joint_plot.ax_joint.set_xlim(xmin, xmax)
    joint_plot.ax_joint.set_ylim(ymin, ymax)

    # Overlay the median curve on the joint plot (assuming get_joint_histogram is defined)
    # nx, ny = 100, 100
    nx, ny = 15, 15  # test to compare with conditional
    jh, x_mid, y_median = get_joint_histogram(nx, ny, xmin, xmax, ymin, ymax,
                                              df_filtered[f'Log_{x_col}'].values,
                                              df_filtered[f'Log_{y_col}'].values)

    joint_plot.ax_joint.plot(x_mid, y_median, color='black', linewidth=2, label='Median Fit')

    # Generate dynamic ticks for x and y axes, ticks original
    # original_Nd_ticks = generate_dynamic_ticks(df_filtered[x_col], num_ticks=num_ticks)
    # original_LWP_ticks = generate_dynamic_ticks(df_filtered[y_col], num_ticks=num_ticks)
    
    # Desired ticks in the original scale
    desired_ticks_x = [10, 50, 200, 800]
    desired_ticks_y = [10, 50, 200, 600, 800]
    
    # Convert desired ticks to log base e
    log_ticks_x = np.log(desired_ticks_x)
    log_ticks_y = np.log(desired_ticks_y)
    
    # Set major ticks at the specified positions
    joint_plot.ax_joint.xaxis.set_major_locator(FixedLocator(log_ticks_x))
    joint_plot.ax_joint.yaxis.set_major_locator(FixedLocator(log_ticks_y))
    formatter = FuncFormatter(loge_to_original)
    joint_plot.ax_joint.xaxis.set_major_formatter(formatter)
    joint_plot.ax_joint.yaxis.set_major_formatter(formatter)
    
    joint_plot.ax_joint.tick_params(labelsize=14)

    # # Apply log-transformed ticks
    # joint_plot.ax_joint.set_xticks(np.log(original_Nd_ticks))  # Log-transformed x ticks
    # joint_plot.ax_joint.set_yticks(np.log(original_LWP_ticks))  # Log-transformed y ticks
    # joint_plot.ax_joint.set_yticklabels([f'{tick:.2f}' for tick in original_LWP_ticks])
    # joint_plot.ax_joint.set_xticklabels([f'{tick:.2f}' for tick in original_Nd_ticks])

    # Set labels and title
    # joint_plot.ax_joint.set_xlabel(f'Log {x_col} ($cm^{-3}$)', fontsize=12)
    # joint_plot.ax_joint.set_ylabel(f'Log {y_col} ($gm^{-2}$)', fontsize=12)

    if ':' in x_col:
        # Extract the number from the variable_name
        channel_number = int(x_col.split(':')[0])
    elif ':' in y_col:
        channel_number = int(y_col.split(':')[0])

    if x_col == 'lwp':
        label_name_x = f'LWP ($gm^{-2}$)'
    elif x_col == 'Nd_max':
        # label_name_x = r'$N_{d,\ max}$ ($cm^{-3}$)'
        label_name_x = r'$N_{d}$ ($cm^{-3}$)'
    elif x_col == 'Reff_cloud_top':
        label_name_x = r'$r_e$ ($μm$)'
    elif x_col == 'cot':
        label_name_x = 'COT'
    # else:
    #     # Assign variable name  based on the channel_number
    #     if (1 <= channel_number <= 19) or channel_number == 26:
    #         label_name_x = f'Reflectance for Channel {x_col}'
    #     elif (20 <= channel_number <= 25) or (27 <= channel_number <= 36):
    #         label_name_x = f'Radiances (W/$m^2$/sr/µm) for Channel {x_col}'
    #     else:
    #         a = 'unknown'  # Optionally handle cases that do not fit any range

    if y_col == 'lwp':
        label_name_y = f'LWP ($gm^{-2}$)'
    elif y_col == 'cot':
        label_name_y = 'COT'
    elif y_col == 'Nd_max':
        label_name_y = r'$N_{d,\ max}$ ($cm^{-3}$)'
    elif y_col == 'Reff_cloud_top':
        label_name_y = r'$r_e$ ($μm$)'
    # else:
    #     # Assign variable name  based on the channel_number
    #     if (1 <= channel_number <= 19) or channel_number == 26:
    #         label_name_y = f'Reflectance for Channel {y_col}'
    #     elif (20 <= channel_number <= 25) or (27 <= channel_number <= 36):
    #         label_name_y = f'Radiances (W/$m^2$/sr/µm) for Channel {y_col}'
    #     else:
    #         a = 'unknown'  # Optionally handle cases that do not fit any range

    joint_plot.ax_joint.set_xlabel(f'{label_name_x}', fontsize=15)
    joint_plot.ax_joint.set_ylabel(f'{label_name_y}', fontsize=15)

    # Set title below the density curve, above the main plot
    joint_plot.ax_joint.set_title('Joint Plot with Log-Transformed Variables', fontsize=14)
    # , pad=20)

    # Adjust layout and return the figure and axis
    fig.tight_layout()
    fig.subplots_adjust()
    # top=0.9)

    return fig, joint_plot.ax_joint


def histograms_dataframe(df, name_file):
    # Plotting histograms for all columns
    print(len(df.columns))
    if len(df.columns) == 36:
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 25))
        axes = axes.flatten()
    elif len(df.columns) == 2:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
        axes = axes.flatten()
    print(len(axes))

    for i, col in enumerate(df.columns):
        # print("-----i", i)
        if ':' in col:
            # Extract the number from the variable_name
            channel_number = int(col.split(':')[0])

        if col == 'lwp':
            label_name_x = f'LWP ($gm^{-2}$)'
        elif col == 'Nd_max':
            label_name_x = r'$N_{d,\ max}$ ($cm^{-3}$)'
        elif col == 'Reff_cloud_top':
            label_name_x = r'$r_e$ ($μm$)'
        elif col == 'cot':
            label_name_x = 'COT'
        else:
            # Assign variable name  based on the channel_number
            if (1 <= channel_number <= 19) or channel_number == 26:
                label_name_x = f'Reflectance for Ch. {col}'
            elif (20 <= channel_number <= 25) or (27 <= channel_number <= 36):
                label_name_x = f'Radiances (W/$m^2$/sr/µm) for Ch. {col}'
            else:
                label_name_x = 'unknown'  # Optionally handle 
                
        df[col].plot(kind='hist', ax=axes[i], bins=50, edgecolor='black')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlabel(label_name_x)
    
    plt.tight_layout()
    # plt.savefig(name_file)
    # plt.show()


def loge_to_original(x, pos):
    return f"{np.exp(x):.0f}"

    
def get_joint_histogram_conditional(nx, ny, xmin, xmax, ymin, ymax, x, y):
    """
    Compute a joint histogram where for each x-bin the counts of y are converted to
    percentages (conditional probability of y given x).
    
    Parameters:
        nx, ny : int
            Number of bins for x and y.
        xmin, xmax : float
            The limits for the x-axis (log scale).
        ymin, ymax : float
            The limits for the y-axis (log scale).
        x, y : array-like
            The log-transformed data arrays.
    
    Returns:
        cond_hist : 2D array (shape [ny, nx])
            The conditional probabilities (in percent) for each (x,y) bin.
        x_mid : 1D array
            The midpoints of the x bins.
        y_mid : 1D array
            The midpoints of the y bins.
        y_median : 1D array
            The median y value for each x-bin.
    """
    # Create bin edges for x and y
    x_bins = np.linspace(xmin, xmax, nx + 1)
    y_bins = np.linspace(ymin, ymax, ny + 1)
    # Calculate bin centers (used for the median curve)
    x_mid = (x_bins[:-1] + x_bins[1:]) / 2
    y_mid = (y_bins[:-1] + y_bins[1:]) / 2

    cond_hist = np.zeros((ny, nx))
    y_median = np.empty(nx)
    
    # Assign each x value to a bin
    x_indices = np.digitize(x, x_bins) - 1

    global_total = np.sum([np.sum(x_indices == i) for i in range(nx)])  # Total points across all x-bins
        
    for i in range(nx):
        mask = (x_indices == i)
        if np.sum(mask) > 0:
            counts, _ = np.histogram(y[mask], bins=y_bins)
            total = counts.sum()
            if total > 0:
                cond_hist[:, i] = counts / total * 100
            else:
                cond_hist[:, i] = np.nan
            y_median[i] = np.median(y[mask])
        else:
            cond_hist[:, i] = np.nan
            y_median[i] = np.nan
            
    return cond_hist, x_mid, y_mid, y_median


def plot_joint_histogram_conditional(df, x_col, y_col, figure_size=(10, 8), num_ticks=5, threshold="filterlog",
                                     min_value_threshold=1e-4):
    # Filter the dataframe to remove small (or invalid) values.
    df_filtered = df.loc[(df[x_col] > min_value_threshold) & (df[y_col] > min_value_threshold)].copy()

    # if threshold == "range3-870":
    if threshold == "range7-870":
        # Set desired ranges for filtering.
        # xmin, xmax = 5, 700
        # ymin, ymax = 2, 800
        xmin, xmax = 5, 870
        ymin, ymax = 5, 800
        # # xmin, xmax = 5, 1000
        # ymin, ymax = 5, 1000
        # xmin, xmax = 7, 870
        # ymin, ymax = 7, 800
        
        df_filtered = df_filtered.loc[
            (df_filtered[x_col] > xmin) &
            (df_filtered[x_col] < xmax) &
            (df_filtered[y_col] > ymin) &
            (df_filtered[y_col] < ymax)
        ].copy()
        
        # Compute the log-transformed columns.
        df_filtered[f'Log_{x_col}'] = np.log(df_filtered[x_col])
        df_filtered[f'Log_{y_col}'] = np.log(df_filtered[y_col])
        
        # Update limits to be in the log scale.
        xmin = df_filtered[f'Log_{x_col}'].min()
        xmax = df_filtered[f'Log_{x_col}'].max()
        ymin = df_filtered[f'Log_{y_col}'].min()
        ymax = df_filtered[f'Log_{y_col}'].max()
    
    elif threshold == "filterlog":
        df_filtered[f'Log_{x_col}'] = np.log(df_filtered[x_col])
        df_filtered[f'Log_{y_col}'] = np.log(df_filtered[y_col])

        # Filter and set limits based on quantiles
        xmin, xmax = df_filtered[f'Log_{x_col}'].quantile([0.01, 0.99])
        ymin, ymax = df_filtered[f'Log_{y_col}'].quantile([0.01, 0.99])

        df_filtered = df_filtered.loc[
            (df_filtered[f'Log_{x_col}'] > xmin) &
            (df_filtered[f'Log_{x_col}'] < xmax) &
            (df_filtered[f'Log_{y_col}'] > ymin) &
            (df_filtered[f'Log_{y_col}'] < ymax)
            ].copy()

    else:
        # If not using the "range3-870" threshold, still create the log columns.
        df_filtered[f'Log_{x_col}'] = np.log(df_filtered[x_col])
        df_filtered[f'Log_{y_col}'] = np.log(df_filtered[y_col])
        xmin = df_filtered[f'Log_{x_col}'].min()
        xmax = df_filtered[f'Log_{x_col}'].max()
        ymin = df_filtered[f'Log_{y_col}'].min()
        ymax = df_filtered[f'Log_{y_col}'].max()

    # Create the joint plot. This plot includes the hexbin joint plot
    # plus the marginal histograms.
    joint_plot = sns.jointplot(
        x=f'Log_{x_col}',
        y=f'Log_{y_col}',
        data=df_filtered,
        kind='hex',
        # gridsize=50,
        # gridsize=30,
        gridsize=20,
        cmap='viridis',
        height=figure_size[1],
        marginal_kws=dict(bins=20)  # Ensure binning is aligned

    )
    
    fig = joint_plot.fig
    fig.set_size_inches(*figure_size)  # Set overall figure size

    # Set axis limits (the limits are in log scale).
    joint_plot.ax_joint.set_xlim(xmin, xmax)
    joint_plot.ax_joint.set_ylim(ymin, ymax)

    # Compute the conditional probability histogram.
    # nx, ny = 100, 100
    # nx, ny = 30, 30
    nx, ny = 20, 20
    cond_hist, x_mid, y_mid, y_median = get_joint_histogram_conditional(
        nx, ny, xmin, xmax, ymin, ymax,
        df_filtered[f'Log_{x_col}'].values,
        df_filtered[f'Log_{y_col}'].values
    )
    
    # Compute the bin edges (needed for pcolormesh, which expects an array
    # with one more element than the dimension of the data array).
    x_bins = np.linspace(xmin, xmax, nx + 1)
    y_bins = np.linspace(ymin, ymax, ny + 1)
    
    # Overlay the conditional probability using pcolormesh.
    # mesh = joint_plot.ax_joint.pcolormesh(x_bins, y_bins, cond_hist, shading='auto', cmap='viridis')
    # Define custom colormap with white at 0, green in low, yellow in middle, red at high
    colors = [(1, 1, 1), (0, 0.5, 0), (1, 1, 0), (0.8, 0, 0)]  # White → Green → Yellow → Red
    positions = [0, 0.2, 0.6, 1]  # Control the transition points
    # Create colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))

    mesh = joint_plot.ax_joint.pcolormesh(x_bins, y_bins, cond_hist, shading='auto', cmap=custom_cmap, vmin=0)  #RdYlGn_r
    
    # Add a colorbar to indicate the conditional probability (%) scale.
    cbar = fig.colorbar(mesh, ax=joint_plot.ax_joint)
    cbar.set_label('Conditional Probability (%)', fontsize=14)
    
    # Overlay the median curve computed for each x bin.
    joint_plot.ax_joint.plot(x_mid, y_median, color='black', linewidth=2, label='Median Fit')

    # Define desired tick marks (in the original scale), convert them to log-scale.
    desired_ticks_x = [10, 50, 200, 800]
    desired_ticks_y = [10, 50, 200, 600, 800]
    log_ticks_x = np.log(desired_ticks_x)
    log_ticks_y = np.log(desired_ticks_y)
    
    # Set the major ticks and apply the formatter.
    joint_plot.ax_joint.xaxis.set_major_locator(FixedLocator(log_ticks_x))
    joint_plot.ax_joint.yaxis.set_major_locator(FixedLocator(log_ticks_y))
    formatter = FuncFormatter(loge_to_original)
    joint_plot.ax_joint.xaxis.set_major_formatter(formatter)
    joint_plot.ax_joint.yaxis.set_major_formatter(formatter)
    joint_plot.ax_joint.tick_params(labelsize=14)
    
    # Set the axis labels and title.
    # label_name_x = 'N'
    # label_name_y = 'L'
    
    # label_name_y = f'LWP ($gm^{-2}$)'
    # label_name_x = r'$N_{d,\ max}$ ($cm^{-3}$)'
    label_name_y = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    label_name_x = r'$N_{\mathrm{d}}$ ($cm^{-3}$)'

    joint_plot.ax_joint.set_xlabel(label_name_x, fontsize=15)
    joint_plot.ax_joint.set_ylabel(label_name_y, fontsize=15)
    joint_plot.ax_joint.set_title('Log-Transformed Variables', fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust()
    
    return fig, joint_plot.ax_joint


def plot_lwp_nd_from_predictions_nc(path_ml_prediction_nd, path_ml_prediction_lwp, min_value_threshold, max_value_threshold, result_folder):
    for type_data in ['prediction', 'target']:
        print(f" ----------------- type data: {type_data}")
            
        ds_nd = xr.open_dataset(path_ml_prediction_nd)
        # print(ds_nd.time.values)
        # month = "05"
        # day= "02"
        # hr = "15"
        # ds_nd = ds_nd.sel(time=f'2013-{month}-{day}T{hr}:00')
        
        ds_lwp = xr.open_dataset(path_ml_prediction_lwp)
        # ds_lwp = ds_lwp.sel(time=f'2013-{month}-{day}T{hr}:00')
    
        df_nd_max = ds_nd.to_dataframe()
        df_lwp = ds_lwp.to_dataframe()
        
        # Combine into a new dataframe
        combined_df = pd.DataFrame({
            'Nd_max': df_nd_max[f'{type_data}_Nd_max'],
            'lwp': df_lwp[f'{type_data}_lwp']
        })
        print(f"Nd range: {combined_df['Nd_max'].min()} to {combined_df['Nd_max'].max()}")
        print(f"LWP range: {combined_df['lwp'].min()} to {combined_df['lwp'].max()}")
    
        filtered_df = filter_dataframe_cloud(combined_df,
                                             min_value_threshold=min_value_threshold,
                                             max_value_threshold=max_value_threshold)
        
        path_file = f"{result_folder}/Nd_lwp_ml_{type_data}_m{month}d{day}T{hr}.png"
        fig, ax = plot_joint_histogram(filtered_df, variable[0], variable[1], threshold="range7-870", figure_size=(6, 6))
        fig.savefig(path_file, bbox_inches="tight")

        fig, ax = plot_joint_histogram_conditional(filtered_df, 
               filtered_df.columns[0], 
               filtered_df.columns[1], 
               threshold="range7-870", 
               figure_size=(6, 6))         
        path_file = f"{result_folder}/Nd_lwp_modis_retrievals_CP_from_nc.png"
        fig.savefig(path_file, bbox_inches="tight")
        



def min_max_mean(variables):
    vmin = np.nanmin(variables)
    vmax = np.nanmax(variables)
    vmean = np.nanmean(variables)
    return vmin, vmax, vmean


def histogram_plot2d(target_values, prediction_values, name_model, variable_target,  metrics, axes, plt, log_plots="False"): #ds_model1, target_name, prediction_name, title_units, 
    """
    ds_model1: ideally RF
    """

    if variable_target == "lwp" or variable_target == 'LWP':
        title_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif variable_target == "Nd_max" or variable_target == r'$N_{d,\ max}$':
        # variable_target_units = r' $N_{\mathrm{d},\max}$ ($cm^{-3}$)'
        title_units = r' $N_{\mathrm{d}}$ ($cm^{-3}$)'

    axli = axes
    # target_values = ds_model1[target_name].values.flatten()
    # prediction_values = ds_model1[prediction_name].values.flatten()
    
    mask = ~np.isnan(target_values) & ~np.isnan(prediction_values)
    clean_target = target_values[mask]
    clean_prediction = prediction_values[mask]

    #------------------------------------------------------   
    min_value_threshold = 5
    max_value_threshold = 1000
    joint_mask = (
        (target_values >= min_value_threshold) &
        (target_values <= max_value_threshold) &
        (prediction_values >= min_value_threshold) &
        (prediction_values <= max_value_threshold)
    )
    
    # Apply the joint mask to both DataFrames
    clean_target = target_values[joint_mask]
    clean_prediction = prediction_values[joint_mask]
    #------------------------------------------------------   

    if log_plots == "False":
        target_values = clean_target
        prediction_values = clean_prediction 
    if log_plots == "True":
        target_values = np.log(clean_target)
        prediction_values = np.log(clean_prediction) 

    vmin, vmax, vmean = min_max_mean([target_values,
                                      prediction_values])
    vmin_true, vmax_true, vmean_true = min_max_mean([target_values])

    # result_rmse, result_score, result_mae, result_mape, result_nrmse, result_rmspe, vmin_true, vmax_true, vmean_true, std_dev_true = metric_evaluation_nrmse(
    #     target_values,
    #     prediction_values)
    values_metrics_list = []

    for metric_name in metrics:
        
        if metric_name == 'mean':
            values_metrics = rf"MODIS mean: {np.nanmean(target_values):.1f}"
            values_metrics_list.append(values_metrics)
        
            values_metrics = rf"Prediction mean: {np.nanmean(prediction_values):.1f}"
            values_metrics_list.append(values_metrics)
        
    #     if metric_name == 'R2':
            # values_metrics = rf"$R^2$: {result_score:.2f}"
    #         values_metrics_list.append(values_metrics)
    #     elif metric_name == 'nRMSE':
    #         values_metrics = rf"nRMSE (%): {result_nRMSE:.2f}"
    #         values_metrics_list.append(values_metrics)
    #     elif metric_name == "RMSPE":
    #         values_metrics = rf"RMSPE (%): {result_rmspe:.2f}"
    #         values_metrics_list.append(values_metrics)
    #     elif metric_name == "MAPE":
    #         values_metrics = rf"MAPE (%): {result_mape:.2f}"
    #         values_metrics_list.append(values_metrics)
    #     elif metric_name == "RMSE":
    #         values_metrics = rf"RMSE: {result_rmse:.2f}"
    #         values_metrics_list.append(values_metrics)
    #     elif metric_name == "MAE":
    #         values_metrics = rf"MAE: {result_mae:.2f}"
    #         values_metrics_list.append(values_metrics)

    # # Join the list into a single string separated by commas
    # metrics_string = ", ".join(values_metrics_list)
    metrics_string = "\n ".join(values_metrics_list)
    
    titles_font_size = 17
    subtitles_font_size = 16
    labels_font_size = 16
    axes_font_size = 14

    vmax_99 = np.nanpercentile(target_values.flatten(), 99) #1140

    # if variable_target == "lwp":
    #     vmax_99 = 300 # lwp only 2016


    ax_idx = 0

    print("mean------------------------", vmean)
    hist_values, _ = np.histogram(target_values, bins=30, density=True)
    max_hist_density = max(hist_values)
    # y_max = 1.45*max_hist_density  # nd
    y_max = 2*max_hist_density  # lwp
    # ===========

    sns.kdeplot(target_values, ax=axli[ax_idx], color='blue', label='MODIS retrievals', linewidth=2, bw_adjust=0.3)   # NaNs are ignored automatically
    sns.kdeplot(prediction_values, ax=axli[ax_idx], color='brown', label='MODIS spectral + NN', linewidth=2,
                bw_adjust=0.3)
    axli[ax_idx].set_xlim(0, vmax_99)  # Set the lower x-limit to 0


    # Aplicar límite al eje Y
    axli[ax_idx].set_ylim(0, y_max)
    axli[ax_idx].set_ylabel('Density', fontsize=labels_font_size)
    axli[ax_idx].set_xlabel('Value', fontsize=labels_font_size)
    axli[ax_idx].legend(fontsize=axes_font_size)
    axli[ax_idx].tick_params(labelsize=axes_font_size)
    axli[ax_idx].set_title(f"{title_units}", fontsize=titles_font_size)
    # axli[ax_idx].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax_idx = 1

    axli[ax_idx].set_title(metrics_string, fontsize=subtitles_font_size)
    hb1 = axli[ax_idx].hexbin(target_values, prediction_values,
                              gridsize=70, cmap='viridis', mincnt=7, norm=colors.LogNorm()) #', gridsize=50  mincnt=1  # gridsize 100 sale muy chiquito el ndmax
    # norm=colors.LogNorm())

    axli[ax_idx].set_xlabel('Modis retrievals', fontsize=labels_font_size)
    axli[ax_idx].set_ylabel(f'{name_model} Prediction', fontsize=labels_font_size)
    axli[ax_idx].tick_params(labelsize=axes_font_size)
    axli[ax_idx].set_xlim(vmin, vmax_99)
    axli[ax_idx].set_ylim(vmin, vmax_99)
    axli[ax_idx].set_aspect('equal')
    ticks = np.linspace(vmin, vmax_99, 4)
    axli[ax_idx].set_xticks(ticks)
    axli[ax_idx].set_yticks(ticks)

    line = axli[ax_idx].plot([vmin_true, vmax_99], [vmin_true, vmax_99], color='red')  # Perfect prediction line

    cb = plt.colorbar(hb1, ax=axli[ax_idx], location='bottom', )
    cb.ax.tick_params(labelsize=axes_font_size)



def plot_geospatial_comparison_scatter(variable_target, dataframes, result_folder, titles=None,
                                       cmap="Spectral_r", vmin=1e0, vmax=1e3, index=None):
    """
    Plots side-by-side scatter maps from a list of dataframes with lat/lon and a target variable.

    Parameters:
    - variable_target: str, variable to plot (must be a column in each dataframe)
    - dataframes: list of pandas DataFrames, each containing 'lat', 'lon', and variable_target columns
    - result_folder: str, path to save the figure
    - titles: list of subplot titles
    - cmap: str, colormap name
    - vmin, vmax: float, for LogNorm scaling
    - index: identifier for filename
    - date: str, date string for title (optional)
    """

    if variable_target.lower() == "lwp":
        variable_target_units = r'$L$ ($\mathrm{g\,m}^{-2}$)'
    elif variable_target.lower() == "nd_max":
        variable_target_units = r'$N_{\mathrm{d}}$ ($cm^{-3}$)'
    else:
        variable_target_units = variable_target


    titles_font_size = 23
    axes_font_size = 19.5
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    log_ticks = [10, 20, 50, 100, 200, 500, 1000]

    fig, axes = plt.subplots(
        1, len(dataframes), figsize=(20/3.5 * len(dataframes), 9),
        subplot_kw={'projection': ccrs.PlateCarree()},
        sharey=True,
        constrained_layout=True
    )

    if len(dataframes) == 1:
        axes = [axes]  # Ensure axes is iterable

    for i, (df, ax) in enumerate(zip(dataframes, axes)):
        sc = ax.scatter(df['lon'], df['lat'], c=df[variable_target],
                        norm=norm, cmap=cmap, s=1, transform=ccrs.PlateCarree())

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {'size': axes_font_size}
        gl.ylabel_style = {'size': axes_font_size}
        gl.xlocator = mticker.MaxNLocator(4)
        gl.ylocator = mticker.MaxNLocator(4)

        if i > 0:
            gl.left_labels = False
        elif i == 0:
            ax.text(
                -0.28, 0.5, f"{variable_target_units}",
                va='center', ha='center',
                rotation='vertical',
                transform=ax.transAxes,
                fontsize=titles_font_size
            )

        if titles:
            # ax.set_title(f"{titles[i]} - {index}", fontsize=titles_font_size)
            ax.set_title(f"{titles[i]}", fontsize=titles_font_size)
        # elif date:
            # ax.set_title(f"{variable_target} - {index}", fontsize=titles_font_size)

    # Add shared colorbar
    cbar = fig.colorbar(sc, ax=axes, location='right', fraction=0.009, pad=0.02)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([str(t) for t in log_ticks])
    cbar.ax.tick_params(labelsize=axes_font_size - 2)

    # Save figure
    path_file = f"{result_folder}/{index}_{variable_target}_scatter_comparison_distribution.png"
    plt.savefig(path_file, bbox_inches='tight', dpi=300)
    plt.close()


