

import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import xarray as xr
import argparse
import numpy as np
import os

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import linregress
import math
from PIL import Image
from sklearn.preprocessing import StandardScaler



def save_correlation_plot(corr_matrix, title, output_path, only_2variables):
    """
    This function saves a heatmap plot of the correlation matrix to a PNG file.

    Args:
        corr_matrix (pd.DataFrame): A correlation matrix DataFrame.
        title (str): The title of the plot.
        output_path (str): The full path where the PNG file should be saved.
    """

    titles_font_size = 16
    subtitles_font_size = 15
    labels_font_size = 15
    axes_font_size = 13

    num_variable = corr_matrix.shape[0]

    if num_variable == 2:
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='RdBu',
            vmin=-1,
            vmax=1,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            linecolor='black',
            cbar=False,  # Do not plot the color bar
            annot_kws={"size": 13}  
        )
    else:
        if only_2variables == "True":
            titles_font_size = 17
            subtitles_font_size = 16
            labels_font_size = 15  # 16
            axes_font_size = 14

            plt.figure(figsize=(3, 8))  # 3

            sns.heatmap(
                corr_matrix.iloc[:, :2],
                annot=True,
                fmt='.1f',
                cmap='RdBu',
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                linecolor='black', 
                cbar=False,
                annot_kws={"size": 13}  
            )

        else:
            plt.figure(figsize=(14, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(
                corr_matrix,
                cmap='RdBu',
                annot=True,
                fmt='.1f',
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                linecolor='black',
                cbar=True,
                mask=mask,
                cbar_kws={'orientation': 'vertical', 'shrink': 0.8},
            )

    plt.title(title, fontsize=titles_font_size)  
    plt.xticks(fontsize=labels_font_size, rotation=90)  
    plt.yticks(fontsize=labels_font_size)  
    # plt.tight_layout()
    plt.tight_layout(pad=0.5)  
    plt.savefig(f"{output_path}.png", bbox_inches='tight')  
    plt.close()

    corr_matrix = corr_matrix.round(3)
    corr_matrix.to_csv(f'{output_path}.csv', index=True)



def pearson_correlation_and_save(df, output_path, only_2variables=None):
    """
    This function calculates the Pearson correlation matrix and saves it as a heatmap PNG file.

    Args:
        df (pd.DataFrame): A pandas DataFrame where each column represents a different variable.
        output_path (str): The full path where the PNG file should be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the Pearson correlation coefficients between the variables.
    """
    
    corr_matrix = df.corr(method='pearson')
    save_correlation_plot(corr_matrix, 'Pearson\nCorrelation', output_path, only_2variables)
    return corr_matrix


def spearman_correlation_and_save(df, output_path, only_2variables=None):
    """
    This function calculates the Spearman correlation matrix and saves it as a heatmap PNG file.

    Args:
        df (pd.DataFrame): A pandas DataFrame where each column represents a different variable.
        output_path (str): The full path where the PNG file should be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the Spearman correlation coefficients between the variables.
    """
    
    corr_matrix = df.corr(method='spearman')
    save_correlation_plot(corr_matrix, 'Spearman\nCorrelation', output_path, only_2variables)
    return corr_matrix


def kendall_correlation_and_save(df, output_path, only_2variables=None):
    """
    This function calculates the kendall correlation matrix and saves it as a heatmap PNG file.

    Args:
        df (pd.DataFrame): A pandas DataFrame where each column represents a different variable.
    output_path (str): The full path where the PNG file should be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the Spearman correlation coefficients between the variables.
    """
    corr_matrix = df.corr(method='kendall')
    save_correlation_plot(corr_matrix, 'Kendall Correlation', output_path, only_2variables)
    return corr_matrix

def load_dataframe_from_netcdf(path_data, prefix):
    """
    Loads a DataFrame from a netCDF file based on the provided path, prefix, and fold number.

    Args:
        path_data: str, the directory where the netCDF files are stored.
        prefix: str, the prefix of the file to identify the type of data (e.g., 'df_icon_pca_train').
        fold_num: int, the fold number for which the data should be loaded.

    Returns:
        df: pandas.DataFrame, the DataFrame loaded from the netCDF file.
    """
    
    filename = f'{path_data}/{prefix}.nc'
    xr_dataset = xr.open_dataset(filename)
    df = xr_dataset.to_dataframe()
    return df


def get_xarray_rad_ref(simulation_ds, data_type="ref_rad_total"):
    """
    Generate a new xarray Dataset merging reflectance and radiance data based on the provided data type.
    The function creates a new Dataset based on the 'ref_rad_total' data type, combining reflectance
    ("ref") and radiance ("rad") values from the input Dataset. The resulting Dataset has reflectance
    and radiance values in designated channel ranges.

    Args:
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
            list(range(0, 19)) + [25])  
        desired_channels_rad = np.array(list(range(19, 25)) + list(
            range(26, 36)))  

        # Delete unwanted attributes
        del new_ds.attrs['standard_name']
        del new_ds.attrs['long_name']
        del new_ds.attrs['units']

        new_ds.attrs['ref_units'] = simulation_ds['ref_total'].attrs['units']
        new_ds.attrs['rad_units'] = simulation_ds['rad_total'].attrs['units']

        # Adding which bands are ref and which are rad
        new_ds.attrs['ref_bands_idx'] = desired_channels_ref + 1
        new_ds.attrs['rad_bands_idx'] = desired_channels_rad + 1

        new_ds.attrs['ref_long_name'] = 'All-sky upwelling reflectance at TOA'
        new_ds.attrs['rad_long_name'] = 'All-sky upwelling radiance at TOA'

      
        # ---------------------- only 18 channels -------------------------------
        name_channels = np.array(['1: 0.645 μm', '2: 0.856 μm', '3: 0.466 μm', '4: 0.554 μm',
       '5: 1.241 μm', '6: 1.628 μm', '7: 2.113 μm', '8: 0.412 μm',
       '9: 0.442 μm', '10: 0.487 μm', '11: 0.530 μm', '12: 0.547 μm',
       '13: 0.666 μm', '14: 0.678 μm', '15: 0.747 μm', '16: 0.867 μm',
       '17: 0.904 μm', '18: 0.936 μm', '19: 0.935 μm', '20: 3.777 μm',
       '21: 3.981 μm', '22: 3.971 μm', '23: 4.061 μm', '24: 4.448 μm',
       '25: 4.526 μm', '26: 1.382 μm', '27: 6.783 μm', '28: 7.344 μm',
       '29: 8.550 μm', '30: 9.720 μm', '31: 11.017 μm', '32: 12.036 μm',
       '33: 13.362 μm', '34: 13.683 μm', '35: 13.923 μm', '36: 14.213 μm'])
        index_channel_relate_clouds = [0,1,2,3,4,5,6,19,20,21,22,28,30,31,32,33,34,35]
        channel_relate_clouds = name_channels[index_channel_relate_clouds]
        combined_labels = channel_relate_clouds
        new_ds.attrs['chan_index'] = np.array(index_channel_relate_clouds) + 1
        new_ds = new_ds.isel(chan=index_channel_relate_clouds)
        # -------------------------------------------------------
        
        new_ds['chan'] = combined_labels

    return new_ds


# Function definitions
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


def plot_mutual_information_heatmap(X, Y, output_path, figsize=(15, 3), cmap='RdBu'):
    """
    Calculates and plots the mutual information between input variables (X) and output variables (Y) as a heatmap.
    The function accepts DataFrames and uses column names for labeling.

    Args:
        X : DataFrame, shape (n_samples, n_features)
            The input variables. Each column represents a different input feature.
        Y : DataFrame, shape (n_samples, n_outputs)
            The output variables. Each column represents a different output variable.
        output_path: path where is save the png iamge 
        figsize : tuple, optional, default=(12, 4)
            Size of the figure for the heatmap.
        cmap : str, optional, default='RdBu'
            The color map to use for the heatmap.

    Returns:
        None
        Save the figure
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(Y, pd.DataFrame):
        raise ValueError("X and Y must be pandas DataFrames")

    mi_matrix = np.zeros((X.shape[1], Y.shape[1]))

    for i, input_name in enumerate(X.columns):  
        for j, output_name in enumerate(Y.columns):  
            mi_matrix[i, j] = mutual_info_regression(X[[input_name]], Y[output_name])

    mi_df = pd.DataFrame(mi_matrix, index=X.columns, columns=Y.columns)

    titles_font_size = 17
    subtitles_font_size = 16
    labels_font_size = 15  # 16
    axes_font_size = 14

    plt.figure(figsize=figsize)
    sns.heatmap(mi_df, annot=True, fmt='.1f', cmap=cmap, cbar=True, linecolor='black', linewidths=0.7, annot_kws={"size": 13}
)

    plt.title('Mutual\nInformation', fontsize=titles_font_size)  
    plt.xticks(fontsize=labels_font_size, rotation=90)  
    plt.yticks(fontsize=labels_font_size-1)  
    plt.tight_layout(pad=0.5)  
    plt.savefig(f"{output_path}.png", bbox_inches='tight') 
    plt.close()
    
    mi_df.round(3)
    mi_df.to_csv(f'{output_path}.csv', index=True)


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

    ticks = np.linspace(min_val, max_val, num_ticks)

    return np.exp(ticks)


def convert_fractional_day_to_time(time_varible: float) -> (int, int, int, int, int, int):
    """
    Args:
        time_varible (float64) -- Given a time value as a fractional day in the format YYYYMMDD.fractionalday. For example time = convert_fractional_day_to_time(ds.time.values)

    Returns:
        formatted_datetimes (string) -- '%Y%m%dT%H%M' representing the specified date and calculated time (year, month, day, hours, minutes).
    """
    
    # Split the input into date and fractional day parts
    date, fractional_day = divmod(time_varible, 1)

    # Convert the date part to an integer and then to a string for processing
    date_str = str(int(date))

    # Extract year, month, and day from the date string
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Convert fractional day to hours, minutes, and seconds
    hours = fractional_day * 24
    minutes = (hours - int(hours)) * 60
    seconds = (minutes - int(minutes)) * 60

    # Round to the nearest whole numbers
    hours = math.floor(hours)
    minutes = math.floor(minutes)
    seconds = round(seconds)

    # If seconds are 60, increment minutes by one and reset seconds to zero
    if seconds == 60:
        minutes += 1
        seconds = 0

    # If minutes are 60, increment hours by one and reset minutes to zero
    if minutes == 60:
        hours += 1
        minutes = 0

    time_data = pd.Timestamp(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds)

    formatted_datetimes = time_data.strftime('%Y-%m-%dT%H:%M')

    # print(formatted_datetimes)

    return time_data


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


def plot_joint_histogram(df, x_col, y_col, figure_size=(10, 8), num_ticks=5, threshold="filterlog",
                         min_value_threshold=1e-4):
    """
    Plot a joint histogram showing the relationship between two log-transformed variables.

    Args:
        df: DataFrame containing the data.
        x_col: The column name for the x-axis variable.
        y_col: The column name for the y-axis variable.
        figure_size: Tuple specifying the size of the figure (width, height).
        num_ticks: Number of ticks to display on the x and y axes.
        min_value_threshold: Minimum value threshold to filter out near-zero values.

    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.
    """

    # Ensure data is strictly positive before applying log transformation
    df_filtered = df.loc[(df[x_col] > min_value_threshold) & (df[y_col] > min_value_threshold)].copy()

    xmin = np.min(df_filtered[f'{x_col}'])
    xmax = np.max(df_filtered[f'{x_col}'])
    ymin = np.min(df_filtered[f'{y_col}'])
    ymax = np.max(df_filtered[f'{y_col}'])

    df_filtered[f'Log_{x_col}'] = np.log(df_filtered[x_col])
    df_filtered[f'Log_{y_col}'] = np.log(df_filtered[y_col])

    # % ---------------test
    if threshold == "nofilter":
        xmin = np.min(df_filtered[f'Log_{x_col}'])
        xmax = np.max(df_filtered[f'Log_{x_col}'])
        ymin = np.min(df_filtered[f'Log_{y_col}'])
        ymax = np.max(df_filtered[f'Log_{y_col}'])


    elif threshold == "filter":
        # Filter and set limits based on quantiles
        xmin, xmax = df_filtered[f'{x_col}'].quantile([0.01, 0.99])
        ymin, ymax = df_filtered[f'{y_col}'].quantile([0.01, 0.99])

        df_filtered = df_filtered.loc[
            (df_filtered[f'{x_col}'] > xmin) &
            (df_filtered[f'{x_col}'] < xmax) &
            (df_filtered[f'{y_col}'] > ymin) &
            (df_filtered[f'{y_col}'] < ymax)
            ].copy()

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
        gridsize=100,
        cmap='viridis',
        height=figure_size[1]  
    )

    fig = joint_plot.fig
    fig.set_size_inches(*figure_size)  

    # Adjust axis limits to focus on relevant data range
    joint_plot.ax_joint.set_xlim(xmin, xmax)
    joint_plot.ax_joint.set_ylim(ymin, ymax)

    # Overlay the median curve on the joint plot (assuming get_joint_histogram is defined)
    nx, ny = 100, 100
    jh, x_mid, y_median = get_joint_histogram(nx, ny, xmin, xmax, ymin, ymax,
                                              df_filtered[f'Log_{x_col}'].values,
                                              df_filtered[f'Log_{y_col}'].values)

    joint_plot.ax_joint.plot(x_mid, y_median, color='black', linewidth=2, label='Median Fit')

    # Generate dynamic ticks for x and y axes
    original_Nd_ticks = generate_dynamic_ticks(df_filtered[x_col], num_ticks=num_ticks)
    original_LWP_ticks = generate_dynamic_ticks(df_filtered[y_col], num_ticks=num_ticks)

    # Apply these ticks to the plot
    joint_plot.ax_joint.set_xticks(np.log(original_Nd_ticks))
    joint_plot.ax_joint.set_xticklabels([f'{tick:.2f}' for tick in original_Nd_ticks])
    joint_plot.ax_joint.set_yticks(np.log(original_LWP_ticks))
    joint_plot.ax_joint.set_yticklabels([f'{tick:.2f}' for tick in original_LWP_ticks])

    # Set labels and title
    if ':' in x_col:
        # Extract the number from the variable_name
        channel_number = int(x_col.split(':')[0])
    elif ':' in y_col:
        channel_number = int(y_col.split(':')[0])

    if x_col == 'lwp':
        label_name_x = f'LWP ($gm^{-2}$)'
    elif x_col == 'Nd_max':
        label_name_x = r'$N_{d,\ max}$ ($cm^{-3}$)'
    elif x_col == 'Reff_cloud_top':
        label_name_x = r'$r_e$ ($μm$)'
    elif x_col == 'cot':
        label_name_x = 'COT'
    else:
        # Assign variable name  based on the channel_number
        if (1 <= channel_number <= 19) or channel_number == 26:
            label_name_x = f'Reflectance for Channel {x_col}'
        elif (20 <= channel_number <= 25) or (27 <= channel_number <= 36):
            label_name_x = f'Radiances (W/$m^2$/sr/µm) for Channel {x_col}'
        else:
            a = 'unknown'  

    if y_col == 'lwp':
        label_name_y = f'LWP ($gm^{-2}$)'
    elif y_col == 'cot':
        label_name_y = 'COT'
    elif y_col == 'Nd_max':
        label_name_y = r'$N_{d,\ max}$ ($cm^{-3}$)'
    elif y_col == 'Reff_cloud_top':
        label_name_y = r'$r_e$ ($μm$)'
    else:
        # Assign variable name  based on the channel_number
        if (1 <= channel_number <= 19) or channel_number == 26:
            label_name_y = f'Reflectance for Channel {y_col}'
        elif (20 <= channel_number <= 25) or (27 <= channel_number <= 36):
            label_name_y = f'Radiances (W/$m^2$/sr/µm) for Channel {y_col}'
        else:
            a = 'unknown'  

    joint_plot.ax_joint.set_xlabel(f'{label_name_x}', fontsize=13)
    joint_plot.ax_joint.set_ylabel(f'{label_name_y}', fontsize=13)

    joint_plot.ax_joint.set_title('Joint Plot with Log-Transformed Variables', fontsize=14)
    # , pad=20)

    fig.tight_layout()
    fig.subplots_adjust()
    # top=0.9)

    return fig, joint_plot.ax_joint



def combine_images_grid(file_list, grid_size, output_path):
    images = [Image.open(file) for file in file_list]
    if not images:
        return  # If no images, do nothing

    # Calculate the maximum dimensions of a single image
    max_width, max_height = max(i.size[0] for i in images), max(i.size[1] for i in images)

    # Grid dimensions
    grid_width, grid_height = grid_size

    # Create a new image with appropriate dimensions
    new_image = Image.new('RGB', (max_width * grid_width, max_height * grid_height))

    # Paste images into the new image
    for index, image in enumerate(images):
        x = index % grid_width * max_width
        y = index // grid_width * max_height
        new_image.paste(image, (x, y))

    new_image.save(output_path)

    if os.path.exists(output_path):
        print(f"Figure saved at: {output_path}")
    else:
        print(f"Figure was not saved")


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
                label_name_x = 'unknown'  
                
        df[col].plot(kind='hist', ax=axes[i], bins=50, edgecolor='black')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlabel(label_name_x)
    
    plt.tight_layout()
    plt.savefig(name_file)
    plt.show()


def change_names_features(variables_names, data_df):    
    if variables_names == "Re-COT" or variables_names == "Re-COT-channels":
        # Rename columns according to your specifications
        data_df = data_df.rename(columns={
            'Reff_cloud_top': r'$r_e$',  # LaTeX style for subscript
            'cot': 'COT'  # All uppercase
    
        })
    
    
    elif variables_names == "LWP-Ndmax" or variables_names == "LWP-Ndmax-channels":
        # Rename columns according to your specifications
        data_df = data_df.rename(columns={
            'Nd_max': r'$N_{d,\ max}$',
            'lwp': 'LWP'
        })


    return data_df

def filter_datafraframe(df, variables_names):
    if variables_names == "LWP-Ndmax" or variables_names == "LWP-Ndmax-channels":
        max_value_threshold = 2000
        min_value_threshold = 2
        filtered_df = df.loc[
            (df['lwp'] > min_value_threshold) &
            (df['lwp'] < max_value_threshold) &
            (df['Nd_max'] > min_value_threshold) &
            (df['Nd_max'] < max_value_threshold)
            ].copy()
        
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
        
    return filtered_df, max_value_threshold



def old_obtain_dataset():    
    # --------------------- ICON --------------------- 
    if variables_names == "Re-COT" or variables_names == "Re-COT-channels":
        variable = ["Reff_cloud_top", "cot"]
        path_icon_timesteps = "/work/bb1036/rttov_share/dataset_ICON/icon_reff_cot_all_data.nc"

    elif variables_names == "LWP-Ndmax" or variables_names == "LWP-Ndmax-channels":
        variable = ["Nd_max", "lwp"]
        # if hr == "-":
        path_icon_timesteps = "/work/bb1036/rttov_share/dataset_ICON/icon_germany_all_data.nc"

    icon_ds = xr.open_dataset(path_icon_timesteps)
    icon_ds = icon_ds[variable]

    if icon_ds['time'].dtype == np.dtype('datetime64[ns]'):
        time_data = icon_ds['time'].values
        print("Formatted datetimes")
    elif icon_ds['time'].dtype == np.dtype('O'):  # 'O' is the dtype for Python objects
        print("Time is object type, converting...")
        icon_ds['time'] = pd.to_datetime(icon_ds['time'].values)
        time_data = icon_ds['time'].values
    else:
        print("Check variable type time ...")

    icon_ds = icon_ds.drop_sel(time=['2013-04-24T17:00', '2013-04-25T17:00'])

    if variables_names == "Re-COT" or variables_names == "Re-COT-channels":
        icon_ds = icon_ds.drop_sel(time=['2013-04-25T06:00'])

    if date != "only_clouds_3days" and hr == "-":
        specific_date = date
        icon_ds = icon_ds.sel(time=slice(f'2013-{specific_date}T00:00', f'2013-{specific_date}T23:59'))
    elif date != "only_clouds_3days" and hr != "-":
        icon_ds = icon_ds.sel(time=[f'2013-05-02T{hr}:00'], method='nearest')

    df = icon_ds.to_dataframe()
    num_rows = len(df)

    # ---------------------------------only for test------------------------------------------------
    # Get the column names
    col1_name = df.columns[0]
    col2_name = df.columns[1]
    # Count zeros in each column
    col1_zeros = (df[col1_name] == 0).sum()
    col2_zeros = (df[col2_name] == 0).sum()
    # Count rows where both columns have zeros
    both_zeros = ((df[col1_name] == 0) & (df[col2_name] == 0)).sum()
    # Count rows where only one column has zero
    one_zero_col1 = ((df[col1_name] == 0) & (df[col2_name] != 0)).sum()
    one_zero_col2 = ((df[col1_name] != 0) & (df[col2_name] == 0)).sum()

    print(f"Number of rows: {num_rows}")
    print(f"Column '{col1_name}' equal to 0: {col1_zeros}")
    print(f"Column '{col2_name}' equal to 0: {col2_zeros}")
    print(f"Both columns equal to 0: {both_zeros}")
    print(f"Column '{col1_name}' is 0 and '{col2_name}' is not 0: {one_zero_col1}")
    print(f"Column '{col1_name}' is not 0 and '{col2_name}' is 0: {one_zero_col2}")
    # ---------------------------------------------------------------------------------

    # --------------------- Ref_Rad --------------------- 
    if variables_names == "LWP-Ndmax-channels" or variables_names == "Re-COT-channels":
        # if hr == "-":
        path_raf_ref_timesteps = "/work/bb1036/rttov_share/dataset_SPECTRAL/S3COM_OP_all_data_rad_ref.nc"
        ref_rad_ds = xr.open_dataset(path_raf_ref_timesteps)
        simulation_ds = ref_rad_ds[['ref_total', "rad_total"]]
        ref_rad_ds = simulation_ds.drop_sel(time=['2013-04-24T17:00', '2013-04-25T17:00'])
        ref_rad_ds['time'] = pd.to_datetime(ref_rad_ds['time'])

        if date != "only_clouds_3days" and hr == "-":
            specific_date = date
            ref_rad_ds = ref_rad_ds.sel(time=slice(f'2013-{specific_date}T00:00', f'2013-{specific_date}T23:59'))
        elif date != "only_clouds_3days" and hr != "-":
            ref_rad_ds = ref_rad_ds.sel(time=[f'2013-05-02T{hr}:00'], method='nearest')

        ref_rad_ds = get_xarray_rad_ref(simulation_ds=ref_rad_ds,
                                        data_type="ref_rad_total")
        df_labels = ref_rad_ds.to_dataframe().unstack(level=['chan'])
        df_labels.columns = df_labels.columns.droplevel(0)

        print(f"All data input size {len(df)}, output {len(df_labels)}")

        df = confirm_indices_and_join(df, df_labels)

    print(f"Number of rows with all: {len(df)}")
    df_only_clouds = df[(df.iloc[:, :2] != 0).all(axis=1)]
    print(f"Number of rows with clouds: {len(df_only_clouds)}")
    if threshold == "filter":
        df, max_value_threshold = filter_datafraframe(df=df_only_clouds, variables_names=variables_names)
    else:
        df = df_only_clouds.copy()

    return df


def main():
    "Checking the correlation in only data with clouds"

    parser = argparse.ArgumentParser(description='Correlation ICON variables')
    arg = parser.add_argument
    arg('--date', type=str, help='The date "04-24", "only_clouds_3days"')
    arg('--results_output_path', type=str, default="/work/bb1036/b381362/output/results_sensitivity/")
    arg('--hr', type=str, help='The hour 10')
    arg('--variables_names', type=str, default="LWP-Ndmax", help='"Re-COT" List of the variables to use.')
    arg('--type_analysis', type=str, default="correlation", help=' type of analysis.')
    arg('--only_2variables', type=str, default="True",
        help='this is used when we do correlation of channels and re, cot or LWP, Nd to have a better visualization.')
    arg('--threshold', type=str, default="nofilter", help='filter, nofilter')
    arg('--path_dataframes_scaler', type=str, default="/work/bb1036/b381362/", help='path where is the dataset save as dataframes and scaler')


    args = parser.parse_args()
    results_output_path = args.results_output_path
    date = args.date
    variables_names = args.variables_names
    type_analysis = args.type_analysis
    hr = args.hr
    only_2variables = args.only_2variables
    threshold = args.threshold
    path_dataframes_scaler = args.path_dataframes_scaler

 
    max_value_threshold = 2000 # for histogramas

    
    df_icon_ref_all = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_ref_rad_clouds_filtered')
    
    df = df_icon_ref_all.drop(columns=['time'])  


    df = change_names_features(variables_names=variables_names, data_df=df)
    print(f"Number of rows to work: {len(df)}")



    
    if type_analysis == "correlation":
        if only_2variables == "True":
            extra_name = "_only_2variables"
        else:
            extra_name = ""

        # Save Spearman Correlation Heatmap
        spearman_output_path = f'{results_output_path}/spearman_correlation_{variables_names}_{date}-{hr}{extra_name}_{threshold}'
        spearman_corr = spearman_correlation_and_save(df, spearman_output_path, only_2variables=only_2variables)
        print(f"Spearman Correlation with {variables_names} ({date}):\n", spearman_corr)

        df_spectral = df.iloc[:, 2:]
        spearman_output_path = f'{results_output_path}/spearman_spectral_{variables_names}_{date}{hr}{threshold}'
        spearman_corr = spearman_correlation_and_save(df=df_spectral, output_path=spearman_output_path, only_2variables="False")

    elif type_analysis == "mutual_information":
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df)  

        
        scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)


        
        X = df.iloc[:, 2:]
        Y = df.iloc[:, :2]
        
        name_output_path = f'{results_output_path}/mutual_information_target_{variables_names}_{date}-{hr}_{threshold}'
        plot_mutual_information_heatmap(X=X, Y=Y, figsize=(3.3, 8), output_path=name_output_path, cmap='Blues')

    elif type_analysis == "histogram":

        df_filtered, max_value_threshold = filter_datafraframe(df=df_only_clouds, variables_names=variables_names)
        df = df_only_clouds.iloc[:,:2].copy()
        name_file = f'{results_output_path}/histograms_of_{variables_names}_part1.png'
        histograms_dataframe(df=df, name_file=name_file)
        df = df_only_clouds.iloc[:,2:].copy()
        name_file = f'{results_output_path}/histograms_of_{variables_names}_part2.png'
        histograms_dataframe(df=df, name_file=name_file)
        
        print(f"Filtered number of samples {len(df_filtered)}, range: {min_value_threshold}-{max_value_threshold})")
        df = df_filtered.iloc[:,:2].copy()
        name_file = f'{results_output_path}/histograms_of_{variables_names}_part1_filter{min_value_threshold}-{max_value_threshold}.png'
        histograms_dataframe(df=df, name_file=name_file)
        df = df_filtered.iloc[:,2:].copy()
        name_file = f'{results_output_path}/histograms_of_{variables_names}_part2_filter{min_value_threshold}-{max_value_threshold}.png'
        histograms_dataframe(df=df, name_file=name_file)


    elif type_analysis == "jointplot":
         # --------------------- ICON --------------------- 
        if variables_names == "Re-COT" or variables_names == "Re-COT-channels":
            variable = ["Reff_cloud_top", "cot"]
    
        elif variables_names == "LWP-Ndmax" or variables_names == "LWP-Ndmax-channels":
            variable = ["Nd_max", "lwp"]

        
        if variables_names == "Re-COT" or variables_names == "LWP-Ndmax":
            fig, ax = plot_joint_histogram(df_only_clouds, variable[0], variable[1], figure_size=(6, 6))
            fig.savefig(f'{results_output_path}/jointplot_{variables_names}_{date}.png', dpi=300)


        elif variables_names == "LWP-Ndmax-channels" or variables_names == "Re-COT-channels":

            # print("here")
            for y_name in variable:
                for i, name in enumerate(df_labels.columns):
                    fig, ax = plot_joint_histogram(df_only_clouds, name, y_name,
                                                   figure_size=(6, 6))  # Pass the axis to the function
                    fig.savefig(f'{results_output_path}/jointplot_{y_name}_{date}_{i}.png')
                    plt.close(fig)

            nimg = len(df_labels.columns)
            ncol = 3
            nrow = 4
            grid_size = (ncol, nrow)

            for y_name in variable:
                for n in range(int(nimg / (ncol * nrow))):
                    file_list = [f'{results_output_path}/jointplot_{y_name}_{date}_{i}.png' for i in
                                 range(n * ncol * nrow, (n + 1) * (ncol * nrow))]
                    output_shap_png_file = f'{results_output_path}/jointplot_{variables_names}_{date}part{n}.png'
                    combine_images_grid(file_list=file_list,
                                        grid_size=grid_size,
                                        output_path=output_shap_png_file)

                    # Delete the files in file_list after concatenation
                    for file_path in file_list:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")

        
if __name__ == '__main__':
    main()
