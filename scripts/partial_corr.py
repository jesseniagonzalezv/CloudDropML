
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
import math
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import argparse


from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import linregress
from scipy.stats import spearmanr, rankdata


def save_correlation_plot(corr_matrix, title, output_path, only_2variables):
    """
    This function saves a heatmap plot of the correlation matrix to a PNG file.

    Parameters:
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
            cbar=False,  
            annot_kws={"size": 13} 
        )
    else:
        if only_2variables == "True":
            titles_font_size = 17
            subtitles_font_size = 16
            labels_font_size = 15 # 16
            axes_font_size = 14

            plt.figure(figsize=(3, 13)) # 3
            sns.heatmap(
                corr_matrix.iloc[:, :2],
                annot=True,
                fmt='.1f',
                cmap='RdBu',
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                cbar=False,
                annot_kws={"size": 12}  
            )

        else:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(
                corr_matrix,
                cmap='RdBu',
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar=True,
                mask=mask,
                cbar_kws={'orientation': 'vertical', 'shrink': 0.8},
            )

    plt.title(title, fontsize=titles_font_size)  
    plt.xticks(fontsize=labels_font_size)  
    plt.yticks(fontsize=labels_font_size)  
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
    save_correlation_plot(corr_matrix, 'Pearson Correlation', output_path, only_2variables)
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
    save_correlation_plot(corr_matrix, 'Spearman Correlation', output_path, only_2variables)
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
    - path_data: str, the directory where the netCDF files are stored.
    - prefix: str, the prefix of the file to identify the type of data (e.g., 'df_icon_pca_train').
    - fold_num: int, the fold number for which the data should be loaded.

    Returns:
    - df: pandas.DataFrame, the DataFrame loaded from the netCDF file.
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

        # Store units as separate attributes
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
        index_channel_relate_clouds = [0 ,1 ,2 ,3 ,4 ,5 ,6 ,19 ,20 ,21 ,22 ,28 ,30 ,31 ,32 ,33 ,34 ,35]
        channel_relate_clouds = name_channels[index_channel_relate_clouds]
        combined_labels = channel_relate_clouds
        new_ds.attrs['chan_index'] = np.array(index_channel_relate_clouds) + 1
        new_ds = new_ds.isel(chan=index_channel_relate_clouds)
        # -------------------------------------------------------

        new_ds['chan'] = combined_labels

    return new_ds


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def get_joint_histgram(nx, ny, xmin, xmax, ymin, ymax, x, y):
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
 
    # Ensure X and Y are DataFrames
    if not isinstance(X, pd.DataFrame) or not isinstance(Y, pd.DataFrame):
        raise ValueError("X and Y must be pandas DataFrames")

    # Initialize an empty matrix to store MI values
    mi_matrix = np.zeros((X.shape[1], Y.shape[1]))

    # Calculate MI for each input-output pair
    for i, input_name in enumerate(X.columns):  # Iterate over input features
        for j, output_name in enumerate(Y.columns):  # Iterate over output variables
            mi_matrix[i, j] = mutual_info_regression(X[[input_name]], Y[output_name])

    # Convert to a DataFrame for better plotting
    mi_df = pd.DataFrame(mi_matrix, index=X.columns, columns=Y.columns)

    titles_font_size = 17
    subtitles_font_size = 16
    labels_font_size = 15 # 16
    axes_font_size = 14

    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(mi_df, annot=True, fmt='.1f', cmap=cmap, cbar=True)

    plt.title('Mutual Information', fontsize=titles_font_size)  
    plt.xticks(fontsize=labels_font_size)  
    plt.yticks(fontsize=labels_font_size)  
    # plt.tight_layout()
    plt.tight_layout(pad=0.5) 
    plt.savefig(output_path, bbox_inches='tight')  
    plt.close()


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
        print \
            (" ---------------------------------------------------------------------------------------------------------------- ")
        print("Indices values are different. Cannot join the DataFrames.")
        return None
    else:
        print \
            (" ---------------------------------------------------------------------------------------------------------------- ")

        print("Indices match in values. Proceeding with join.")
        df_combined = df.join(df_labels, how='inner')  
    
    return df_combined


def partial_mutual_information(x, y, z):
    """
    Compute Partial Mutual Information (PMI) using Scikit-learn.
    """
    
    # Reshape data
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    yz = np.column_stack((y, z)) 

    mi_xy = mutual_info_regression(x, y)[0]      # I(X;Y)
    mi_xz = mutual_info_regression(x, z)[0]      # I(X;Z)
    mi_yz = mutual_info_regression(y, z)[0]      # I(Y;Z)
#
    pmi = mi_xy - (mi_xz + mi_yz)
    
    return pmi


def partial_correlation(X, y, z):
    """
    Calculate the partial correlation between X and y given z.
    
    Args:
        X: array-like, shape (n_samples,)
        y: array-like, shape (n_samples,)
        z: array-like, shape (n_samples,)
    
    Return: 
        Partial correlation coefficient
    """
    
    X_resid = X - np.polyfit(z, X, 1)[0] * z

    y_resid = y - np.polyfit(z, y, 1)[0] * z

    return pearsonr(X_resid, y_resid)[0]


def spearman_partial_correlation(X, y, z):
    """
    Calculate the Spearman partial correlation between X and y given z.

    Args:
        X: array-like, shape (n_samples,) The first variable.
        y: array-like, shape (n_samples,) The second variable.
        z: array-like, shape (n_samples,) The control variable.
    
    Return: 
        Partial Spearman correlation coefficient
    """
    
    X_ranked = rankdata(X)
    y_ranked = rankdata(y)
    z_ranked = rankdata(z)

    model_X = LinearRegression().fit(z_ranked.reshape(-1, 1), X_ranked)
    X_resid = X_ranked - model_X.predict(z_ranked.reshape(-1, 1))

    model_y = LinearRegression().fit(z_ranked.reshape(-1, 1), y_ranked)
    y_resid = y_ranked - model_y.predict(z_ranked.reshape(-1, 1))

    partial_spearman_corr, _ = spearmanr(X_resid, y_resid)
    
    return partial_spearman_corr


def change_names_features(variables_names, data_df):
    if variables_names == "Re-COT" or variables_names == "Re-COT-channels":
        data_df = data_df.rename(columns={
            'Reff_cloud_top': r'$r_e$',  # LaTeX style for subscript
            'cot': 'COT'  
        })


    elif variables_names == "LWP-Ndmax" or variables_names == "LWP-Ndmax-channels":
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


def old_obtaining_data():
    if variables_names == "Re-COT" or variables_names == "Re-COT-channels":
        variable = ["Reff_cloud_top", "cot"]
        path_icon_timesteps = "/work/bb1036/rttov_share/dataset_ICON/icon_reff_cot_all_data.nc"

    elif variables_names == "LWP-Ndmax" or variables_names == "LWP-Ndmax-channels":
        variable = ["Nd_max", "lwp"]
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

    # Print the results
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
        path_raf_ref_timesteps = "/work/dataset_SPECTRAL/S3COM_OP_all_data_rad_ref.nc"
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


def joint_entropy(data, k=5):
    """Compute joint entropy using k-nearest neighbors."""
    n_samples, n_features = data.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # Use the k-th nearest neighbor distance
    k_distances = distances[:, k]
    volume_unit_ball = np.pi ** (0.5 * n_features) / np.math.gamma(0.5 * n_features + 1)  # Ball volume
    avg_log_dist = np.mean(np.log(k_distances + 1e-10))
    return -avg_log_dist + np.log(volume_unit_ball) + np.log(n_samples - 1)


def partial_mutual_info(X, Y, Z, k=5):
    """
    Compute PMI(X; Y | Z) using joint entropy.
    """
    
    XZ = np.concatenate([X, Z], axis=1)
    YZ = np.concatenate([Y, Z], axis=1)
    XYZ = np.concatenate([X, Y, Z], axis=1)
    
    # Compute entropies
    H_XZ = joint_entropy(XZ, k=k)
    H_YZ = joint_entropy(YZ, k=k)
    H_Z = joint_entropy(Z, k=k)
    H_XYZ = joint_entropy(XYZ, k=k)
    
    # Calculate PMI
    return H_XZ + H_YZ - H_Z - H_XYZ



def main():
    "Checking the correlation in only data with clouds"
    parser = argparse.ArgumentParser(description='Correlation ICON variables')
    arg = parser.add_argument
    arg('--date', type=str, help='The date "04-24", "only_clouds_3days"')
    arg('--results_output_path', type=str, default="/work/bb1036/b381362/output/results_sensitivity/")
    arg('--hr', type=str, help='The hour 10')
    arg('--variables_names', type=str, default="LWP-Ndmax", help='"Re-COT" List of the variables to use.')
    arg('--type_analysis', type=str, default="correlation", help=' type of analysis.')
    arg('--threshold', type=str, default="nofilter", help='filter, nofilter')
    arg('--path_dataframes_scaler', type=str, default="/work/bb1036/b381362/", help='path where is the dataset save as dataframes and scaler')

    args = parser.parse_args()
    results_output_path = args.results_output_path
    date = args.date
    variables_names = args.variables_names
    type_analysis = args.type_analysis
    hr = args.hr
    threshold = args.threshold
    path_dataframes_scaler = args.path_dataframes_scaler

    print("Arguments received:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    output_path = f"{results_output_path}/{type_analysis}_{variables_names}_{date}{hr}{threshold}"

    df_icon_ref_all = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_ref_rad_clouds_filtered')
    df = df_icon_ref_all.drop(columns=['time'])  

    data = change_names_features(variables_names=variables_names, data_df=df)
    print(f"Number of rows to work: {len(df)}")

    channels = data.iloc[:, 2:].columns

    partial_A_given_B = {}
    partial_B_given_A = {}


    for channel in channels:
        if type_analysis == "pearson_partial_correlation":
            partial_A_given_B[channel] = partial_correlation(data[channel], data[data.columns[0]], data[data.columns[1]])
            partial_B_given_A[channel] = partial_correlation(data[channel], data[data.columns[1]], data[data.columns[0]])
            title = "Pearson\nPartial\nCorrelation"

        elif type_analysis == "spearman_partial_correlation":
            stats_1 = data.partial_corr(x=channel, y=data.columns[0], covar=data.columns[1], method='spearman')
            stats_2 = data.partial_corr(x=channel, y=data.columns[1], covar=data.columns[0], method='spearman')
            partial_A_given_B[channel] = stats_1['r'].iloc[0]
            partial_B_given_A[channel] = stats_2['r'].iloc[0]
            title = "Spearman\nPartial\nCorrelation"

        elif type_analysis == "partial_mutual_information":
            X = data[[channel]].values  
            Y = data[[data.columns[0]]].values 
            Z = data[[data.columns[1]]].values 
            
            # Compute PMI
            partial_A_given_B[channel] = partial_mutual_info(X, Y, Z)
            partial_B_given_A[channel] = partial_mutual_info(X, Z, Y)
            title = "Partial\nMutual\nInformation"

    partial_df = pd.DataFrame({
        'Channel': channels,
        f'{data.columns[0]} | {data.columns[1]}': [partial_A_given_B[channel] for channel in channels],
        f'{data.columns[1]} | {data.columns[0]}': [partial_B_given_A[channel] for channel in channels]
    })

    partial_df.set_index('Channel', inplace=True)

    print(partial_df)

    titles_font_size = 17
    subtitles_font_size = 16
    labels_font_size = 15 # 16
    axes_font_size = 14

    if type_analysis == "partial_mutual_information":
        plt.figure(figsize=(3.3, 8.7)) # 3

        sns.heatmap(
            partial_df,
            annot=True,
            # cmap='RdBu',
            linecolor='black',
            cmap='Blues',
            # vmin=-1,
            # vmax=1,
            fmt='.1f',
            # square=True,
            linewidths=0.7,
            cbar=True,
            annot_kws={"size": 13}
        )
    else:
        plt.figure(figsize=(3, 8)) # 3

        sns.heatmap(
            partial_df,
            annot=True,
            cmap='RdBu',
            vmin=-1,
            vmax=1,
            fmt='.1f',
            # square=True,
            linewidths=0.5,
            cbar=False,
            linecolor='black',
            annot_kws={"size": 13}
        )

    plt.title(title, fontsize=titles_font_size)
    plt.ylabel('')  
    plt.xticks(fontsize=labels_font_size, rotation=90)  
    plt.yticks(fontsize=labels_font_size)  
    plt.tight_layout(pad=0.5)  
    plt.savefig(f"{output_path}.png", bbox_inches='tight')  
    plt.show()

    partial_df.round(3)
    partial_df.to_csv(f'{output_path}.csv', index=True)

if __name__ == '__main__':
    main()

