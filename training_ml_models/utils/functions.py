
import xarray as xr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS DATA--------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

def convert_fractional_day_to_time(time_varible: float) -> (int, int, int, int, int, int):
    """
    Convert time format
    
    Args::
        time_varible (float64) -- Given a time value as a fractional day in the format YYYYMMDD.fractionalday (ds.time.values)
    
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

    # formatted_datetimes = time_data.strftime('%Y-%m-%dT%H:%M')
    formatted_datetimes = time_data.strftime('%Y%m%dT%H%M')

    # print(formatted_datetimes)

    return formatted_datetimes


def select_channels(all_data, variable_target, type_model):
    '''
    Function to select the channels to be use all channels or the most important channels given the model and the target
    
    Args:
        all_data (str): all_features or most_importances
        variable_target (str): lwp or Nd_max
        type_model (str): NN, RF, XGBoost
    
    Return
        channel_relate_clouds (array): names of the channels to be use 
    '''
    
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
        index_channel_relate_clouds = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 28, 30, 31, 32, 33, 34, 35]
        
    else:
        # ------------------- MIF  -------------------
        if variable_target == "lwp":
            if type_model == "NN":
                index_channel_relate_clouds = [0, 4, 5]  
            elif type_model == "RF":
                index_channel_relate_clouds = [2, 3, 5, 6] 
            elif type_model == "XGBoost":
                index_channel_relate_clouds = [0, 2, 4, 5] 


        elif variable_target == "Nd_max":
            if type_model == "NN":
                index_channel_relate_clouds = [2, 4, 5, 19, 22, 28, 32] 
            elif type_model == "RF":
                index_channel_relate_clouds = [2, 5, 6, 28, 30, 34, 35] 
            elif type_model == "XGBoost":
                index_channel_relate_clouds = [1, 2, 5, 6, 19, 22, 28, 30] 

    channel_relate_clouds = name_channels[index_channel_relate_clouds]
    print(f" ==================== Channels used: {channel_relate_clouds}  ==================== ")

    return channel_relate_clouds


def load_dataframe_from_netcdf(path_data, prefix, fold_num):
    """
    Loads a DataFrame from a netCDF file based on the provided path, prefix, and fold number.

    Args:
        path_data: str, the directory where the netCDF files are stored.
        prefix: str, the prefix of the file to identify the type of data (e.g., 'df_icon_pca_train').
        fold_num: int, the fold number for which the data should be loaded.

    Returns:
        df: pandas.DataFrame, the DataFrame loaded from the netCDF file.
    """
    
    filename = f'{path_data}/{prefix}_fold_{fold_num}.nc'
    xr_dataset = xr.open_dataset(filename)
    #  the index is already restarte in 0
    df = xr_dataset.to_dataframe()
    # .reset_index()
    # idf.drop(["index"],axis=1)#.reset_index()
    return df


# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS METRICS--------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def smape(actual, prediction):
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE)
    """
    return 100 * np.mean(2 * np.abs(prediction - actual) / (np.abs(actual) + np.abs(prediction)))


def metric_calculation_all_data(outputs, targets):
    """
    Evaluate R-squared (R2) and Root Mean Squared Error (RMSE), other metrics for each output of the model and their uniform averages.

    Args:
        outputs (dataframe): The model's predictions, expected to be a dataframe of shape (n_samples, n_outputs).
        targets (dataframe): The ground truths (actual values), expected to be a dataframe of shape (n_samples, n_outputs).

    Returns:
        metrics (dict): A dictionary containing RMSE and R2 for all output in averaged and optionally for each channel.
    """

    targets_array = targets.to_numpy()
    outputs_array = outputs.to_numpy()
    
    mse_avg = mean_squared_error(targets_array, outputs_array, multioutput='uniform_average')
    r2_avg = r2_score(targets_array, outputs_array, multioutput='uniform_average')
    mae_avg = mean_absolute_error(targets_array, outputs_array, multioutput='uniform_average')
    smape_avg = smape(targets_array, outputs_array)

    metrics = {
        "R-squared (R2)": r2_avg,
        "Mean Squared Error (MSE)": mse_avg,
        "Mean Absolute Error (MAE)": mae_avg,
        "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(targets, outputs, multioutput='uniform_average')),
        "Symmetric Mean Absolute Percentage Error (SMAPE)": smape_avg,

    }

    return metrics


# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS PLOTS--------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

def plot_loss(train_losses, val_losses, path_output, type_model):
    f = plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    figure_name = f'{path_output}/history_{type_model}.png'
    f.tight_layout()
    f.savefig(figure_name, dpi=60)
    # plt.show()
    plt.close()
