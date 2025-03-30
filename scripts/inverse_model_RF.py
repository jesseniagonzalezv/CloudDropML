import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import os
import joblib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.functions import load_dataframe_from_netcdf, select_channels
from utils.functions import metric_calculation_all_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator
# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def metric_calculation(x, gt, model_ml, data_name, verbose=0):
    """
    Metric to calculate the metric for the multioutput data
    x  dataframe
    gt dataframe
    data_name: training, testing
    pred xarray
    """
    """
    Computes various metrics for model predictions against ground truth values.

    This function evaluates the predictions of a given model against the provided ground truth values. 
    It calculates metrics such as R2 score, MAE, MSE, and RMSE. It supports both Keras models and scikit-learn 
    models, determined based on the type of the input model.

    Args:
        x (pandas.DataFrame): Input features for prediction. 
        gt (pandas.DataFrame): Ground truth values for comparison.
        model_ml (Scikit-learn estimator): The machine learning model used for prediction.
        data_name (str): A descriptor of the data being evaluated (e.g., "training", "testing").
        verbose (int, optional): Verbosity mode. Default is 0, meaning silent. For Keras models, 1 indicates progress bar.

    Returns:
        dict: A dictionary containing the computed metrics for all the outputs.
        numpy.ndarray: Model's predictions for the given input.

    Notes:
        - The function automatically detects if the model is a model or not and uses the appropriate prediction method.
        # ???????????????????????????????????????????????????????????????????????
    """
    if isinstance(model_ml, Sequential):  
        pred = model_ml.predict(x, verbose=verbose)
    else:  
        pred = model_ml.predict(x)

    gt = np.array(gt)
    pred = np.array(pred)
    # Convertir a DataFrames de pandas
    df_gt = pd.DataFrame(gt)
    df_pred = pd.DataFrame(pred)

    mask_not_nan = ~df_gt.isnull().any(axis=1) 
    filtered_gt = df_gt[mask_not_nan].values  
    filtered_pred = df_pred[mask_not_nan].values 
    pred[np.isnan(gt)] = np.nan

    metrics = {
        "R-squared (R2)": r2_score(filtered_gt, filtered_pred, multioutput='uniform_average'),
        "Mean Absolute Error (MAE)": mean_absolute_error(filtered_gt, filtered_pred, multioutput='uniform_average'),
        "Mean Squared Error (MSE)": mean_squared_error(filtered_gt, filtered_pred, multioutput='uniform_average'),
        "Root Mean Squared Error (RMSE)": np.sqrt(
            mean_squared_error(filtered_gt, filtered_pred, multioutput='uniform_average')),
    }

    pred_nan_with_gt_not_nan = np.isnan(pred) & ~np.isnan(gt)
    if np.any(pred_nan_with_gt_not_nan):
        print("Warning: NaN predictions found where ground truth is not NaN. Check model predictions.")

    print(f"------------------ {data_name} metrics ----------------------")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return metrics, pred


def save_model(model, path_output, name_model):  # scaler_x, pca_x, scaler_y,
    """
    The function saves the model in Joblib format

    Args:
        model (object): Trained machine learning model to be saved.
        path_output (str): Path to the directory where the model and PCA transformer will be saved.
        name_model (str): Name for the model file.

    Returns:
        None: The function doesn't return any value but saves the model to the specified path.

    Notes:
        The model is saved in '.joblib' format
    """

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    if isinstance(model, tf.keras.Model):
        model.save(os.path.join(path_output, name_model + ".h5"))
    elif isinstance(model, BaseEstimator):
        joblib.dump(model, os.path.join(path_output, name_model + ".joblib"))
    else:
        raise ValueError("Unsupported model type!")


def model_config(type_model, x_train, y_train, x_val, y_val):
    """
    Configure a neural network model based on the specified type.

    This function builds a Sequential model based on the type of model specified by 'type_model'.
    Currently, it supports several predefined neural network and random forest configurations

    Args:
        type_model (str): Type of the model to be created.
        # lr (float): Learning rate for the optimizer.
        # n_epochs (int): Number of epochs for training the model.
        x_train (pandas.DataFrame): Training data.
        y_train (pandas.DataFrame): Target values for the training data.
        x_val (pandas.DataFrame): Validation data.
        y_val (pandas.DataFrame): Target values for the validation data.
        # batch_size (int): Batch size for the training data
    Returns:
        model: A model configured based on the provided 'type_model'.

    """
    if type_model == "RF":

        print(" -------------- MODEL RF_type 2 with mse -------------- ")
        
        rf_model = RandomForestRegressor(
            bootstrap=True,
            ccp_alpha=0.0,
            criterion='squared_error',
            max_depth=20,  
            # max_depth=15,  
            max_features=0.8,  
            max_leaf_nodes=None,
            # min_samples_leaf=10, 
            min_samples_leaf=8,  
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=50,  
            # n_estimators=48,  
            n_jobs=-1, 
            oob_score=True,  
            random_state=42,
            verbose=1,
            warm_start=False
        )
        

      
        model = rf_model.fit(x_train, y_train)  

# .fit(x_train, y_train)
        history = []

    elif type_model == "XGBoost":

        model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)
        model.fit(x_train, y_train)
        history = []



    elif type_model == "RF3":

        print(" -------------- MODEL RF_type 3 with mse -------------- ")
        rf_model = RandomForestRegressor(
            n_estimators=120, 
            max_depth=10,  
            min_samples_split=5, 
            min_samples_leaf=4,  
            random_state=42, 
            n_jobs=32)

        model = rf_model.fit(x_train, y_train)
        history = []

    return history, model


def print_configuration(args):
    configuration = f"""
    Configuration:
    - Model Type: {args.type_model}
    - Learning Rate: {args.lr}
    - Number of Epochs: {args.n_epochs}
    - Batch size: {args.batch_size}
    - Channel Number List: {args.channel_number_list}
    - Fold Number: {args.fold_num}
    - Dataser Path: {args.path_dataframes_pca_scaler}
    - Output Path: {args.path_models}
    """
    print(configuration)


# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- MAIN CODE --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--type-model', type=str, default='NN', help='select the model NN, RF')
    arg('--results_output_path', type=str, default="/work/bb1036/b381362/output/",
        help='path of the folder to save the outputs')
    arg('--variables_names', type=str, help='"Re-COT" List of the variables to use.')
    arg('--variable_target', type=str, help='"Re, COT, LWP, Ndmax" List of the variables to use.')
    arg('--fold-num', type=int, default=1, help='n k-fold to run')
    arg('--path_dataframes_scaler', type=str, default="/work/bb1036/b381362/",
        help='path where is the dataset save as dataframes and scaler')
    arg('--all_data', type=str, default="all_features", help='number of flods to split the dataset')

    args = parser.parse_args()

    type_model = args.type_model
    variables_names = args.variables_names
    variable_target = args.variable_target
    results_output_path = args.results_output_path
    fold_num = args.fold_num
    path_dataframes_scaler = args.path_dataframes_scaler
    all_data = args.all_data

    print(" ========================================== ")
    print("Arguments received:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    name_saving_files = f"{type_model}_{variable_target}_k_fold_{fold_num}"



    channel_relate_clouds = select_channels(all_data=all_data,
                                            variable_target=variable_target,
                                            type_model=type_model)

    print(f" ==================== {all_data} used  ==================== ")

    # --------------------------------------- load dataset and scalers ----------------------------
    try:
        scaler_y = pickle.load(open(f'{path_dataframes_scaler}/scaler_{variables_names}_k_fold_{fold_num}.npy',
                                    'rb'))  # , allow_pickle=True)
        print("Scaler_y loaded.")
    except FileNotFoundError:
        print("The file scaler_y was not found. Continue with the ejecution using None as scaler.")
        scaler_y = None

    # with open(f"{path_dataframes_scaler}/times_data_k_fold_{fold_num}.pkl", 'rb') as f:
    with open(f"{path_dataframes_scaler}/val_data_k_fold_{fold_num}.pkl", 'rb') as f:
        times_data = pickle.load(f)

    df_icon_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_train', fold_num)
    df_icon_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    # df_icon_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_test', fold_num)
    df_icon_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    df_ref_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_train', fold_num)
    df_ref_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)
    # df_ref_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_test', fold_num)
    df_ref_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)

    loaded_index_all_test = np.load(f'{path_dataframes_scaler}/index_only_clouds_all_val_{fold_num}.npy',
                                    allow_pickle=True)

    if isinstance(loaded_index_all_test[0], tuple):  
        loaded_index_all_test = pd.MultiIndex.from_tuples(loaded_index_all_test)

    df_icon_test.index = loaded_index_all_test
    df_ref_test.index = loaded_index_all_test

    # -------------- dividing features and labels ------- train - val
    y_train = df_icon_train[[variable_target]]
    X_train = df_ref_train.loc[:, channel_relate_clouds]
    y_val = df_icon_val[[variable_target]]
    X_val = df_ref_val.loc[:, channel_relate_clouds]
    y_test = df_icon_test[[variable_target]]
    X_test = df_ref_test.loc[:, channel_relate_clouds]

    print(
        f" -------------------- shape x_train {np.shape(X_train)} and y_traina {np.shape(y_train)} after squeeze y {np.shape(y_train.squeeze())}")

    history, model = model_config(type_model=f"{type_model}",
                                  x_train=X_train,
                                  y_train=y_train.squeeze(),
                                  x_val=X_val,
                                  y_val=y_val.squeeze(),
                                  )

    save_model(model=model,
               path_output=results_output_path,
               name_model=f"{all_data}_{name_saving_files}")

    metrics_train, _ = metric_calculation(x=X_train,
                                          gt=y_train.squeeze(),
                                          model_ml=model,
                                          data_name="training")

    metrics_val, _ = metric_calculation(x=X_val,
                                        gt=y_val.squeeze(),
                                        model_ml=model,
                                        data_name="validation")

    metrics_test, test_predictions_ref_rad = metric_calculation(x=X_test,
                                                                gt=y_test.squeeze(),
                                                                model_ml=model,
                                                                data_name="testing")

    # ========== Create DataFrame ==========
    # print("\n----------------metrics with the log target -------")
    print("\n----------------metrics with the scaled target -------")
    df_metrics = pd.DataFrame({
        f"Train": metrics_train,
        f"Validation": metrics_val,
        f"Test": metrics_test,
    }).T

    df_metrics = df_metrics.round(4)
    df_metrics.to_csv(f"{results_output_path}/{all_data}_table_metrics_{name_saving_files}_with_scaler.csv")
    print(df_metrics)

    # --------------- invert scaled target --------------
    target_index = df_icon_train.columns.get_loc(variable_target)

    if isinstance(scaler_y, StandardScaler):
        target_means = scaler_y.mean_[target_index]
        target_vars = scaler_y.var_[target_index]

        # ------------------------ Training metric ---------------------
    pred = model.predict(X_train)
    df_unstandardized_all_preds_train = pd.DataFrame(pred * np.sqrt(target_vars) + target_means)
    df_unstandardized_all_targets_train = pd.DataFrame(y_train * np.sqrt(target_vars) + target_means)

    metrics_train = metric_calculation_all_data(outputs=df_unstandardized_all_preds_train,
                                                targets=df_unstandardized_all_targets_train)

    # --------------------- validation metric -------------------------
    pred = model.predict(X_val)
    df_unstandardized_all_preds_val = pd.DataFrame(pred * np.sqrt(target_vars) + target_means)
    df_unstandardized_all_targets_val = pd.DataFrame(y_val * np.sqrt(target_vars) + target_means)
    metrics_val = metric_calculation_all_data(outputs=df_unstandardized_all_preds_val,
                                              targets=df_unstandardized_all_targets_val)

    # --------------------- testing metric -------------------------
    pred = model.predict(X_test)
    df_unstandardized_all_preds_test = pd.DataFrame(pred * np.sqrt(target_vars) + target_means,
                                                    index=loaded_index_all_test, columns=y_test.columns)
    df_unstandardized_all_targets_test = pd.DataFrame(y_test * np.sqrt(target_vars) + target_means,
                                                      index=loaded_index_all_test)

    print(
        f" --------------------check nans {df_unstandardized_all_preds_test.isnull().sum()}, {df_unstandardized_all_targets_test.isnull().sum()}")

    metrics_test = metric_calculation_all_data(outputs=df_unstandardized_all_preds_test,
                                               targets=df_unstandardized_all_targets_test)

    # ========== Create DataFrame  metrics  =======================
    # print("\n----------------metrics without the log target -------")
    print("\n----------------metrics without scaled target -------")
    df_metrics = pd.DataFrame({
        f"Train k={fold_num}:": metrics_train,
        f"Validation": metrics_val,
        f"Test": metrics_test,
    }).T

    df_metrics = df_metrics.round(4)
    df_metrics.to_csv(f"{results_output_path}/{all_data}_table_metrics_{name_saving_files}_without_scaler.csv")
    print(df_metrics)

    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------

    # print("\n----------------metrics sample test without log -------")
    print("\n----------------metrics each sample test -------")
    for i, n_time in enumerate(times_data['val_times']):
        # for i, n_time in enumerate(times_data['test_times']):
        df_target_ntime = df_unstandardized_all_targets_test.loc[
            df_unstandardized_all_targets_test.index.get_level_values(0) == n_time]
        df_pred_ntime = df_unstandardized_all_preds_test.loc[
            df_unstandardized_all_preds_test.index.get_level_values(0) == n_time]

        metrics = metric_calculation_all_data(outputs=df_pred_ntime,
                                              targets=df_target_ntime)

        print(f"------------------ {n_time} metrics ----------------------")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    importances = model.feature_importances_
    feature_names = X_train.columns

    fold_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Fold": fold_num,
        "Importance": importances
    })

    fold_importances_df.to_csv(f"{results_output_path}/{all_data}_features_importances_{name_saving_files}.csv",
                               index=False)


if __name__ == '__main__':
    main()
