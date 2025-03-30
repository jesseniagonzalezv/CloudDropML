import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import torch.nn as nn
import sys
import os

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.functions import select_channels
from utils.functions import load_dataframe_from_netcdf
from inverse_model_NN import MyModel, train_model, evaluate_model


def evaluate_model_by_features_RFE(model, variable_target, df_x_train, df_y_train, 
                                   df_x_val, df_y_val, figure_name, model_name="Model"):
    """
    Evaluates model performance using RFE for feature selection.
    
    For each k in 1 to total number of features, RFE is used to select the top k features.
    The model is then trained on these features and performance metrics (R² and MAE)
    are computed on the validation set.
    
    Args:
      model: The base estimator (e.g., RandomForestRegressor) used for both RFE and final training.
      variable_target: Name of the target variable (for plot titles).
      df_x_train: Training features (DataFrame).
      df_y_train: Training target (DataFrame).
      df_x_val: Validation features (DataFrame).
      df_y_val: Validation target (DataFrame).
      figure_name: Filename to save the performance plot.
      model_name: Name of the model (for display purposes).
    
    Returns:
      results: List of dictionaries with keys 'num_features', 'r2', and 'mae'.
    """
    
    n_total_features = df_x_train.shape[1]
    feature_counts = []
    r2_scores = []
    mae_scores = []
    results = []
    
    for k in range(1, n_total_features + 1):
        estimator = clone(model)
        rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
        rfe.fit(df_x_train, df_y_train.values.ravel())
        top_features = df_x_train.columns[rfe.support_].tolist()
        print(f"{k}total_features:{top_features}==============================================")
        new_model = clone(model)
        new_model.fit(df_x_train[top_features], df_y_train.values.ravel())
        y_val_pred = new_model.predict(df_x_val[top_features])
        
        r2 = r2_score(df_y_val, y_val_pred)
        mae = mean_absolute_error(df_y_val, y_val_pred)
        
        feature_counts.append(k)
        r2_scores.append(r2)
        mae_scores.append(mae)
        results.append({'num_features': k, 'r2': r2, 'mae': mae})
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('R²', color=color)
    ax1.plot(feature_counts, r2_scores, color=color, marker='o', label='R²')
    ax1.set_xticks(feature_counts)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f'{model_name} Performance vs. Number of Features for {variable_target}')
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('MAE', color=color)
    ax2.plot(feature_counts, mae_scores, color=color, marker='s', label='MAE')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.grid(True)
    
    fig.tight_layout()
    fig.savefig(figure_name, dpi=60)
    plt.close()
    
    return results


def rank_features_with_lasso(df_x_train, df_y_train, alpha=0.01, max_iter=10000, tol=1e-4):
    lasso = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
    lasso.fit(df_x_train, df_y_train.values.ravel())

    coef_df = pd.DataFrame({
        'feature': df_x_train.columns,
        'coef': np.abs(lasso.coef_)
    })
    coef_df.sort_values(by='coef', ascending=False, inplace=True)
    return coef_df


def evaluate_model_by_features(model, variable_target, df_x_train, df_y_train, df_x_val, df_y_val, feature_ranking,
                               figure_name,
                               model_name="Model"):
    n_features = len(feature_ranking)
    feature_counts = []
    r2_scores = []
    mae_scores = []
    results = []

    for k in range(1, n_features + 1):
        top_features = feature_ranking['feature'].iloc[:k].tolist()
        X_train_subset = df_x_train[top_features]
        X_val_subset = df_x_val[top_features]

        model.fit(X_train_subset, df_y_train.values.ravel())

        y_val_pred = model.predict(X_val_subset)
        r2 = r2_score(df_y_val, y_val_pred)
        mae = mean_absolute_error(df_y_val, y_val_pred)

        feature_counts.append(k)
        r2_scores.append(r2)
        mae_scores.append(mae)
        results.append({'num_features': k, 'r2': r2, 'mae': mae})

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('R²', color=color)
    ax1.plot(feature_counts, r2_scores, color=color, marker='o', label='R²')
    ax1.set_xticks(feature_counts)

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f'{model_name} Performance vs. Number of Features for {variable_target}')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('MAE', color=color)
    ax2.plot(feature_counts, mae_scores, color=color, marker='s', label='MAE')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.grid(True)

    fig.tight_layout()
    fig.savefig(figure_name, dpi=60)
    plt.close()

    return results


def evaluate_model_by_features_nn_lasso(n_epochs, lr, batch_size, variable_target, df_x_train, df_y_train, df_x_val,
                                  df_y_val, feature_ranking,
                                  figure_name,
                                  model_name="Model"):
    n_features = len(feature_ranking)
    feature_counts = []
    r2_scores = []
    mae_scores = []
    results = []

    for k in range(1, n_features + 1):
        top_features = feature_ranking['feature'].iloc[:k].tolist()
        X_train_subset = df_x_train[top_features]
        X_val_subset = df_x_val[top_features]

        x_train_np = X_train_subset.to_numpy(dtype=float)
        x_val_np = X_val_subset.to_numpy(dtype=float)
        y_train_np = df_y_train.to_numpy(dtype=float)
        y_val_np = df_y_val.to_numpy(dtype=float)

        x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
        x_val_tensor = torch.tensor(x_val_np, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f" -------------------- Using device: {device} --------------- ")
        model = MyModel(input_size=x_train_np.shape[1],
                        output_size=y_train_np.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss(beta=0.5)
        
        train_losses, val_losses, train_rmses, val_rmses, train_r2s, val_r2s, best_model_state = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=device)

        model.load_state_dict(best_model_state)

        val_loss, val_rmse_avg, val_r2_avg, all_preds_val, all_targets_val = evaluate_model(model, val_loader,
                                                                                            criterion,
                                                                                            device)
        y_val_pred = all_preds_val.cpu().detach().numpy()

        r2 = r2_score(y_val_np, y_val_pred)
        mae = mean_absolute_error(y_val_np, y_val_pred)

        feature_counts.append(k)
        r2_scores.append(r2)
        mae_scores.append(mae)
        results.append({'num_features': k, 'r2': r2, 'mae': mae})

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('R²', color=color)
    ax1.plot(feature_counts, r2_scores, color=color, marker='o', label='R²')
    ax1.set_xticks(feature_counts)

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f'{model_name} Performance vs. Number of Features for {variable_target}')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('MAE', color=color)
    ax2.plot(feature_counts, mae_scores, color=color, marker='s', label='MAE')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True)

    fig.tight_layout()
    # Save the figure
    fig.savefig(figure_name, dpi=60)
    # plt.show()
    plt.close()
    # plt.show()

    return results


def evaluate_model_by_features_nn_with_ranking(n_epochs, lr, batch_size, variable_target, df_x_train, df_y_train, df_x_val,
                                  df_y_val, feature_ranking,
                                  figure_name,
                                  model_name="Model"):
    n_features = len(feature_ranking)
    feature_counts = []
    r2_scores = []
    mae_scores = []
    results = []

    for k in range(1, n_features + 1):
        top_features = feature_ranking[:k]

        X_train_subset = df_x_train[top_features]
        X_val_subset = df_x_val[top_features]

        x_train_np = X_train_subset.to_numpy(dtype=float)
        x_val_np = X_val_subset.to_numpy(dtype=float)
        y_train_np = df_y_train.to_numpy(dtype=float)
        y_val_np = df_y_val.to_numpy(dtype=float)

        x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
        x_val_tensor = torch.tensor(x_val_np, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f" -------------------- Using device: {device} --------------- ")
        model = MyModel(input_size=x_train_np.shape[1],
                        output_size=y_train_np.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss(beta=0.5)
        # criterion = nn.MSELoss()
        
        train_losses, val_losses, train_rmses, val_rmses, train_r2s, val_r2s, best_model_state = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=device)

        model.load_state_dict(best_model_state)

        val_loss, val_rmse_avg, val_r2_avg, all_preds_val, all_targets_val = evaluate_model(model, val_loader,
                                                                                            criterion,
                                                                                            device)
        y_val_pred = all_preds_val.cpu().detach().numpy()

        r2 = r2_score(y_val_np, y_val_pred)
        mae = mean_absolute_error(y_val_np, y_val_pred)

        feature_counts.append(k)
        r2_scores.append(r2)
        mae_scores.append(mae)
        results.append({'num_features': k, 'r2': r2, 'mae': mae})

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('R²', color=color)
    ax1.plot(feature_counts, r2_scores, color=color, marker='o', label='R²')
    ax1.set_xticks(feature_counts)

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f'{model_name} Performance vs. Number of Features for {variable_target}')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('MAE', color=color)
    ax2.plot(feature_counts, mae_scores, color=color, marker='s', label='MAE')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(figure_name, dpi=60)
    plt.close()

    return results



def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--type_model', type=str, default='NN', help='select the model NN, RF')
    arg('--path_dataframes_scaler', type=str, default="/work/bb1036/b381362/",
        help='path where is the dataset save as dataframes and scaler')
    arg('--path_results', type=str, default="/work/bb1036/b381362/output/",
        help='path of the folder to save the outputs')
    arg('--fold-num', type=int, default=1, help='n k-fold to run')
    arg('--lr', type=float, default=1e-3)
    arg('--n-epochs', type=int, default=30)
    arg('--batch-size', type=int, default=64)
    arg('--variable_target', type=str, default="lwp", help='"Re, COT, LWP, Ndmax" List of the variables to use.')
    arg('--csv_ranking_path', type=str, default="/work/bb1036/b381362/output/final_results/", help='"Re, path where the feature importances (median of the different methods) is located (NN model).')
    arg('--type_NN_ranking', type=str, default='mean', help='mean or median')

    args = parser.parse_args()

    type_model = args.type_model
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    fold_num = args.fold_num
    path_dataframes_scaler = args.path_dataframes_scaler
    path_results = args.path_results
    variable_target = args.variable_target
    csv_ranking_path = args.csv_ranking_path
    type_NN_ranking = args.type_NN_ranking

    if type_model == "NN":
        if type_NN_ranking == "mean":
            csv_file_name = f'{csv_ranking_path}/{type_model}_feature_importance_all_features_mean.csv'
        elif type_NN_ranking == "median":
            csv_file_name = f'{csv_ranking_path}/{type_model}_feature_importance_all_features_median.csv'

    figure_name = f'{path_results}/{variable_target}_performance_num_features_{type_model}_fold{fold_num}_ranking.png'
    filename = f"{path_results}/{variable_target}_performance_{type_model}_fold{fold_num}_ranking.csv"

    channel_relate_clouds = select_channels(all_data="all_features",
                                            variable_target=variable_target,
                                            type_model=type_model)

    
    df_icon_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_train', fold_num)
    df_icon_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    df_ref_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_train', fold_num)
    df_ref_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)

    df_y_train = df_icon_train[[variable_target]]
    df_x_train = df_ref_train.loc[:, channel_relate_clouds]
    df_y_val = df_icon_val[[variable_target]]
    df_x_val = df_ref_val.loc[:, channel_relate_clouds]


    if type_model != "NN":
        feature_ranking = rank_features_with_lasso(df_x_train, df_y_train, alpha=0.1)

    if type_model == "RF":
        rf_model = RandomForestRegressor(
            bootstrap=True,
            ccp_alpha=0.0,
            criterion='squared_error',
            max_depth=20,  
            max_features=0.8,  
            max_leaf_nodes=None,
            min_samples_leaf=8,  
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=50,  
            n_jobs=-1,  
            oob_score=True,  
            random_state=42,
            verbose=1,
            warm_start=False
        )

        
        results = evaluate_model_by_features_RFE(model=rf_model,
                                             df_x_train=df_x_train,
                                             df_y_train=df_y_train,
                                             df_x_val=df_x_val,
                                             df_y_val=df_y_val,
                                             variable_target=variable_target,
                                             figure_name=figure_name,
                                             model_name="RF")



    elif type_model == "XGBoost":

        model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)

        results = evaluate_model_by_features_RFE(model=model,
                                             df_x_train=df_x_train,
                                             df_y_train=df_y_train,
                                             df_x_val=df_x_val,
                                             df_y_val=df_y_val,
                                             variable_target=variable_target,
                                             figure_name=figure_name,
                                             model_name="XGBoost")
    elif type_model == "MLP":

        nn_model = MLPRegressor(random_state=42, max_iter=500)

        results = evaluate_model_by_features(model=nn_model,
                                             df_x_train=df_x_train,
                                             df_y_train=df_y_train,
                                             df_x_val=df_x_val,
                                             df_y_val=df_y_val,
                                             feature_ranking=feature_ranking,
                                             variable_target=variable_target,
                                             figure_name=figure_name,
                                             model_name="MLP")

    elif type_model == "NN_lasso":

            
        results = evaluate_model_by_features_nn_lasso(n_epochs=n_epochs,
                                                lr=lr,
                                                batch_size=batch_size,
                                                df_x_train = df_x_train,
                                                df_y_train = df_y_train,
                                                df_x_val = df_x_val,
                                                df_y_val = df_y_val,
                                                feature_ranking = feature_ranking,
                                                variable_target = variable_target,
                                                figure_name = figure_name,
                                                model_name = "NN")


    elif type_model == "NN":
        if variable_target=="lwp":
            variable_target_csv = '$L$'
        elif variable_target=="Nd_max": 
            variable_target_csv = r'$N_{\mathrm{d},\max}$'
        df_importances = pd.read_csv(csv_file_name, index_col=0)[[variable_target_csv]]  
        df_importances = df_importances.sort_values(by=variable_target_csv , ascending=False)
        feature_ranking = df_importances.index.tolist()

        results = evaluate_model_by_features_nn_with_ranking(n_epochs=n_epochs,
                                                            lr=lr,
                                                            batch_size=batch_size,
                                                            df_x_train = df_x_train,
                                                            df_y_train = df_y_train,
                                                            df_x_val = df_x_val,
                                                            df_y_val = df_y_val,
                                                            feature_ranking = feature_ranking,
                                                            variable_target = variable_target,
                                                            figure_name = figure_name,
                                                            model_name = "NN")
    df_results = pd.DataFrame(results)
    df_results['fold'] = fold_num
    df_results['model'] = type_model

    df_results.to_csv(filename, index=False)
    print(f"Saved results to {filename}")



if __name__ == '__main__':
    main()