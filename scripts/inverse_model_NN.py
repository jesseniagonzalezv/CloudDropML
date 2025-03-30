import numpy as np
import pandas as pd
import pickle
import argparse
import copy
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.functions import load_dataframe_from_netcdf, select_channels
from utils.functions import metric_calculation_all_data
from utils.functions import plot_loss

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset



# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def metric_dictionary(rmse_avg, r2_avg, data_name):
    """
     R-squared (R2) and Root Mean Squared Error (RMSE) for each output of the model and their uniform averages.

    Args:
        rmse_avg
        r2_avg
        data_name

    Returns:
    - metrics (dict): A dictionary containing RMSE and R2 for all output in averaged.
    """
    
    metrics = {
        "R-squared (R2)": r2_avg,
        "Root Mean Squared Error (RMSE)": rmse_avg,
    }

    print(f"------------------ {data_name} metrics ----------------------")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return metrics


def compute_rmse_r2(outputs, targets):
    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    rmse_avg = np.sqrt(mean_squared_error(targets, outputs, multioutput='uniform_average'))
    r2_avg = r2_score(targets, outputs, multioutput='uniform_average')
    return rmse_avg, r2_avg



class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            ## nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)




def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    loss = total_loss / len(data_loader)
    rmse_avg, r2_avg = compute_rmse_r2(all_preds, all_targets)

    return loss, rmse_avg, r2_avg, all_preds, all_targets


def train_model(model, train_loader, val_loader, n_epochs, optimizer, criterion, device):
    """
    Train a neural network model.

    Args:
        model: model sequential
        n_epochs (int): Number of epochs for training the model.
        train_loader (dataloader tensor): Training data.
        val_loader (dataloader tensor): Validation data.
        optimizer: torch.optim.Optimizer
        criterion: torch.nn.Module
        device: torch.device
    Returns:
        model: A model configured.
    """

    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []
    train_r2s, val_r2s = [], []

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_train_loss += loss.item()
            all_train_preds.append(outputs)
            all_train_targets.append(targets)

            loss.backward()
            optimizer.step()

        train_loss = total_train_loss / len(train_loader)

        all_train_preds = torch.cat(all_train_preds)
        all_train_targets = torch.cat(all_train_targets)
        train_rmse_avg, train_r2_avg = compute_rmse_r2(all_train_preds, all_train_targets)

        train_losses.append(train_loss)
        train_rmses.append(train_rmse_avg)
        train_r2s.append(train_r2_avg)

        val_loss, val_rmse, val_r2, all_preds, all_targets = evaluate_model(model, val_loader, criterion, device)

        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        val_r2s.append(val_r2)

        print(
            ' --------------------------------------------------------------------------------------------------------')
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f},  Val Loss: {val_loss:.4f}')
        print(f'Train RMSE --  {train_rmse_avg:.4f}')
        print(f'Val RMSE --    {val_rmse:.4f}')
        print(f'Train R2 --    {train_r2_avg:.4f}')
        print(f'Val R2 --      {val_r2:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch + 1}: New best model saved with val_loss: {best_val_loss}")

    return train_losses, val_losses, train_rmses, val_rmses, train_r2s, val_r2s, best_model_state


def print_configuration(args):
    configuration = f"""
    Configuration:
    - Model Type: {args.type_model}
    - Learning Rate: {args.lr}
    - Number of Epochs: {args.n_epochs}
    - Batch size: {args.batch_size}
    - Fold Number: {args.fold_num}
    - Dataser Path: {args.path_dataframes_pca_scaler}
    - Output Path: {args.path_models}
    """
    print(configuration)


class ReweightedLoss(nn.Module):
    def __init__(self, low_weight=2.0, high_weight=2.0, low_threshold=4.37, high_threshold=5.91):
        super(ReweightedLoss, self).__init__()
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.mse_loss = nn.MSELoss(reduction='none')  

    def forward(self, y_pred, y_true):
        base_loss = self.mse_loss(y_pred, y_true)

        low_mask = (y_true < self.low_threshold).float()
        base_loss += self.low_weight * low_mask * base_loss

        high_mask = (y_true > self.high_threshold).float()
        base_loss += self.high_weight * high_mask * base_loss

        return base_loss.mean()


# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- MAIN CODE --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--type-model', type=str, default='NN', help='select the model NN, RF')
    arg('--path_dataframes_scaler', type=str, default="/work/bb1036/b381362/",
        help='path where is the dataset save as dataframes and scaler')
    arg('--path_models', type=str, default="/work/bb1036/b381362/output/",
        help='path of the folder to save the outputs')
    arg('--fold-num', type=int, default=1, help='n k-fold to run')
    arg('--lr', type=float, default=1e-3)
    arg('--n-epochs', type=int, default=30)
    arg('--batch-size', type=int, default=64)
    arg('--variables_names', type=str, default="LWP-Ndmax", help='"Re-COT" List of the variables to use.')
    arg('--variable_target', type=str, default="lwp", help='"Re, COT, LWP, Ndmax" List of the variables to use.')
    arg('--all_data', type=str, default="all_features", help='number of flods to split the dataset')
    arg('--loss_type', type=str, default="MSELoss", help='type Loss')

    args = parser.parse_args()

    type_model = args.type_model
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    fold_num = args.fold_num
    path_dataframes_scaler = args.path_dataframes_scaler
    path_models = args.path_models

    variables_names = args.variables_names
    # threshold = args.threshold
    variable_target = args.variable_target

    all_data = args.all_data
    loss_type = args.loss_type
    # steps = "train"


    print(f" ============  {type_model} ============================== ")
    print("Arguments received:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    name_saving_files = f"{type_model}_{variable_target}_k_fold_{fold_num}"

    channel_relate_clouds = select_channels(all_data=all_data,
                                            variable_target=variable_target,
                                            type_model=type_model)

    print(f" ==================== {all_data} used  ==================== ")


    try:
        scaler_y = pickle.load(open(f'{path_dataframes_scaler}/scaler_{variables_names}_k_fold_{fold_num}.npy',
                                    'rb'))  
        print("Scaler_y loaded.")
    except FileNotFoundError:
        print("The file scaler_x was not found. Continue with the ejecution using None as scaler.")
        scaler_y = None

    with open(f"{path_dataframes_scaler}/val_data_k_fold_{fold_num}.pkl", 'rb') as f:
        times_data = pickle.load(f)

    df_icon_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_train', fold_num)
    df_icon_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    df_icon_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_icon_val', fold_num)
    df_ref_train = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_train', fold_num)
    df_ref_val = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)
    df_ref_test = load_dataframe_from_netcdf(path_dataframes_scaler, 'df_ref_val', fold_num)

    loaded_index_all_test = np.load(f'{path_dataframes_scaler}/index_only_clouds_all_val_{fold_num}.npy',
                                    allow_pickle=True)

    if isinstance(loaded_index_all_test[0], tuple):  
        loaded_index_all_test = pd.MultiIndex.from_tuples(loaded_index_all_test)

    # -------------- dividing features and labels ------- train - val
    y_train_np = df_icon_train[[variable_target]].to_numpy(dtype=float)
    x_train_np = df_ref_train.loc[:, channel_relate_clouds].to_numpy(dtype=float)
    y_val_np = df_icon_val[[variable_target]].to_numpy(dtype=float)
    x_val_np = df_ref_val.loc[:, channel_relate_clouds].to_numpy(dtype=float)
    y_test_np = df_icon_test[[variable_target]].to_numpy(dtype=float)
    x_test_np = df_ref_test.loc[:, channel_relate_clouds].to_numpy(dtype=float)

    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f" -------------------- Using device: {device} --------------- ")
    model = MyModel(input_size=x_train_np.shape[1],
                    output_size=y_train_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if loss_type=="MSELoss":
        criterion = nn.MSELoss()
    elif loss_type=="SmoothL1Loss":
        criterion = nn.SmoothL1Loss(beta=0.5)
    print(f"=================================loss:{loss_type} =================================")
    
    # Train the model
    train_losses, val_losses, train_rmses, val_rmses, train_r2s, val_r2s, best_model_state = train_model(model=model,
                                                                                                         train_loader=train_loader,
                                                                                                         val_loader=val_loader,
                                                                                                         n_epochs=n_epochs,
                                                                                                         optimizer=optimizer,
                                                                                                         criterion=criterion,
                                                                                                         device=device)

    # Save the model
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f'{path_models}/{all_data}_{name_saving_files}.pth')

    # Call the plotting function
    plot_loss(train_losses=train_losses,
              val_losses=val_losses,
              path_output=path_models,
              type_model=f"{name_saving_files}")

    # ============================= Evaluate your model ==========
    # Calculate metrics all data train, val, tes
    # ============================================================
    train_loss, train_rmse_avg, train_r2_avg, all_preds_train, all_targets_train = evaluate_model(model, train_loader,
                                                                                                  criterion, device)

    val_loss, val_rmse_avg, val_r2_avg, all_preds_val, all_targets_val = evaluate_model(model, val_loader, criterion,
                                                                                        device)

    test_loss, test_rmse_avg, test_r2_avg, all_preds_test, all_targets_test = evaluate_model(model, test_loader,
                                                                                             criterion, device)

    print(f'Train RMSE --  {train_rmse_avg:.4f}')
    print(f'Val RMSE   --  {val_rmse_avg:.4f}')
    print(f'Test RMSE  --  {test_rmse_avg:.4f}')
    print(f'Train R2  --   {train_r2_avg:.4f}')
    print(f'Val R2    --   {val_r2_avg:.4f}')
    print(f'Test R2   --   {test_r2_avg:.4f}')

    metrics_train = metric_dictionary(rmse_avg=train_rmse_avg,
                                      r2_avg=train_r2_avg,
                                      data_name="training")

    metrics_val = metric_dictionary(rmse_avg=val_rmse_avg,
                                    r2_avg=val_r2_avg,
                                    data_name="validation")

    metrics_test = metric_dictionary(rmse_avg=test_rmse_avg,
                                     r2_avg=test_r2_avg,
                                     data_name="testing")

    # ========== Create DataFrame ==========
    print("\n----------------metrics with the scaled target -------")
    df_metrics = pd.DataFrame({
        f"Train k={fold_num}:": metrics_train,
        f"Validation": metrics_val,
        f"Test": metrics_test,
    }).T

    df_metrics = df_metrics.round(4)
    df_metrics.to_csv(f"{path_models}/{all_data}_table_metrics_{name_saving_files}_with_scaler.csv")
    print(df_metrics)

    # ------------------ training --------- --------------
    # --------------- invert scaled target --------------
    target_index = df_icon_train.columns.get_loc(variable_target)

    if isinstance(scaler_y, StandardScaler):
        target_means = scaler_y.mean_[target_index]
        target_vars = scaler_y.var_[target_index]

        # Unstandardize features
        df_unstandardized_all_preds_train = pd.DataFrame(
            all_preds_train.cpu().detach().numpy() * np.sqrt(target_vars) + target_means)
        df_unstandardized_all_targets_train = pd.DataFrame(
            all_targets_train.cpu().detach().numpy() * np.sqrt(target_vars) + target_means)
    
    metrics_train = metric_calculation_all_data(outputs=df_unstandardized_all_preds_train,
                                                targets=df_unstandardized_all_targets_train)

    # ------------------ training --------- --------------
    # --------------- invert standard scaler --------------
    df_unstandardized_all_preds_val = pd.DataFrame(
        all_preds_val.cpu().detach().numpy() * np.sqrt(target_vars) + target_means)
    df_unstandardized_all_targets_val = pd.DataFrame(
        all_targets_val.cpu().detach().numpy() * np.sqrt(target_vars) + target_means)
    
    metrics_val = metric_calculation_all_data(outputs=df_unstandardized_all_preds_val,
                                              targets=df_unstandardized_all_targets_val)

    # ------------------ testing --------- --------------
    # --------------- invert standard scaler --------------
    df_unstandardized_all_preds_test = pd.DataFrame(
        all_preds_test.cpu().detach().numpy() * np.sqrt(target_vars) + target_means,
        index=loaded_index_all_test)
    df_unstandardized_all_targets_test = pd.DataFrame(
        all_targets_test.cpu().detach().numpy() * np.sqrt(target_vars) + target_means,
        index=loaded_index_all_test)
    
    metrics_test = metric_calculation_all_data(outputs=df_unstandardized_all_preds_test,
                                               targets=df_unstandardized_all_targets_test)

    # print("\n----------------metrics without the log target -------")
    print("\n----------------metrics without scaled target -------")
    df_metrics = pd.DataFrame({
        f"Train k={fold_num}:": metrics_train,
        f"Validation": metrics_val,
        f"Test": metrics_test,
    }).T

    df_metrics = df_metrics.round(4)
    df_metrics.to_csv(f"{path_models}/{all_data}_table_metrics_{name_saving_files}_without_scaler.csv")
    print(df_metrics)

    print("\n----------------metrics each sample test -------")
    for i, n_time in enumerate(times_data['val_times']):
        df_target_ntime = df_unstandardized_all_targets_test.loc[
            df_unstandardized_all_targets_test.index.get_level_values(0) == n_time]
        df_pred_ntime = df_unstandardized_all_preds_test.loc[
            df_unstandardized_all_preds_test.index.get_level_values(0) == n_time]

        metrics = metric_calculation_all_data(outputs=df_pred_ntime,
                                              targets=df_target_ntime)

        print(f"------------------ {n_time} metrics ----------------------")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == '__main__':
    main()

