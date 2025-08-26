
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def plot_comparison_model_train_val(df_all_models, variable_target, output_path):
    """
    Plots mean ± std of R² and MAE with a grouped-bar approach:
      - X-axis: model names (RF, XGBoost, NN)
      - Two bars per model: Train vs. Validation
      - Only a legend for Train/Validation hatches (no model legend).

    Args:
    
        df_all_models (pd.DataFrame): columns= [fold, model, subset, R2, MAE]
        variable_target (str) Target variable name (e.g. 'lwp' or 'Nd_max').
        output_path (str) Where to save the figure, including filename (e.g. 'path/to/figure.png').
    """
    
    fontsize_lables = 20

    grouped = (
        df_all_models
        .groupby(['model', 'subset'], as_index=False)
        .agg({'R2': ['mean', 'std'], 'MAE': ['mean', 'std']})
    )
    grouped.columns = ['model', 'subset', 'R2_mean', 'R2_std', 'MAE_mean', 'MAE_std']

    model_order = ['RF', 'XGBoost', 'NN']
    subset_order = ['Train', 'Validation']

    grouped['model'] = pd.Categorical(grouped['model'], categories=model_order, ordered=True)
    grouped['subset'] = pd.Categorical(grouped['subset'], categories=subset_order, ordered=True)
    grouped = grouped.sort_values(['model', 'subset'])

    fig, (ax_r2, ax_mae) = plt.subplots(ncols=2, figsize=(10, 5))

    x = np.arange(len(model_order))  
    bar_width = 0.3  

    color_map = {
        'RF': 'saddlebrown',
        'XGBoost': 'darkorange',
        'NN': 'C0'  # default blue
    }

    hatch_map = {
        'Train': '',  # no hatch
        'Validation': '//'  # diagonal hatch
    }

    for i, model in enumerate(model_order):
        row_train = grouped[(grouped['model'] == model) & (grouped['subset'] == 'Train')]
        row_val = grouped[(grouped['model'] == model) & (grouped['subset'] == 'Validation')]

        if not row_train.empty:
            ax_r2.bar(
                x[i] - bar_width / 2,  
                row_train['R2_mean'].values[0],
                yerr=row_train['R2_std'].values[0],
                width=bar_width,
                color=color_map[model],
                hatch=hatch_map['Train'],
                alpha=0.6,
                edgecolor='k',
                capsize=4
            )
        if not row_val.empty:
            ax_r2.bar(
                x[i] + bar_width / 2, 
                row_val['R2_mean'].values[0],
                yerr=row_val['R2_std'].values[0],
                width=bar_width,
                color=color_map[model],
                hatch=hatch_map['Validation'],
                alpha=0.6,
                edgecolor='k',
                capsize=4
            )

    # R² axis styling
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(model_order, ha='center', fontsize=fontsize_lables - 3)  # , rotation=45,
    ax_r2.tick_params(axis='y', labelsize=fontsize_lables - 3)
    ax_r2.set_ylabel(r'$R^2$', fontsize=fontsize_lables)
    ax_r2.set_ylim(0.2, 1)  # optional
    # ax_r2.set_title("Mean ± Std")
    ax_r2.grid(True)

    # Plot MAE as grouped bars
    for i, model in enumerate(model_order):
        row_train = grouped[(grouped['model'] == model) & (grouped['subset'] == 'Train')]
        row_val = grouped[(grouped['model'] == model) & (grouped['subset'] == 'Validation')]

        if not row_train.empty:
            ax_mae.bar(
                x[i] - bar_width / 2,
                row_train['MAE_mean'].values[0],
                yerr=row_train['MAE_std'].values[0],
                width=bar_width,
                color=color_map[model],
                hatch=hatch_map['Train'],
                alpha=0.6,
                edgecolor='k',
                capsize=4
            )
        if not row_val.empty:
            ax_mae.bar(
                x[i] + bar_width / 2,
                row_val['MAE_mean'].values[0],
                yerr=row_val['MAE_std'].values[0],
                width=bar_width,
                color=color_map[model],
                hatch=hatch_map['Validation'],
                alpha=0.6,
                edgecolor='k',
                capsize=4
            )

    if variable_target == 'lwp':
        variable_target_title = 'LWP'
        unit_title = r'$gm^{-2}$'
    elif variable_target == "Nd_max":
        variable_target_title = '$N_{d,\\ max}$'
        unit_title = r'$cm^{-3}$'
    else:
        variable_target_title = variable_target

    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(model_order, ha='center', fontsize=fontsize_lables - 3)  # , rotation=45,
    ax_mae.tick_params(axis='y', labelsize=fontsize_lables - 3)
    ax_mae.set_ylabel(f"MAE ({unit_title})", fontsize=fontsize_lables)
    ax_mae.grid(True)

    train_patch = mpatches.Patch(facecolor='white', hatch='', edgecolor='k', label='Train')
    val_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='k', label='Validation')
    fig.legend(handles=[train_patch, val_patch], loc='lower center', ncol=2, fontsize=fontsize_lables,
               bbox_to_anchor=(0.5, -0.1))


    fig.suptitle(f"Train vs. Validation for {variable_target_title} (Mean ± SD)", fontsize=fontsize_lables, y=0.95)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_models_train_val(main_folder="/work/bb1143/b381362/output/final_results"):
    variable_targets = ["lwp", "Nd_max"]
    type_models = ["RF", "XGBoost", "NN"]
    all_data_options = ["all_features", "most_importances"]

    for variable_target in variable_targets:
        for all_data in all_data_options:
            rows = []
            for type_model in type_models:
                base_path = f"{main_folder}/{type_model}/{variable_target}_{all_data}"
                for fold_num in range(4):
                    csv_path = f"{base_path}/{all_data}_table_metrics_{type_model}_{variable_target}_k_fold_{fold_num}_without_scaler.csv"

                    df_fold = pd.read_csv(csv_path)

                    # Extract Train/Validation rows
                    train_row = df_fold[df_fold['Unnamed: 0'].str.contains('Train')].reset_index(drop=True)
                    val_row = df_fold[df_fold['Unnamed: 0'] == 'Validation'].reset_index(drop=True)

                    r2_train = train_row["R-squared (R2)"].values[0]
                    mae_train = train_row["Mean Absolute Error (MAE)"].values[0]

                    r2_val = val_row["R-squared (R2)"].values[0]
                    mae_val = val_row["Mean Absolute Error (MAE)"].values[0]

                    rows.append({
                        'fold': fold_num,
                        'model': type_model,
                        'subset': 'Train',
                        'R2': r2_train,
                        'MAE': mae_train
                    })
                    rows.append({
                        'fold': fold_num,
                        'model': type_model,
                        'subset': 'Validation',
                        'R2': r2_val,
                        'MAE': mae_val
                    })

            df_all_models = pd.DataFrame(rows)
            out_fig_path = f"{main_folder}/{all_data}_{variable_target}_train_val_metrics.png"
            print(out_fig_path)

            plot_comparison_model_train_val(
                df_all_models=df_all_models,
                variable_target=variable_target,
                output_path=out_fig_path
            )


# ---------------------------------------------------------------------
def plot_comparison_model_validation(df_all_models, variable_target, output_path):
    """
    Plots mean ± std of R² and MAPE for the Validation subset using two bars per model:
      - One bar for 'all_features'
      - One bar for 'most_importances'

    Args:
        df_all_models (pd.DataFrame): Columns must include: ['fold', 'model', 'subset', 'R2', 'MAPE', 'all_data'](Only Validation rows are expected.)
    variable_target (str): Target variable name (e.g. 'lwp' or 'Nd_max').
    output_path (str): Where to save the figure (including filename).
    """
    fontsize_labels = 20

    df_val = df_all_models[df_all_models['subset'] == 'Validation'].copy()

    grouped = (
        df_val
        .groupby(['model', 'all_data'], as_index=False)
        .agg({'R2': ['mean', 'std'], 'MAE': ['mean', 'std']})
    )
    grouped.columns = ['model', 'all_data', 'R2_mean', 'R2_std', 'MAE_mean', 'MAE_std']

    model_order = ['RF', 'XGBoost', 'NN']
    all_data_order = ['all_features', 'most_importances']

    # Ordering
    grouped['model'] = pd.Categorical(grouped['model'], categories=model_order, ordered=True)
    grouped['all_data'] = pd.Categorical(grouped['all_data'], categories=all_data_order, ordered=True)
    grouped = grouped.sort_values(['model', 'all_data'])

    fig, (ax_r2, ax_mae) = plt.subplots(ncols=2, figsize=(10, 5))

    x = np.arange(len(model_order))  
    bar_width = 0.3  

    color_map = {
        'RF': 'saddlebrown',
        'XGBoost': 'darkorange',
        'NN': 'C0'  # default blue
    }
    
    hatch_map = {
        'all_features': '',  # plain
        'most_importances': '//'  # diagonal hatch
    }
    # --- Plot R² ---
    for i, model in enumerate(model_order):
        for j, data_option in enumerate(all_data_order):
            row = grouped[(grouped['model'] == model) & (grouped['all_data'] == data_option)]
            if not row.empty:
                # For two bars per model, we shift left/right based on j
                offset = (-bar_width / 2) if data_option == 'all_features' else (bar_width / 2)
                ax_r2.bar(
                    x[i] + offset,
                    row['R2_mean'].values[0],
                    yerr=row['R2_std'].values[0],
                    width=bar_width,
                    color=color_map[model],
                    hatch=hatch_map[data_option],
                    alpha=0.6,
                    edgecolor='k',
                    capsize=4
                )

    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(model_order, ha='center', fontsize=fontsize_labels - 3)
    ax_r2.tick_params(axis='y', labelsize=fontsize_labels - 3)
    ax_r2.set_ylabel(r'$R^2 $', fontsize=fontsize_labels)
    ax_r2.set_ylim(0.2, 1)
    ax_r2.grid(True)

    # --- Plot MAE ---
    for i, model in enumerate(model_order):
        for j, data_option in enumerate(all_data_order):
            row = grouped[(grouped['model'] == model) & (grouped['all_data'] == data_option)]
            if not row.empty:
                offset = (-bar_width / 2) if data_option == 'all_features' else (bar_width / 2)
                ax_mae.bar(
                    x[i] + offset,
                    row['MAE_mean'].values[0],
                    yerr=row['MAE_std'].values[0],
                    width=bar_width,
                    color=color_map[model],
                    hatch=hatch_map[data_option],
                    alpha=0.6,
                    edgecolor='k',
                    capsize=4
                )

    if variable_target == 'lwp':
        variable_target_title = 'LWP'
        variable_target_title = '$L$'
        unit_title = r'$\mathrm{g\,m}^{-2}$'
    elif variable_target == "Nd_max":
        # variable_target_title = '$N_{d,\\ max}$'
        variable_target_title = '$N_{\mathrm{d},\max}$'
        unit_title = r'$cm^{-3}$'
    else:
        variable_target_title = variable_target

    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(model_order, ha='center', fontsize=fontsize_labels - 3)
    ax_mae.tick_params(axis='y', labelsize=fontsize_labels - 3)
    ax_mae.set_ylabel(f"MAE ({unit_title})", fontsize=fontsize_labels)
    ax_mae.grid(True)

    patch_all = mpatches.Patch(facecolor='white', hatch=hatch_map['all_features'], edgecolor='k', label='All channels')
    patch_most = mpatches.Patch(facecolor='white', hatch=hatch_map['most_importances'], edgecolor='k',
                                label='Most importances channels')
    fig.legend(handles=[patch_all, patch_most], loc='lower center', ncol=2, fontsize=fontsize_labels,
               bbox_to_anchor=(0.5, -0.1))

    fig.suptitle(f"Validation Metrics for {variable_target_title} (Mean ± SD)",
                 fontsize=fontsize_labels, y=0.95)

    plt.tight_layout()
    
    grouped.to_csv(f"{output_path}.csv")
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()


def plot_comparison_models_all_mif_features(main_folder):

    variable_targets = ["lwp", "Nd_max"]
    type_models = ["RF", "XGBoost", "NN"]
    all_data_options = ["all_features", "most_importances"]

    for variable_target in variable_targets:
        rows = []
        for all_data in all_data_options:
            for type_model in type_models:
                base_path = f"{main_folder}/{type_model}/{variable_target}_{all_data}"
                for fold_num in range(4): 
                    csv_path = f"{base_path}/{all_data}_table_metrics_{type_model}_{variable_target}_k_fold_{fold_num}_without_scaler.csv"
                    df_fold = pd.read_csv(csv_path)

                    val_row = df_fold[df_fold['Unnamed: 0'] == 'Validation'].reset_index(drop=True)
                    if val_row.empty:
                        continue  # Skip if there is no Validation row

                    r2_val = val_row["R-squared (R2)"].values[0]
                    mae_val = val_row["Mean Absolute Error (MAE)"].values[0]

                    rows.append({
                        'fold': fold_num,
                        'model': type_model,
                        'subset': 'Validation',
                        'R2': r2_val,
                        'MAE': mae_val,
                        'all_data': all_data
                    })

        df_all_models = pd.DataFrame(rows)
        out_fig_path = f"{main_folder}/{variable_target}_validation_metrics_comparison"
        print(out_fig_path)
        plot_comparison_model_validation(
            df_all_models=df_all_models,
            variable_target=variable_target,
            output_path=out_fig_path
        )



def main():
    """ Plotting the metrics along the models and sets"""
    parser = argparse.ArgumentParser(description='Create dataframes')
    arg = parser.add_argument
    arg('--main_folder_path', type=str, default="/work/bb1036/b381362/output/final_results")

    args = parser.parse_args()
    main_folder = args.main_folder_path

    plot_comparison_models_train_val(main_folder=main_folder)

    plot_comparison_models_all_mif_features(main_folder=main_folder)


if __name__ == '__main__':
    main()