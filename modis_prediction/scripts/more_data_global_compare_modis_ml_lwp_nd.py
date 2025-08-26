import os
import argparse
import sys
import pandas as pd
import numpy as np
import xarray as xr

from match_files_modis import read_files_modis, match_level1_level2, obtain_files_match

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.functions_ml import predict_with_modis_level1
from utils.functions_modis import plot_lwp_nd_modis_retrievals, data_level1_level2
# modis_level1_level2
from plotting.plots import plot_pcolormesh, plot_pcolormesh_no_lat_lon, histogram_method_comparison, plot_joint_histogram_conditional, histogram_method_comparison_modis, plot_geospatial_comparison
from utils.functions_general import save_df_to_netcdf_fold
                                         
def main():
    parser = argparse.ArgumentParser(description='compare methods')
    arg = parser.add_argument
    arg('--main_folder_ml_lwp', type=str, default="/work/")
    arg('--main_folder_ml_nd', type=str, default="/work/")
    arg('--result_folder', type=str, default="/work/global_data")
    arg('--folder_path_modis_data', type=str, default="/work/modis_data")

    arg('--all_data', type=str, default="all_features", help='all_features or most_importances')  
    arg('--fold_num', type=int, default=0, help='number of the flod')  
    
    arg('--type_model', type=str, default="NN")
    arg('--variables_names', type=str, default="LWP-Ndmax", help='"Re-COT" List of the variables to use.')
    arg('--num_channels', type=str, default="18_chan", help='"18_chan"  or 17_chan')
    # arg('--type_data_modis', type=str, default="retrievals_level2", help='retrievals_level2 or level1_level2.')

    args = parser.parse_args()
    
    main_folder_ml_lwp = args.main_folder_ml_lwp
    main_folder_ml_nd = args.main_folder_ml_nd
    result_folder = args.result_folder
    folder_path_modis_data = args.folder_path_modis_data
    all_data = args.all_data
    fold_num = args.fold_num
    type_model = args.type_model
    variables_names = args.variables_names
    num_channels = args.num_channels

    min_value_threshold = 5
    max_value_threshold = 2000
    
    map_boundaries = None
    # map_boundaries = [50.008, 54.5, 4.5, 14.5] 

    variables = ["Nd_max", "lwp"]
    mask_cldph=[1]

    result_folder_hist = f"{result_folder}/hist"
    os.makedirs(result_folder_hist, exist_ok=True)  


    # ----------------------------- ML  parameter ----------------------------------
    # -----------------  more data -----------------------
    print("using more data =========================================================== ")
    main_folder_model_ml_lwp = f"{main_folder_ml_lwp}/more_data_output_main/final_results"
    main_folder_model_ml_nd = f"{main_folder_ml_nd}/more_data_output_main/final_results"
    
    path_model_file_lwp=f"{main_folder_model_ml_lwp}/{type_model}/lwp_{all_data}/{all_data}_{type_model}_lwp_k_fold_{fold_num}.pth"
    path_model_file_nd=f"{main_folder_model_ml_nd}/{type_model}/Nd_max_{all_data}/{all_data}_{type_model}_Nd_max_k_fold_{fold_num}.pth"
    
    path_dataframes_scaler_lwp = f"{main_folder_ml_lwp}/more_data_dataframe_all_data"
    path_dataframes_scaler_nd = f"{main_folder_ml_nd}/more_data_dataframe_all_data"
    
    # -----------------  only germany data -----------------------
    # print("using only data germany =========================================================== ")
    # main_folder_model_ml_lwp = f"{main_folder_ml_lwp}/output_main/final_results"
    # main_folder_model_ml_nd = f"{main_folder_ml_nd}/output_main/final_results"
    
    # path_model_file_lwp=f"{main_folder_model_ml_lwp}/{type_model}/lwp_{all_data}/{all_data}_{type_model}_lwp_k_fold_{fold_num}.pth"
    # path_model_file_nd=f"{main_folder_model_ml_nd}/{type_model}/Nd_max_{all_data}/{all_data}_{type_model}_Nd_max_k_fold_{fold_num}.pth"
    
    # path_dataframes_scaler_lwp = f"{main_folder_ml_lwp}/dataframe_all_data"
    # path_dataframes_scaler_nd = f"{main_folder_ml_nd}/dataframe_all_data"
   

    # ================ obtaine files ====================
    matched = obtain_files_match(folder_path_modis_data)
    print(f"==========sizes matched ======== {np.shape(matched)}=====================")
 
    # =============================================================
    # ---------------------- RETRIEVAL MODIS LEVEL 2 --------------  
    # =============================================================
    print(" ========================= Retrievals ==============================")
    # df_concatenate_ml_level1 = pd.DataFrame()
    # df_concatenate_level2 = pd.DataFrame()
    # df_concatenate_level1_level2 = pd.DataFrame()
    
    # for i, times_step, path_level1, path_level2 in matched[64:65]:
    # for i, times_step, path_level1, path_level2 in matched:
    
    # Instead of looping all files:
    i = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    x, times_step, path_level1, path_level2 = matched[i]
    # i=64
    # x, times_step, path_level1, path_level2 = matched[64]  # germany

    
    processed_nc_modis_level1 = path_level1
    processed_nc_modis_level2_path = path_level2
    
    retrievals_ds = xr.open_dataset(processed_nc_modis_level2_path)
    
    print(f"======================={processed_nc_modis_level2_path}======================= map_boundaries: {map_boundaries} ================================")
            
    retrievals_ds = xr.open_dataset(path_level1)
    
    print(f"======================={path_level1}======================= map_boundaries: {map_boundaries} ================================")

    cleaned_index, df_spectral, df_spectral_lwp, df_spectral_nd, filtered_df_modis_retrievals, channel_relate_clouds_lwp, channel_relate_clouds_nd = data_level1_level2(map_boundaries=map_boundaries, 
                                                           result_folder=result_folder,
                                                           all_data=all_data,
                                                           mask_cldph=mask_cldph, 
                                                           variable = ["Nd_max", "lwp"], 
                                                          min_value_threshold=min_value_threshold, 
                                                           max_value_threshold=max_value_threshold,
                                                           level1_path=processed_nc_modis_level1, 
                                                           level2_path=processed_nc_modis_level2_path,
                                                                                                                                                                           num_channels=num_channels) 

    # ----------- With MODIS channels level1 and mask level2 and ML predic LWP, Nd -----------
    df_unstandardized_modis_lwp, df_unstandardized_modis_nd= predict_with_modis_level1(cleaned_index=cleaned_index,
                                                                                       df_spectral=df_spectral,

                                                                                       df_spectral_lwp=df_spectral_lwp,
                                                                                       df_spectral_nd=df_spectral_nd,
                                                                                       channel_relate_clouds_lwp=channel_relate_clouds_lwp, 
                                                                                       channel_relate_clouds_nd=channel_relate_clouds_nd, 
                                                                                      path_model_file_lwp=path_model_file_lwp, 
                                                                                      path_model_file_nd=path_model_file_nd, 
                                                                                      path_dataframes_scaler_lwp=path_dataframes_scaler_lwp, 
                                                                                      path_dataframes_scaler_nd=path_dataframes_scaler_nd, 
                                                                                      variables_names=variables_names, 
                                                                                      fold_num=fold_num)
                        

    # Combine into a new dataframe
    filtered_df_modis_ml = pd.DataFrame({
        'Nd_max': df_unstandardized_modis_nd["Nd_max"],
        'lwp': df_unstandardized_modis_lwp["lwp"]
    })

    joint_mask = (
        (filtered_df_modis_ml[variables] >= min_value_threshold) &
        (filtered_df_modis_ml[variables] <= max_value_threshold) &
        (filtered_df_modis_retrievals[variables] >= min_value_threshold) &
        (filtered_df_modis_retrievals[variables] <= max_value_threshold)
    ).all(axis=1)  # all selected columns must be valid
    
    # Apply the joint mask to both DataFrames
    filtered_df_modis_ml = filtered_df_modis_ml.where(joint_mask)
    filtered_df_modis_retrievals = filtered_df_modis_retrievals.where(joint_mask)

    
    # index = times_step.split('.')[1]
    times_step = times_step.replace('.', '_')
    index = times_step


    if filtered_df_modis_retrievals is not None:
        filtered_df_modis_retrievals = filtered_df_modis_retrievals.reset_index()
        # df_concatenate_level2 = pd.concat([df_concatenate_level2, filtered_df_modis_retrievals], ignore_index=True)
        
    if filtered_df_modis_ml is not None:
        # Convert lat/lon index to columns before appending
        filtered_df_modis_ml = filtered_df_modis_ml.reset_index()
        # Append to final dataframe
        # df_concatenate_ml_level1 = pd.concat(
            # [df_concatenate_ml_level1, filtered_df_modis_ml], ignore_index=True
        # )

    # ============================= Distribution plot ====================
    # modis_ml_ds = filtered_df_modis_ml.to_xarray()
    modis_ml_ds = filtered_df_modis_ml.set_index(['lat', 'lon'], inplace=False).to_xarray()
    # modis_level2 = filtered_df_modis_retrievals.to_xarray()
    modis_level2 = filtered_df_modis_retrievals.set_index(['lat', 'lon'], inplace=False).to_xarray()
    
    for variable_target in variables:
        plot_geospatial_comparison(variable_target=variable_target,
            datasets=[modis_level2, modis_ml_ds],
            titles=["MODIS retrievals", "MODIS spectral + ML"],
            vmin=1e0,
            vmax=1e3,
                                   result_folder=result_folder_hist,
                                   index=index
        
        )
    # # ========================   

    
    save_df_to_netcdf_fold(df=filtered_df_modis_ml,
                            fold_num=fold_num,
                            prefix=f'df_level1_ml_predictions_{index}', 
                            path_output=result_folder_hist
    )


    save_df_to_netcdf_fold(df=filtered_df_modis_retrievals, 
                               fold_num=fold_num,
                               prefix=f'df_level2_modis_{index}', 
                               path_output=result_folder_hist)
    

    filtered_df_modis_retrievals = filtered_df_modis_retrievals.rename(columns={'Nd_max': r'$N_{d,\ MODIS L2}$', 
                                                 'lwp': r'LWP_{MODIS L2}'}, inplace=False)
    filtered_df_modis_ml = filtered_df_modis_ml.rename(columns={'Nd_max': r'$N_{d,\ MODIS L1}$', 
                                             'lwp': r'LWP_{MODIS L1}'}, inplace=False)
    
    histogram_method_comparison_modis(df_modis_retrievals=filtered_df_modis_retrievals, 
                                df_modis_ml=filtered_df_modis_ml, 
                                clip_min=5, 
                                clip_maximun=1000,
                                result_folder=result_folder_hist,
                                index=index)
    
        # ===================================== end concatenate ===========================

    # save_df_to_netcdf_fold(df=df_concatenate_ml_level1,
    #                         fold_num=fold_num,
    #                         prefix='df_level1_ml_predictions', 
    #                         path_output=result_folder
    # )


    # save_df_to_netcdf_fold(df=df_concatenate_level2, 
    #                            fold_num=fold_num,
    #                            prefix='df_level2_modis', 
    #                            path_output=result_folder)
    

    # df_concatenate_ml_level1 = df_concatenate_ml_level1.drop(columns=['lat', 'lon'])
    # df_concatenate_level2 = df_concatenate_level2.drop(columns=['lat', 'lon'])

    # print("============================ Plotting =============================")
    
    # path_file = f"{result_folder}/nd_lwp_ml_modis_level1_global.png"
    # fig, ax = plot_joint_histogram_conditional(df_concatenate_ml_level1, 
    #                                df_concatenate_ml_level1.columns[0], 
    #                                df_concatenate_ml_level1.columns[1], 
    #                                threshold="range7-870", 
    #                                # threshold="all", 
    #                                figure_size=(6, 6))
    # fig.savefig(path_file, bbox_inches="tight")
 
    # fig, ax = plot_joint_histogram_conditional(df_concatenate_level2, 
    #                                            df_concatenate_level2.columns[0], 
    #                                            df_concatenate_level2.columns[1], 
    #                                            threshold="range7-870", 
    #                                            figure_size=(6, 6))       
    
    # path_file = f"{result_folder}/nd_lwp_modis_retrievals_global.png"
    # fig.savefig(path_file, bbox_inches="tight")

    
    

    # df_concatenate_level2.rename(columns={'Nd_max': r'$N_{d,\ MODIS L2}$', 
    #                                                  'lwp': r'LWP_{MODIS L2}'}, inplace=True)
    # df_concatenate_ml_level1.rename(columns={'Nd_max': r'$N_{d,\ MODIS L1}$', 
    #                                          'lwp': r'LWP_{MODIS L1}'}, inplace=True)


    # histogram_method_comparison_modis(df_modis_retrievals=df_concatenate_level2, 
    #                             df_modis_ml=df_concatenate_ml_level1, 
    #                             clip_min=5, 
    #                             clip_maximun=1000,
    #                            result_folder=result_folder)

    


if __name__ == '__main__':
    main()
