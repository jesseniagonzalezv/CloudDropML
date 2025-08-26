import os
import argparse
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.functions_icon_rttov import plot_lwp_nd_icon, get_lwp_Nd, obtain_nd_lwp_ml_prediction_with_rttov_channels
from utils.functions_modis import plot_lwp_nd_modis_retrievals, obtain_lwp_nd_from_modis_level1_and_ML
from plotting.plots import plot_pcolormesh, plot_pcolormesh_no_lat_lon, histogram_method_comparison


def compare_different_methods_predict_lwp_nd(main_folder_ml_lwp, main_folder_ml_nd, output_results):
    # main_folder =  "/work/bb1036/b381362/"

    time = "11_40"
    lat_min = 50
    lat_max = 54.50
    # min_value_threshold = 5
    # max_value_threshold = 2000
    
    min_value_threshold = None
    max_value_threshold = None
    variables = ["Nd_max", "lwp"]
    variables_names = "LWP-Ndmax"
    map_boundaries = [50.008, 54.5, 4.5, 14.5] 

    type_model="NN"
    all_data="all_features"
    # all_data = "most_importances" 
    fold_num=0

    result_folder = f"{output_results}/lwp_nd_compare_methods/plot_cap3/{all_data}/both_1143"
    
    main_folder_model_ml_lwp = f"{main_folder_ml_lwp}/output_main/final_results"
    main_folder_model_ml_nd = f"{main_folder_ml_nd}/output_main/final_results"
    os.makedirs(result_folder, exist_ok=True)  

    # ------------------------ ICON - MODIS RTTOV simulations -----------------------
    path_icon = f"/work/bb1036/rttov_share/dataset_ICON/icon_germany_20130502_T{time}.nc"
    path_icon_timesteps = f"/work/bb1036/rttov_share/dataset_ICON/icon_germany_20130502_T{time}_lwp_Nd.nc"
    path_channels_timesteps = f"/work/bb1036/rttov_share/dataset_SPECTRAL/S3COM_OP_20130502_T{time}_36ch_fromlat50_rad.nc"
    
    # ----------- ML  predictions with MODIS RTTOV simulations ------------------    
    path_ml_prediction_lwp = f"{main_folder_model_ml_lwp}/NN/lwp_{all_data}/prediction/prediction_target_{type_model}_lwp_k_fold_{fold_num}.nc"    
    path_ml_prediction_nd = f"{main_folder_model_ml_nd}/NN/Nd_max_{all_data}/prediction/prediction_target_{type_model}_Nd_max_k_fold_{fold_num}.nc"
    
    # ----------------------------- Level 1 configuration -------------------------
    # processed_nc_modis_level1 = "/work/bb1036/rttov_share/MODIS_data/Level1-2-test/MYD021KM.A2013122.1140.061.2018046032403.nc"
    processed_nc_modis_level1 = "/work/bb1143/b381362/MODIS_data/MYD021KM.A2013122.1140.061.2018046032403.nc"

    # ----------------------------- Level 2 configuration ---------------------------
    # mask_cldph=[1, 6]  # it has more points 
    mask_cldph=[1]
    # processed_nc_modis_level2 = "/work/bb1036/rttov_share/MODIS_data/Level1-2-test/processed_MYD06_L2_may_v2.nc"
    processed_nc_modis_level2_path = "/work/bb1143/b381362/MODIS_data/MYD06_L2.A2013122.1140.061.2018046112442.nc"

    # ----------------------------- ML  parameter ----------------------------------
    path_model_file_lwp=f"{main_folder_model_ml_lwp}/{type_model}/lwp_{all_data}/{all_data}_{type_model}_lwp_k_fold_{fold_num}.pth"
    path_model_file_nd=f"{main_folder_model_ml_nd}/{type_model}/Nd_max_{all_data}/{all_data}_{type_model}_Nd_max_k_fold_{fold_num}.pth"

    path_dataframes_scaler_lwp = f"{main_folder_ml_lwp}/dataframe_all_data"
    path_dataframes_scaler_nd = f"{main_folder_ml_nd}/dataframe_all_data"
    # pytorch_inverse_only_clouds_channels/StandardScaler_StandardScaler/filter/LWP-Ndmax/lwp/NN_lwp_k_fold_0.pth"
    # path_model_file_nd = "/work/bb1036/b381362/output/pytorch_inverse_only_clouds_channels/StandardScaler_StandardScaler/filter/LWP-Ndmax/Nd_max/NN_Nd_max_k_fold_0.pth"
    # path_dataframes_scaler = "/work/bb1036/b381362/output/pytorch_inverse_only_clouds_channels/StandardScaler_StandardScaler/filter/LWP-Ndmax/"
    # -----------------------------------

    

    # =============================================================
    # --------------------- LWP and Nd ICON  ---------------------
    # =============================================================
    if not os.path.exists(path_icon_timesteps):
        ds = get_lwp_Nd(path_icon = path_icon) #it will create a file with ICON lwp and nd
    
    filtered_df_icon_les = plot_lwp_nd_icon(path=path_icon_timesteps, 
                                                  variables = variables,
                                                  result_folder=result_folder,
                                                  lat_min=lat_min, 
                                                  lat_max=lat_max,
                                                  min_value_threshold=min_value_threshold, 
                                                  max_value_threshold=max_value_threshold)
        
    
    # # =============================================================
    # --------------------- Retrieval level 2 MODIS ---------------------
    # =============================================================

    modis_level2_ds, masked_subset_ds, subset_level2, filtered_df_modis_retrievals = plot_lwp_nd_modis_retrievals(path=processed_nc_modis_level2_path, 
                                                                                                                      map_boundaries=map_boundaries,
                                                                                                                      result_folder=result_folder,
                                                                                                                      mask_cldph=mask_cldph,  # only liquid water clouds
                                                                                                                      min_value_threshold=min_value_threshold, 
                                                                                                                      max_value_threshold=max_value_threshold)
    
    variable_name = "lwp"
    plot_pcolormesh(longitude=modis_level2_ds.lon, 
                    latitude=modis_level2_ds.lat, 
                    variable=modis_level2_ds[variable_name], 
                    name_var=variable_name, 
                    level_name="2")
    # --------------- subset germany ---------------------------------
    # Germany, need to invert
    plot_pcolormesh_no_lat_lon(variable=subset_level2[variable_name], 
                               name_var=variable_name, 
                           level_name="2")                                            

    # # =============================================================
    # # --------- RTTOV channels with ML to predict LWP and Nd icon-les--------------  
    # # =============================================================

    if os.path.exists(path_ml_prediction_nd) and os.path.exists(path_ml_prediction_nd):
        print("predictions already exist")
        plot_lwp_nd_from_predictions_nc(path_ml_prediction_nd=path_ml_prediction_nd, 
                                          path_ml_prediction_lwp=path_ml_prediction_lwp,
                                          min_value_threshold=min_value_threshold, 
                                          max_value_threshold=max_value_threshold,
                                        result_folder=result_folder)
    else:
        print("predictions not exist")
    
        filtered_df_ml_rttov, ds_rttov_ini, masked_ds_rttov = obtain_nd_lwp_ml_prediction_with_rttov_channels(path_icon=path_icon_timesteps,
                                                                                                              path_rttov=path_channels_timesteps,
                                                                                                              all_data=all_data,
                                                                                                              path_model_file_lwp=path_model_file_lwp,
                                                                                                              path_model_file_nd=path_model_file_nd,
                                                                                                              path_dataframes_scaler_lwp=path_dataframes_scaler_lwp,
                                                                                                              path_dataframes_scaler_nd=path_dataframes_scaler_nd,
                                                                                                              fold_num=fold_num, 
                                                                                                              variables=variables, 
                                                                                                              variables_names=variables_names,
                                                                                                              lat_min=lat_min, 
                                                                                                              lat_max=lat_max, 
                                                                                                              min_value_threshold=min_value_threshold, 
                                                                                                              max_value_threshold=max_value_threshold,
                                                                                                             result_folder=result_folder)

                                                                                  
    # =============================================================
    # --------- ML model + MODIS (level1) to predict LWP and Nd --------------  
    # =============================================================
    # ----------- With MODIS channels level1 and mask level2 and ML predic LWP, Nd -----------
    filtered_df_modis_ml, ref_rad_18_chan = obtain_lwp_nd_from_modis_level1_and_ML(level1_path=processed_nc_modis_level1, level2_path=processed_nc_modis_level2_path, 
                                                                                   result_folder=result_folder,
                                                                                   map_boundaries=map_boundaries, 
                                                                                   all_data=all_data, 
                                                                                   mask_cldph=mask_cldph, 
                                                                                   variables_names=variables_names,
                                                                                   min_value_threshold=min_value_threshold, 
                                                                                   max_value_threshold=max_value_threshold,
                                                                                  path_model_file_lwp=path_model_file_lwp, 
                                                                                    path_model_file_nd=path_model_file_nd,
                                                                                   path_dataframes_scaler_lwp=path_dataframes_scaler_lwp,
                                                                                   path_dataframes_scaler_nd=path_dataframes_scaler_nd,
                                                                              
                                                                                    fold_num=fold_num)


    # # =================== Histograms
    filtered_df_icon_les.rename(columns={'Nd_max': r'$N_{d,\ ICON-LES}$', 
                                         'lwp': r'LWP_{ICON-LES}'}, inplace=True)
    
    filtered_df_modis_retrievals.rename(columns={'Nd_max': r'$N_{d,\ MODIS L2}$', 
                                                 'lwp': r'LWP_{MODIS L2}'}, inplace=True)
    
    filtered_df_ml_rttov.rename(columns={'Nd_max': r'$N_{d,\ MODIS simulated}$', 
                                         'lwp': r'LWP_{MODIS simulated}'}, inplace=True)
    
    filtered_df_modis_ml.rename(columns={'Nd_max': r'$N_{d,\ MODIS L1}$', 
                                         'lwp': r'LWP_{MODIS L1}'}, inplace=True)

    histogram_method_comparison(df_icon_les=filtered_df_icon_les, 
                                df_modis_retrievals=filtered_df_modis_retrievals, 
                                df_ml_rttov=filtered_df_ml_rttov, 
                                df_modis_ml=filtered_df_modis_ml, 
                                clip_min=0, 
                                clip_maximun=1000,
                               result_folder=result_folder)

                                         
def main():
    parser = argparse.ArgumentParser(description='compare methods')
    arg = parser.add_argument
    arg('--main_folder_ml_lwp', type=str, default="/work/bb1143/b381362-")
    arg('--main_folder_ml_nd', type=str, default="/work/bb1143/b381362-")
    arg('--output_results', type=str, default="/work/bb1143/b381362-")
 
    args = parser.parse_args()
    main_folder_ml_lwp = args.main_folder_ml_lwp
    main_folder_ml_nd = args.main_folder_ml_nd
    output_results = args.output_results
     
    compare_different_methods_predict_lwp_nd(main_folder_ml_lwp=main_folder_ml_lwp, 
                                             main_folder_ml_nd=main_folder_ml_nd, 
                                             output_results=output_results)

    # mejor old model for ndmax bb1036 and for lwp mejor new model.


if __name__ == '__main__':
    main()