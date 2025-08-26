# Project Overview

This project implements the use of the ML model with MODIS level 1 and also we process MODIS level 2:    
        
    modis_prediction/
        ├── preprocess_level1_level2.sh              
            ├── preprocess_modis_level1.ncl              
            └── preprocess_modis_level2.ncl    
        ├── global_compare_methods.sh                  
        ├── mergue-plot.sh                  
        └── pred_grid_11_40_compare_methods.sh  (optional) 
           
        ├── scripts/               
        │   ├── obtain_name_files.py
        │   ├── 11_40_predict_reproject.py
        │   ├── global_compare_modis_ml_lwp_nd.py
        │   ├── match_files_modis.py
        │   ├── mergue_netcdf_plot.py
        │   ├── methods_obtaine_lwp_nd.py
        │   └── more_data_global_compare_modis_ml_lwp_nd.py
        
        
        ├── plotting/               
        │   ├── plotting_prediction_target.py
        │   ├── plot_compare_model_metrics.py
        │   ├── NN_join_heatmaps.py
        │   ├── join_heatmaps.py
        │   └── plot_features_vs_metric.py
        
        ├── utils/                  
        │   ├── functions_icon_rttov.py
        │   ├── functions_ml.py
        │   ├── functions_general.py
        │   └── functions_modis.py
    
        ├── notebooks/               
        │   └── IN PROCESS
    
            
Below is a detailed breakdown of each step along with the corresponding scripts.

---
### DataFrame Creation
- **Description:**  
- **How to Run:**
   - **Step 1: Data Preparation 
     ```bash
     bash preprocess_level1_level2.sh
- **Scripts:**
  - `preprocess_modis_level1.ncl`
  - `preprocess_modis_level2.ncl`

---
## ML Methods
### 1. Model Prediction
- **Description:**  
    This code allows us to predict with the ML models given a NetCDF file.
- **Main Scripts:**
  - `train_predict_plot.sh`  
- **How to Run:**
   - **Step 2: Prediction and Reproject**
     ```bash
     bash global_compare_methods.sh
     ```
### 2. Mergue by latitude and longitude and plot
- **Description:**  
    This code allows us to concatenate at global given different NetCDF file predictions.
- **Main Scripts:**
  - `mergue-plot.sh`  
- **How to Run:**
   - **Step 2: mergue**
     ```bash
     bash mergue-plot.sh (given a specific timestep at global select the option requiered  type_aggregation="mean_by_date", "plot_hist_plot2d_date" "plot_hexbin", given a different timestep at global select the option requiered to obtaine a plot as the mean of all dates  type_aggregation="mean_all_dates", "plot_averaged_global_hexbin" "plot_averaged_global_hist"))
     ```