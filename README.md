
# Project Overview

This project implements both standard statistical methods and machine learning (ML) methods for data analysis and prediction. The pipeline is divided into two main sections:

    main_folder/
    ├── dataframe_all_data.sh              
    ├── correlation_joinplot.sh                  
    ├── partial_corr.sh                  
    ├── train_predict_plot.sh              
    ├── NN_feature_importances.sh              
    ├── RF_feature_importances.sh              
    ├── feature_reduction_all_methods.sh              
    ├── plots_results.sh              
    ├── predict_plot_sample.sh              
    ├── README.md               
    
    ├── scripts/               
    │   ├── dataframe_all_data.py
    │   ├── correlation_joinplot.py
    │   ├── partial_corr.py
    │   ├── inverse_model_RF.py
    │   ├── inverse_model_NN.py
    │   ├── get_prediction_test.py
    │   ├── NN_feature_importances.py
    │   ├── RF_feature_importances.py
    │   └── feature_reduction_all_methods.py
    │   └── predict_plot_sample.py
    
    
    ├── plotting/               
    │   ├── plotting_prediction_target.py
    │   ├── plot_compare_model_metrics.py
    │   ├── NN_join_heatmaps.py
    │   ├── join_heatmaps.py
    │   └── plot_features_vs_metric.py
    
    ├── utils/                  
    │   └── functions.py

    ├── notebooks/               
    │   └── predict_plot_sample.py

    ├── models/                 
    │   └── NN_model.pth

<!-- ├── results/                # Resultados generados por los modelos y gráficas
│   ├── output.csv
│   └── feature_importance.png
 -->
1. **Standard Methods**
2. **ML Methods**

Each section contains scripts for data processing, model training, and evaluation. Below is a detailed breakdown of each step along with the corresponding scripts.

---
### DataFrame Creation
- **Description:**  
  These scripts generate dataframes based on the ICON-LES simulation (target) and radiances, reflectances simulations (features of the ML) stored in two NetCDF files. Configure the name files in the get_x_y_dataframes function of dataframe_all_data.py. We select the domain corresponding to "clouds".

  The data obtained is cleaned and also it is apply standarization to features and target.
  It will create 5 dataframes corresponding to each fold of the cross-validation, taking into account the time step.
- **How to Run:**
   - **Step 1: Data Preparation (dataframe_all_data.py)**
     ```bash
     bash dataframe_all_data.sh
- **Scripts:**
  - `dataframe_all_data.py`
  - `dataframe_all_data.sh`

---
## Standard Methods

### 1. Correlation Analysis
- **Scripts:**
  - `correlation_joinplot.py`
  - `correlation_joinplot.sh`
- **Description:**  
  These scripts calculate and visualize correlations in the dataset using join plots.
- **How to Run:**
   - Run the correlation analysis:
     ```bash
     bash correlation_joinplot.sh (with the option type_analysis="correlation")
     ```


### 2. Partial Correlation
- **Scripts:**
  - `partial_corr.py`
  - `partial_corr.sh`
- **Description:**  
  These scripts compute partial correlations, allowing you to examine the relationship between variables while controlling for the effect of additional variables.
- **How to Run:**
   - Run the partial correlation analysis:
     ```bash
     bash partial_corr.sh
     ```


### 3. Mutual information
- **Scripts:**
  - `correlation_joinplot.py`
  - `correlation_joinplot.sh`
- **Description:**  
  These scripts compute mutual information, allowing you to examine the relationship between variables.
- **How to Run:**
   - Run the mutual information:
     ```bash
     bash correlation_joinplot.sh (with the option type_analysis="mutual_information")
     ```

---
## ML Methods
### 1. Model Training & Prediction
- **Description:**  
    TThis code allows us to train the different ML models with the different data frames of each fold.
- **Main Scripts:**
  - `train_predict_plot.sh`  
- **How to Run:**
   - **Step 2: Training, Prediction, and Plotting**
     ```bash
     bash train_predict_plot.sh  (select the option requiered process="train-predict-plot")
     ```
#### Model Training
- **Scripts:**
  - `inverse_model_NN.py`  
    *Note: For running the NN - This script assumes that dataframes have been prepared.*
    
  - `inverse_model_RF.py`  
    *Note: For RF and XGBoost. This script requires pre-prepared dataframes in the DataFrame Creation step. It also generate the feature importances*

#### Prediction and Plotting
- **Scripts:**
  - `get_prediction_test.py`
  - `plotting_prediction_target.py`
    *Description: This will create a NetCDF file with the prediction and then it is possible to plot the density and distribution of that sample.*

#### Comparison models plots
- **Scripts:**
  - `plot_compare_model_metrics.py`
  - `plots_results.sh`
- **Description:**  
  Generate plots to visualize the comparison of the metrics using all channels and the most importance channels.
- **How to Run:**
     ```bash
     bash plots_results.sh (choose the option type_plot="model_metrics")
     ```


### 2. Feature Importances
### For Random Forest (RF) & XGBoost:**
- **Description:**  
    Compute the shap values using RF and XGBoost models.
    RF_feature_importances.py
    RF_feature_importances.sh
    Additional:
      permutation_RF.py
      RF_permutation.sh
- **How to Run:**
   - **Step 3.1:**
     ```bash
     bash RF_feature_importances.sh
     ```
### For Neural Networks (NN):**
- **Description:**  
    Compute the shap values, permutation, and other methods for NN model.
- **Scripts:**
  - `NN_feature_importances.py`
  - `NN_feature_importances.sh`
- **How to Run:**
   - **Step 3.2:**
     ```bash
     bash NN_feature_importances.sh

## NN mean feature importances 
- **Scripts:**
  - `NN_join_heatmaps.py`
  - `plots_results.sh`
- **Description:**  
  Generate a csv a heatmap plots showing the mean of the feature importances along folds and methods.
- **How to Run:**
     ```bash
     bash plots_results.sh (choose the option type_plot="join_NN_feature_importance")

## Feature importances all methods 
- **Scripts:**
  - `join_heatmaps.py`
  - `plots_results.sh`
- **Description:**  
  Generate a csv and heatmaps for all the methods including standard methods.
  It will print the most important spectral channels for each model
- **How to Run:**
     ```bash
     bash plots_results.sh (choose the option type_plot="heatmap_along_methods", all_data="all_features")
     ```   

     
### 3. Feature Reduction
- **Scripts:**
  - `feature_reduction_all_methods.py`
  - `feature_reduction_all_methods.sh`
- **Description:**  
  Perform feature reduction to analyze how many features we can use and obtain a simplify model and enhance interpretability. Generate a csv with the performance vs feature
- **How to Run:**
     ```bash
     bash feature_reduction_all_methods.sh  
     ```
### feature reduction plot 
- **Scripts:**
  - `plot_features_vs_metric.py`
  - `plots_results.sh`
- **Description:**  
  Generate a plot showing the performance vs number of feature.
- **How to Run:**
     ```bash
     bash plots_results.sh (choose the option type_plot="feature_metrics")
     ```   


### 4. Most important feature importances plots
- **Scripts:**
  - `join_heatmaps.py`
  - `plots_results.sh`
- **Description:**  
  It will generate feature importance of the most important feature
- **How to Run:**
     ```bash
     bash plots_results.sh (choose the option type_plot="heatmap_along_methods", all_data="most_importances")
     ```   

---
### Predict new sample with the Model trained
- **Scripts:**
  - `predict_plot_sample.py`
  - `predict_plot_sample.ipynb`
  - `predict_plot_sample.sh`
- **Description:**  
  Given a new sample in a NetCDF file with the different spectral channels (raf_red_data). It will generate a prediction using the model selected. It will generate a new NetCDF with the prediction and histogram of the sample.
- **How to Run:**
     ```bash
     bash predict_plot_sample.sh
     ```   
    *Note: check predict_plot_sample.ipynb as an example of the prediction*

    



