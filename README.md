
# Project Overview

The folder "training_ml_models"  implements both standard statistical methods and machine learning (ML) methods for data analysis and prediction. The pipeline is divided into two main sections.

The folder "modis_prediction"  implements the use of the ML model with MODIS level 1 and also we process MODIS level 2.

    training_ml_models
    modis_prediction

1. training_ml_models
 Overview of the thermodynamic variables used for the ML models based on ICON-LES data and simulated MODIS . 
 
<figure>
  <figcaption>Overview of the thermodynamic variables used for the ML models based on ICON-LES data and simulated MODIS. </figcaption>
  <img src="img/table_ml_lwp_nd.png" alt="table_ml_lwp_nd" width="75%">
</figure>

<figure>
  <figcaption>Overview of diagram ML models based with inputs simulated MODIS and output ICON-LES data. </figcaption>
  <img src="img/diagrama_channels.png" alt="diagrama_channels" width="75%">
</figure>

1. modis_prediction
<figure>
  <figcaption>Overview of diagram using MODIS data with inputs MODIS and the trained ML (previous step).  </figcaption>
  <img src="img/block5.png" alt="block5" width="75%">
</figure>

 

 
<table>
  <tr>
    <td>
        <figure>
        <figcaption>Example prediction using simulated MODIS for L) </figcaption>
        <img src="img/icon_les_ml_lwp_scatter_comparison_distribution.png" alt="Image 1" width="80%" />
        </figure>
</td>
    <td>
        <figure>
        <figcaption>Example prediction using simulated MODIS for Nd) </figcaption>
        <img src="img/icon_les_ml_Nd_max_scatter_comparison_distribution.png" alt="Image 2" width="80%" />
         </figure>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
        <figure>
        <figcaption>Example prediction using MODIS level 1 for L) </figcaption>
        <img src="img/modis_NN_lwp_k_fold_0.png" alt="Image 1" width="80%" />
        </figure>
</td>
    <td>
        <figure>
        <figcaption>Example prediction using MODIS level 1 for Nd) </figcaption>
        <img src="img/modis_NN_Nd_max_k_fold_0.png" alt="Image 2" width="80%" />
         </figure>
    </td>
  </tr>
</table>
