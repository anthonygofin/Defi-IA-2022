# Defi-IA-2022
Défi IA 2022 Kaggle - Local prediction of rainfall using data measure on ground stations and results from MétéoFrance predictive models (Arpège_2D).

## Instructions to download the data:
- Create your account on Kaggle
- Get your API credentials 

" To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json "
  
  - Be sure to have the `kaggle` command installed using `pip install kaggle`
  - Run `dowload_data.sh` in your working directory. A ./DATA_RAINFALL/ directory will be created containing all the data from ground stations and Arpege.
  
## Preprocessing the data:
  
  - Install the `xarray` python library (plus `netcdf4` and  `h5netcdf` if necessary) to collect Arpege_2D data
  - Run preprocess_train.py (passing your working directory in argument) to remove nans from X_station_train and Y_train, to merge X_station with 2D_Arpgège and then reshape the training features to hourly features. Two files full_X_train and full_Y_train are created in the directory ./DATA_RAINFALL/Train/Train/. 
  
  
