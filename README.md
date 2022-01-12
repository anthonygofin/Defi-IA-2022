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
  - Run `preprocess_train.py` (passing your working directory in argument) to fill nans from X_station_train and Y_train, to merge X_station_train with 2D_arpege_train and then reshape the training features to hourly features (all features at each hour). Two files full_X_train.csv and full_Y_train.csv are created in the directory ./DATA_RAINFALL/Train/Train/. Preprocessing the training set is time-consumming (approximately 6 hours). 
   - Run `preprocess_test.py` (passing your working directory in argument) to fill nans from X_station_test, to merge X_station_test with 2D_arpege_test and then reshape the features to hourly features. A file full_X_test.csv is created in the directory ./DATA_RAINFALL/Test/Test/. Preprocessing the test set could last for around 30 minutes. Note that Y_test is not provided by MeteoFrance. 
  
  
