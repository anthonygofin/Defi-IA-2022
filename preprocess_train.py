#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:41:25 2022

@author: anthony
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
from datetime import timezone
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import pairwise_distances
import swifter
import sys

def preprocess_train(workdir):
    print('Preprocessing has started.')
    
    # Utils paths
    path_utils = workdir+'utils/'
    
    sys.path.insert(0, path_utils )
    from utils_preprocessing_train import remove_missing_hours,ker_Gauss, avg_weights, fill_nan_row_train, create_id
    
    # Data_paths
    path_train = workdir + 'DATA_RAINFALL/Train/Train/'
    
    #Load the datasets 
    print('-> Loading X_station_train.csv.')
    X_station_train=pd.read_csv(path_train+"X_station_train.csv",sep=",",header=0)
    print('-> Loading Y_train.csv.')
    Y_train=pd.read_csv(path_train+'Y_train.csv',parse_dates=['date'],infer_datetime_format=True)
    
    #Copy X_station_train
    new_X_station_train = X_station_train
    
    #Compute the distance matrix between each station 
    id_stations = pd.unique(new_X_station_train['number_sta'])
    stations_coordinates=pd.read_csv(workdir+'DATA_RAINFALL/Other/Other/'+"stations_coordinates.csv",sep=",",header=0)
    dist_mat =  pairwise_distances(stations_coordinates[stations_coordinates['number_sta'].isin(id_stations)][['lat','lon']])
    
    #Fill nans 
    print('-> Filling nans in X_station_train.')
    filled_new_X_station_train = new_X_station_train.swifter.apply(fill_nan_row_train, args = [new_X_station_train, id_stations, dist_mat, 10], axis = 1, result_type='broadcast')
    
    #Add some columns to both Y-train and the filled X_station_train to simplify preprocessing steps 
    filled_new_X_station_train['date'] = pd.to_datetime(filled_new_X_station_train['date'])
    filled_new_X_station_train['month'] = filled_new_X_station_train['date'].dt.month
    filled_new_X_station_train['day'] = filled_new_X_station_train['date'].dt.date
    Y_train['date'] = pd.to_datetime(Y_train['date'])
    Y_train['day'] = Y_train['date'].dt.date
    
    # add a new_id column that contains station_day info from the Id and an hour column
    filled_new_X_station_train["new_id"]=filled_new_X_station_train["Id"].apply(lambda x: x.split("_")[0]+"_"+x.split("_")[1])
    filled_new_X_station_train["hour"]=filled_new_X_station_train["Id"].apply(lambda x: int(x.split("_")[2]))
    
    # Remove station which do not have 24 hours of observations 
    filled_new_X_station_train=remove_missing_hours(filled_new_X_station_train)
    
    # Update Y_train : 
    # Use the filled X_station_trainie and compute the sum of precipitation for each station and each day to fill Y_train.
    # Remind that date in X_train are from 2016-01-01 to 2017-12-30 (and not 2017-12-31)
    # In Y_train, date are 2016-01-02 to 2017-12-31
    
    print('-> Filling nans in Y_train.')
    import datetime
    #Compute ground_truth by summing the precipitations of the day 
    created_Y_train = filled_new_X_station_train[['number_sta','day','precip']].groupby(['number_sta','day'],as_index=False).sum('precip')[['day','number_sta','precip']]
    created_Y_train = created_Y_train.sort_values(by=['day','number_sta']).reset_index()
    del created_Y_train['index']
    
    # Delete rows where the date equals 2016-01-01
    created_Y_train = created_Y_train.drop(created_Y_train[created_Y_train.day == datetime.date(2016,1,1)].index)
    created_Y_train = created_Y_train.reset_index()
    del created_Y_train['index']
    
    # Rename columns 'precip' 
    created_Y_train = created_Y_train.rename(columns={'precip':'Ground_truth'})
    
    #Add rows from Y_train where the date equals 2017-12-31
    created_Y_train = pd.concat([created_Y_train, Y_train[Y_train.day == datetime.date(2017,12,31)][['day','number_sta','Ground_truth']]])
    
    #Add ids
    created_Y_train['Id'] = Y_train['Id']
    
    ## Ids to remove from X_station_train later
    bad_ids= np.array((created_Y_train[created_Y_train['Ground_truth'].isnull()])["Id"])
    
    #Delete data in created_Y_train where there are a missing data for Y_train
    created_Y_train = created_Y_train.rename(columns={'day':'date'})
    created_Y_train = created_Y_train.drop((created_Y_train[created_Y_train['Ground_truth'].isnull()]).index).reset_index()
    del created_Y_train['index']
    
    # Delete data in filled_new_X_station_train where there are a missing data for Y_train
    data2 = filled_new_X_station_train[(filled_new_X_station_train['new_id'].isin(bad_ids))]
    filled_new_X_station_train = filled_new_X_station_train.drop(data2.index)
    
    print('-> Loading 2D_arpege_train.csv.')
    #Load arpège2D
    X_arpege2D_train=pd.read_csv(path_train+"2D_arpege_train.csv",sep=",",header=0)
    
    #Update X_station_train, which is now the filled one
    X_station_train = filled_new_X_station_train
    
    #Update Y_train
    Y_train = created_Y_train
    
    #Delete useless rows 
    del X_arpege2D_train['Unnamed: 0']
    del X_station_train['hour']
    
    #Create date and day columns
    X_arpege2D_train['date'] = pd.to_datetime(X_arpege2D_train['date'])
    X_arpege2D_train['day'] = X_arpege2D_train['date'].dt.date
    
    #Sort X_arpege2D_train 
    X_arpege2D_train = X_arpege2D_train.sort_values(['number_sta','day','date']).reset_index(drop=True)
    
    #Add an id column that contains station_day info from the Id and an hour column
    X_arpege2D_train["id"]=X_arpege2D_train.apply(create_id, axis=1)
    X_arpege2D_train['hour'] = X_arpege2D_train['date'].dt.hour
    
    # Get hourly features 
    print('-> Reshaping 2D_arpege_train.csv.')
    data = {'id': X_arpege2D_train['id'].unique()}        
    New_X_arpege2D=pd.DataFrame(data)
    
    NaN=np.nan
    for k in (X_arpege2D_train['hour'].unique()):
        New_X_arpege2D["ws_%d" %(k)] = NaN
        New_X_arpege2D["p3031_%d" %(k)] = NaN
        New_X_arpege2D["u10_%d" %(k)] = NaN
        New_X_arpege2D["v10_%d" %(k)] = NaN
        New_X_arpege2D["t2m_%d" %(k)] = NaN
        New_X_arpege2D["d2m_%d" %(k)] = NaN
        New_X_arpege2D["r_%d" %(k)] = NaN
        New_X_arpege2D["tp_%d" %(k)] = NaN
        New_X_arpege2D["msl_%d" %(k)] = NaN
    
    for i in range(len(New_X_arpege2D)):
        if i%10000==0 : 
            print(i,'/', len(New_X_arpege2D) )
        for k in (X_arpege2D_train['hour'].unique()):
            New_X_arpege2D.at[i,"ws_%d" %(k)] = X_arpege2D_train.at[i*24+k,"ws"]
            New_X_arpege2D.at[i,"p3031_%d" %(k)] = X_arpege2D_train.at[i*24+k,"p3031"]
            New_X_arpege2D.at[i,"u10_%d" %(k)] = X_arpege2D_train.at[i*24+k,"u10"]
            New_X_arpege2D.at[i,"v10_%d" %(k)] = X_arpege2D_train.at[i*24+k,"v10"]
            New_X_arpege2D.at[i,"t2m_%d" %(k)] = X_arpege2D_train.at[i*24+k,"t2m"]
            New_X_arpege2D.at[i,"d2m_%d" %(k)] = X_arpege2D_train.at[i*24+k,"d2m"]
            New_X_arpege2D.at[i,"r_%d" %(k)] = X_arpege2D_train.at[i*24+k,"r"]
            New_X_arpege2D.at[i,"tp_%d" %(k)] = X_arpege2D_train.at[i*24+k,"tp"]
            New_X_arpege2D.at[i,"msl_%d" %(k)] = X_arpege2D_train.at[i*24+k,"msl"]
    
    # Merge data from X_arpege2D_train and X_station_train
    Full_X_train = New_X_arpege2D.merge(X_station_train, on ="id")
    
    #All the following cells udpate Y_train (to be consistant with Full_X_train)
    Y_train = Y_train.rename(columns={'Id':'id'})
    
    # Get a dataframe with both X_train and Y_train (aligned)
    X_Y = Y_train.merge(Full_X_train, on ="id")
    
    # Separate X from Y 
    Full_Y_train = X_Y.iloc[:,:4]
    Full_X_train = X_Y.iloc[:,3:]
    
    #Save the datasets 
    print('-> Saving Full_X_train.cvs in.'+path_train)
    print('-> Saving Full_Y_train.cvs in.'+path_train)
    Full_X_train.to_csv(path_train +'full_X_train.csv',sep=',')
    Full_Y_train.to_csv(path_train +'full_Y_train.csv',sep=',')
    print("Done.")

if __name__ == "__main__":
    workdir = sys.argv[1]
    preprocess_train(workdir)
