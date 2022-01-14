#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:37:12 2022

@author: anthony
"""
from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import pairwise_distances
import swifter
import sys

def preprocess_test(workdir):
    print('Preprocessing has started.')
    
    # Utils paths
    path_utils = workdir+'utils/'
    
    sys.path.insert(0, path_utils )
    from utils_preprocessing_test import to_float,number_sta_row,get_day,ker_Gauss,avg_weights,fill_nan_row_test, create_id
    
    # Data_paths
    path_test = workdir+'DATA_RAINFALL/Test/Test/'
    
    # Import the original X_station_test
    print('-> Loading X_station_test.csv.')
    X_station_test=pd.read_csv(path_test+"X_station_test.csv",sep=",",header=0)
    
    #Get the day and the hour from orignal ids in the Dataframe 
    X_station_test['date_float']=X_station_test['Id'].apply(to_float)
    #Get number staion from orignal ids in the Dataframe 
    X_station_test['number_sta']=X_station_test.apply(number_sta_row,axis=1)
    #Create a colums in X_station_test that indicates the day 
    X_station_test['day']=X_station_test['date_float'].apply(get_day) 
    
    # Count hours for each combination station x day 
    count = X_station_test.groupby(['number_sta','day']).size()
    # Get the dataframe where there are not enough hours 
    not_24 = count[count!=24]
    # Get indexes
    index_not_24 = np.array(not_24.index)
    sta = [index_not_24[i][0] for i in range(len(index_not_24))]
    days = [index_not_24[i][1] for i in range(len(index_not_24))]
    
    # Get the entire dataset X_station_test where there are not enough hours 
    X_not_24 = X_station_test.loc[(X_station_test['day'].isin(days)) & (X_station_test['number_sta'].isin(sta))][['month','number_sta','day']]
    X_not_24  = X_not_24.drop_duplicates().reset_index()
    del X_not_24['index']  
    
    # Define an empty dataframe merged given ['month','number_sta','day']
    df = pd.merge(pd.DataFrame(index=range(len(X_not_24)),columns={'dd','hu','td','t','ff','precip'}),X_not_24, left_index=True, right_index=True)
    
    # Expand the dataset df to provide 24 hours 
    X = pd.concat([df]*24).sort_values(['number_sta','day']).reset_index()
    del X['index']
    X['index']=X.index
    X['Id']=X['number_sta'].astype('string') + '_' + X['day'].astype('string') + '_' + (X['index']%24).astype('string')
    X['date_float']= X['day'] + (X['index']%24)*10**-2
    del X['index']
    ids = X_station_test['Id'].values
    X2 = X[X['Id'].isin(ids)]
    X = X.drop(X2.index)
    
    X_concat = pd.concat([X,X_station_test])
    
    X_concat = X_concat.sort_values(['number_sta','date_float']).reset_index()
    del X_concat['index']
    
    # Filling nan 
    id_stations = pd.unique(X_concat['number_sta'])
    stations_coordinates=pd.read_csv(workdir+'DATA_RAINFALL/Other/Other/'+"stations_coordinates.csv",sep=",",header=0)
    
    #Compute the distance matrix betxeen each station 
    dist_mat =  pairwise_distances(stations_coordinates[stations_coordinates['number_sta'].isin(id_stations)][['lat','lon']])
    
    #Fill nans ================================================================================================================
    print('-> Filling nans in X_station_test.')
    X_concat = X_concat.swifter.apply(fill_nan_row_test, args = [X_concat, id_stations, dist_mat, 10], axis = 1, result_type='broadcast')
    
    #Save file and delete unseless columns
    del X_concat['day']
    del X_concat['date_float']
    
    # Get hourly features for X_station_test 
    print('-> Reformatting 2D_arpege_test.csv.')
    X_station_test=X_concat 
    X_station_test["new_id"]=X_station_test["Id"].apply(lambda x: x.split("_")[0]+"_"+x.split("_")[1])
    
    data = {'id': X_station_test['new_id'].unique()}     
    New_X_test=pd.DataFrame(data)
    
    NaN=np.nan
    for k in range(0,24):
        New_X_test["ff_%d" %(k)] = NaN
        New_X_test["t_%d" %(k)] = NaN
        New_X_test["td_%d" %(k)] = NaN
        New_X_test["hu_%d" %(k)] = NaN
        New_X_test["dd_%d" %(k)] = NaN
        New_X_test["precip_%d" %(k)] = NaN
        New_X_test["month"] = NaN
    
    for i in range(len(New_X_test)):
        print(i)
        for k in range(0,24):
            New_X_test.at[i,"ff_%d" %(k)] = X_station_test.at[i*24+k,"ff"]
            New_X_test.at[i,"t_%d" %(k)] = X_station_test.at[i*24+k,"t"]
            New_X_test.at[i,"td_%d" %(k)] = X_station_test.at[i*24+k,"td"]
            New_X_test.at[i,"hu_%d" %(k)] = X_station_test.at[i*24+k,"hu"]
            New_X_test.at[i,"dd_%d" %(k)] = X_station_test.at[i*24+k,"dd"]
            New_X_test.at[i,"month"] = X_station_test.at[i*24+k,"month"]
            New_X_test.at[i,"precip_%d" %(k)] = X_station_test.at[i*24+k,"precip"]
    
    # Get hourly features for X_arpege_2D 
    
    #The dataframe X_arpege2D, without nans and without hourly features !!! 
    print('-> Loading 2D_arpege_test.csv.')
    X_arpege2D_test=pd.read_csv(path_test+"2D_arpege_test.csv",sep=",",header=0)
    
    #The dataframe X_station_test, without nans but with hourly features 
    X_station_test= New_X_test 
    
    ### add an id column that contains station_day info from the Id and an hour column
    X_arpege2D_test["id"]=X_arpege2D_test.apply(create_id, axis=1)
    
    # Edit an new X_arpege2D with  hourly features 
    print('-> Reshaping 2D_arpege_test.csv.')
    data = {'id': X_arpege2D_test['id'].unique()}        
    New_X_arpege2D=pd.DataFrame(data)
    
    NaN=np.nan
    for k in range(0,24):
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
            print(i," /",len(New_X_arpege2D))
        for k in range(0,24):
            New_X_arpege2D.at[i,"ws_%d" %(k)] = X_arpege2D_test.at[i*24+k,"ws"]
            New_X_arpege2D.at[i,"p3031_%d" %(k)] = X_arpege2D_test.at[i*24+k,"p3031"]
            New_X_arpege2D.at[i,"u10_%d" %(k)] = X_arpege2D_test.at[i*24+k,"u10"]
            New_X_arpege2D.at[i,"v10_%d" %(k)] = X_arpege2D_test.at[i*24+k,"v10"]
            New_X_arpege2D.at[i,"t2m_%d" %(k)] = X_arpege2D_test.at[i*24+k,"t2m"]
            New_X_arpege2D.at[i,"d2m_%d" %(k)] = X_arpege2D_test.at[i*24+k,"d2m"]
            New_X_arpege2D.at[i,"r_%d" %(k)] = X_arpege2D_test.at[i*24+k,"r"]
            New_X_arpege2D.at[i,"tp_%d" %(k)] = X_arpege2D_test.at[i*24+k,"tp"]
            New_X_arpege2D.at[i,"msl_%d" %(k)] = X_arpege2D_test.at[i*24+k,"msl"]                                        
    
   
    # Merge data from X_arpege2D_test and X_station_test
    Full_X_test = New_X_arpege2D.merge(X_station_test, on ="id")
    
    #Save the final datasets (X_station_test with arpege data and hourly features )
    print("-> Saving  Full_X_test.csv in"+path_test)
    Full_X_test.to_csv(path_test + 'full_X_test.csv',sep=',')
    print("Done.")

if __name__ == "__main__":
    workdir = sys.argv[1]
    preprocess_test(workdir)


