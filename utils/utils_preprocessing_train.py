#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:44:55 2022

@author: anthony
"""
from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

################################################################################
#                                                                              #
#                    Usefull functions for preprocessing                       #
#                                                                              #
################################################################################


#Create a colums with id 
def create_id(row): 
    number_sta = row['number_sta']
    day = row.loc['day']
    ids = str(number_sta)+"_"+str((row.loc['day']-dt.date(2016, 1, 2)).days)
    return ids


# Remove  missing hours using 
def remove_missing_hours(data):
    data_cpy =  data
    a =  data_cpy['new_id'].value_counts()
    a = a[a!=24].index
    a = np.asarray(a)
    
    data2= data[(data['new_id'].isin(a))]
    data=data.drop(data2.index)
    return data


################################################################################
#                                                                              #
#                         Functions to fill Nans in X_station_train            #
#                                                                              #
################################################################################

def ker_Gauss(d, L = 1./3): 
    return np.exp(-(d**2) / (L**2))
 
def avg_weights(df, weights, skipna = True):
    ncol = df.shape[1]
    w_series = pd.Series(weights, index = df.index)
    
    w_df = pd.concat([w_series]*ncol, axis = 1)
    w_df = w_df.set_axis(df.columns, axis =1)
    nan_pos = np.array(df.isnull())
    
    w_df= w_df.where(~nan_pos, other = 0)      
    w_nonan = w_df  
    denom = w_nonan.sum(axis = 0, skipna = skipna)
    w_nonan = w_nonan.divide(denom, axis=1)
    
    weighted_df = df.multiply(w_nonan)
    res = weighted_df.sum(axis = 0, skipna = skipna)
    return pd.Series(res, index = df.columns)
    
    
def fill_nan_row_train(row, df, id_stat, dist_mat, k_ngb):
    if row.isnull().any().item():
        t = row['date']
        stat = row['number_sta']
        
        indx_stat = np.where(id_stat==stat)[0][0]
        k_ngbs = np.argsort(dist_mat[indx_stat,:])[1:k_ngb+1]
        k_ngbs = np.sort(k_ngbs) # to be consistent with the dataframe where ids are sorted # AJOUT
    
        
        ngb_stations = id_stat[k_ngbs] 
        col_nan = row.index[row.isnull()]
        df_t = df.loc[df['date']==t]
        ngb_t = df_t.loc[df_t['number_sta'].isin(ngb_stations)]
         
        ngb_stations_t = ngb_t['number_sta']   # Not all neighbors have data at time t
        k_ngbs_t = k_ngbs[np.isin(ngb_stations, ngb_stations_t)]
        
        dist = dist_mat[indx_stat,k_ngbs_t] # AJOUT
        weights = ker_Gauss(dist) #AJOUT
        
        ngb_t = ngb_t[col_nan]
        #filling = ngb_t.mean(axis=0,skipna=True) # pd.Series
        filling = avg_weights(ngb_t, weights, skipna = True) # AJOUT
        #filling = pd.DataFrame([filling], columns = filling.index) # pd.DataFrame
        row_full = row.copy()
        #print("filling=",filling)
        row_full[col_nan] = filling
        return row_full
    else:
        return row