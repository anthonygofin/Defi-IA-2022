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
#                    Usefull functions to manipulate columns                   #
#                                                                              #
################################################################################

#Get the day and the hour from orignal ids in the Dataframe 
def to_float(string):
    _,a,b=string.split("_")  
    number=float(a)
    decimal=float(b)*10**(-2)
    return number+decimal

#Get number staion from orignal ids in the Dataframe 
def number_sta_row(row):
    t=row.loc['Id'][:8]
    #if row.name % 1e5 == 0:
        #print(row.name)
    return int(t)

#Get the day and the hour from orignal ids in the Dataframe 
def date_row_float(row):
    t=to_float(row.loc['Id'][9:])
    if row.name % 1e5 == 0:
        print(t)
    return t

def date_row(row):
    t=row.loc['Id'][9:]
    if row.name % 1e5 == 0:
        print(row.name)
    return t

#Create a colums in X_station_test that indicates the day 
def get_day(x): 
    str_x = str(x)
    index = str_x.find('.')
    x = int(str_x[:index])
    return x


#Create a colums with id 
def create_id(row): 
    number_sta = row['number_sta']
    day = row.loc['date'].split("_")[0]
    ids = str(number_sta)+"_"+day
    return ids

################################################################################
#                                                                              #
#                      Functions to fill Nans in X_station_test                #
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
    

def fill_nan_row_test(row, df, id_stat, dist_mat, k_ngb):
    if row.isnull().any().item():
        t = row['date_float']
        stat = row['number_sta']
        
        indx_stat = np.where(id_stat==stat)[0][0]
        k_ngbs = np.argsort(dist_mat[indx_stat,:])[1:k_ngb+1]
        k_ngbs = np.sort(k_ngbs) # so that weights are alined with the right neighbor 

        
        ngb_stations = id_stat[k_ngbs] 
        col_nan = row.index[row.isnull()]
        df_t = df.loc[df['date_float']==t]
        
        ngb_t = df_t.loc[df_t['number_sta'].isin(ngb_stations)]
         
        ngb_stations_t = ngb_t['number_sta']   # Not all neighbors have data at time t
        k_ngbs_t = k_ngbs[np.isin(ngb_stations, ngb_stations_t)]

        dist = dist_mat[indx_stat,k_ngbs_t]
        weights = ker_Gauss(dist)
        
        ngb_t = ngb_t[col_nan]
        filling = avg_weights(ngb_t, weights, skipna = True) 

        row_full = row.copy()
        row_full[col_nan] = filling
        return row_full
    else:
        return row