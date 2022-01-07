#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:25:58 2022

@author: chedozea
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import xarray as xr

import os


from tqdm import tqdm



def collect_arpege2D(workdir):

    # Data_paths
    path_train = workdir+'DATA_RAINFALL/Train/Train/'
    path_test = workdir+'DATA_RAINFALL/Test/Test/'
    
    
    # Retrieve Arpege Grid
    fname = path_train+'X_forecast/2D_arpege_20170214.nc'
    data = xr.open_dataset(fname)
    arpege_lat = np.array(data['latitude'])
    arpege_lon = np.array(data['longitude'])
    
    # Retrieve station coordinates
    stat_coords = pd.read_csv(workdir+"DATA_RAINFALL/Other/Other/stations_coordinates.csv")
    
    
    # Correspondance table between station coordinates and grid points
    arpg_lats = []
    arpg_lons = []
    arpg_ilat = []
    arpg_ilon = []
    for i in stat_coords.index:
        lat = stat_coords.loc[i, 'lat']
        lon = stat_coords.loc[i, 'lon']
    
        indx_lat = np.argmin(np.abs(arpege_lat - lat))
        indx_lon = np.argmin(np.abs(arpege_lon - lon))
    
        arpg_ilat += [indx_lat]
        arpg_ilon += [indx_lon]
    
        arpg_lats += [arpege_lat[indx_lat]]
        arpg_lons += [arpege_lon[indx_lon]]
    #print([arpg_lats, arpg_lons])

    ################################################################################
    #                                                                              #
    # Collecting X_forecast (train) at certain grid points and saving to .csv file #
    #                                                                              #
    ################################################################################
    

    nb_dates = 637
    nb_stations = 325
    features = ['ws','p3031', 'u10', 'v10', 't2m', 'd2m', 'r', 'tp', 'msl']
    
    
    i=0
    ARPEGE_2D = np.zeros((nb_dates*nb_stations*24, len(features) ))   # date, station, feature, time of day   
    dates_and_stats = np.empty((nb_dates*nb_stations*24,2), dtype='object')
    path_arpg_train = path_train+'2D_arpege_train'
    with tqdm(total=nb_dates, desc="Collecting X_forecast (training)", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for r,d,f in os.walk(path_arpg_train):
            if f!= []:
                for filename in f:
                    f_path = os.path.join(r,filename)
                    data = xr.open_dataset(f_path)
                    dates = np.tile((np.array(data['valid_time']))[:24],nb_stations)
                    stations = np.repeat(stat_coords['number_sta'],24)
                    dates_and_stats[i*(nb_stations*24) : (i+1)*(nb_stations*24) ,0] = dates
                    dates_and_stats[i*(nb_stations*24) : (i+1)*(nb_stations*24) ,1] = stations
                    for f in range(len(features)):
                        feat = features[f]
                        if feat == 'tp': # rainfall per hour instead of cumulated rainfall
                            a = np.array(data[feat])[:,arpg_ilat, arpg_ilon]
                            a[0,:] = 0
                            b = a[1:,:] - a[:-1,:]
                            ARPEGE_2D[i*(nb_stations*24) : (i+1)*(nb_stations*24),f] = (b.T).flatten()
                        else:
                            a = np.array(data[feat])[:24,arpg_ilat, arpg_ilon]
                            ARPEGE_2D[i*(nb_stations*24) : (i+1)*(nb_stations*24),f] = (a.T).flatten()
                    i+=1
                    pbar.update(1)
                
    X_arpege_2D= pd.DataFrame(dates_and_stats, columns= ['date', 'number_sta'])
    X_arpege_2D['date'] = pd.to_datetime(X_arpege_2D['date'])
    X_arpege_2D[features] = ARPEGE_2D
    print("Saving .csv file......")
    X_arpege_2D.to_csv(path_train+'2D_arpege_train.csv', sep = ',')
    print("Done.")
    
    
    
    ################################################################################
    #                                                                              #
    # Collecting X_forecast (test) at certain grid points and saving to .csv file  #
    #                                                                              #
    ################################################################################
    
    
    
    nb_dates = 363
    
    i=0
    ARPEGE_2D = np.zeros((nb_dates*nb_stations*24,9))   # date, station, feature, time of day   
    ARPEGE_2D[:] = np.nan
    dates_and_stats = np.empty((nb_dates*nb_stations*24,2), dtype='object')
    path_arpg_test = 'DATA_RAINFALL/Test/Test/2D_arpege_test'
    with tqdm(total=nb_dates, desc="Collecting X_forecast (test)", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar2:
        for r,d,f in os.walk(path_arpg_test):
            if f!= []:
                for filename in f:
                    f_path = os.path.join(r,filename)
                    data = xr.open_dataset(f_path)
                    dates = np.tile((np.array(data['Id']))[:24],nb_stations)
                    stations = np.repeat(stat_coords['number_sta'],24)
                    dates_and_stats[i*(nb_stations*24) : (i+1)*(nb_stations*24) ,0] = dates
                    dates_and_stats[i*(nb_stations*24) : (i+1)*(nb_stations*24) ,1] = stations
                    for f in range(len(features)):
                        feat = features[f]
                        if feat in list(data.data_vars):
                            if feat == 'tp':
                                a = np.array(data[feat])[:,arpg_ilat, arpg_ilon]
                                a[0,:] = 0
                                b = a[1:,:] - a[:-1,:]
                                ARPEGE_2D[i*(nb_stations*24) : (i+1)*(nb_stations*24),f] = (b.T).flatten()
                            else:
                                a = np.array(data[feat])[:24,arpg_ilat, arpg_ilon]
                                ARPEGE_2D[i*(nb_stations*24) : (i+1)*(nb_stations*24),f] = (a.T).flatten()
                        #print(f_path)
                    i+=1
                    pbar2.update(1)
                    
                for day in [137,351]: # days with no data but necessary for kaggle submission
                    date_ids = []
                    for hr in range(24):
                        date_ids += [ str(day)+"_"+str(hr)]
                    dates = np.tile(date_ids,nb_stations)
                    stations = np.repeat(stat_coords['number_sta'],24)
                    dates_and_stats[i*(nb_stations*24) : (i+1)*(nb_stations*24) ,0] = dates
                    dates_and_stats[i*(nb_stations*24) : (i+1)*(nb_stations*24) ,1] = stations
                    i+=1
                    pbar2.update(1)
    
    X_arpege_2D= pd.DataFrame(dates_and_stats, columns= ['date', 'number_sta'])
    X_arpege_2D[features] = ARPEGE_2D
    
    # Remplissage des dates sans données par la moyenne des colonnes
    means = np.nanmean(X_arpege_2D.iloc[:,2:].values, axis=0)
    means = pd.Series(means, index = X_arpege_2D.columns[2:])
    X_arpege_2D = X_arpege_2D.fillna(value = means, axis = 0)
    
    print("Saving .csv file......")
    X_arpege_2D.to_csv(path_test+'2D_arpege_test.csv', sep = ',')
    print("Done.")
    
    
    
if __name__ == "__main__":
    collect_arpege2D('/home/chedozea/5eANNEE/AI-Frameworks/Defi-IA_scratch/')