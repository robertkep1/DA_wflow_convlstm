#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:06:27 2021

@author: robert
"""

import numpy as np
from tensorflow import keras
import pandas as pd
import datetime
import subprocess
import configparser
from datetime import timedelta
import netCDF4 as nc
from pcraster import report, setclone, numpy_operations, Scalar
import os
from tqdm import tqdm


# set wd
os.chdir('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM')

# load mask and invert
mask = np.load('saved_models/mask.npy')
mask = np.invert(mask[0])

# read forcing data
forcing = 'wflow_sbm/Nahe/inmaps_orig/forcing-1979_2019.nc'

# read state maps to initiate states for t-1
setclone('wflow_sbm/Nahe/run_1979-2019_2/outstate/LandRunoff.map')

# load state file
# state_file = 'wflow_sbm/Nahe/run_1979-2019_2/outmaps.nc'
# state_file = nc.Dataset(state_file)
state_file = 'syn_exp/final/q_true/states_true.nc'
state_file = nc.Dataset(state_file)


# load feature input for ConvLSTM
# features_convlstm_test = np.load('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/feature_lags_test_period/5d_feature_tensor_lag_median_10.npy')
features_convlstm_test = np.load('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/feature_lags_test_period/5d_feature_tensor_lag_median_biased_ln10_final_observations_pert.npy')


# create state dictioniary
state_dict = {'can':'CanopyStorage', 'lal':'WaterLevelL', 'lan':'LandRunoff', 'lev':'WaterLevelR',
              'run':'RiverRunoff', 'sat':'SatWaterDepth', 'sno':'Snow', 'snw':'SnowWater',
              'sub':'SubsurfaceFlow', 'tso':'TSoil', 'ust_0_':'UStoreLayerDepth_0',
              'ust_1_':'UStoreLayerDepth_1', 'ust_2_':'UStoreLayerDepth_2',
              'ust_3_':'UStoreLayerDepth_3', 'lwl':'LakeWaterLevel'}
state_keys = list(state_dict.keys())

test_split = [13514, 14975]
    

def data_assimilation(begin_date, n_days, perform_DA = False, ConvLSTM_arch = None):
    '''
    
    Keyword arguments:
    begin_date: Define date when DA starts with shape: (YYYY-MM-DD)
    n_days: Define for how many days DA is performed from begin date
    perform_DA: if TRUE: perform DA assimilation, if FALSE: no DA is performed, open loop run
    ConvLSTM_arch: Select architecture: 'stacked_sep_1' (for parallel model) or 'stacked_2' (for stacked model)
    '''
    
    begin_date = begin_date + str(' 00:00:00')
    
    q_val_modeled_6335115 = []
    q_val_modeled_6335117 = []
    q_val_modeled_9316159 = []
    q_val_modeled_9316160 = []
    q_val_modeled_9316161 = []
    q_val_modeled_9316163 = []
    q_val_modeled_9316166 = []
    q_val_modeled_9316168 = []
    q_val_modeled_9316170 = []
    date_list = []
    
    states = np.zeros((n_days, 91, 134))
    
    for i in tqdm(np.arange(n_days)):
        if i == 0:
            begin_date = datetime.strptime(begin_date, '%Y-%m-%d %H:%M:%S')
            begin_date = begin_date - timedelta(days = 1)#2
            start_time = str(begin_date)
        else:
            start_time = end_time
        
        for name in state_keys:
            global state_file
            state = state_file[name][test_split[0] - 1 + i] #-2
            # state = state_file[name][i]
            state = np.ma.getdata(state)
            state = numpy_operations.numpy2pcr(Scalar, state, -9999)
            # aguila(state)
            report(state, ('wflow_sbm/Nahe/instate/' 
                       + state_dict[name] + str('.map')))

                
        end_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_time = end_time + timedelta(days = 1)
        print_time = end_time + timedelta(days = 1)
        print_time = str(print_time)
        end_time = str(end_time)

        config = configparser.ConfigParser()
        config.optionxform = str
        config.read('wflow_sbm/Nahe/wflow_sbm.ini')
        config.set('run', 'starttime', start_time)
        config.set('run', 'endtime', end_time)

        with open('wflow_sbm/Nahe/wflow_sbm.ini', 'w') as configfile:
            config.write(configfile)

        if perform_DA == True:
            if ConvLSTM_arch == 'stacked_sep_1':
                model_stacked_sep_1 = keras.models.load_model('saved_models/model_stacked_sep_1.h5', compile = False)
                prediction = model_stacked_sep_1.predict(x = [features_convlstm_test[0+i:1+i,:,:,:,0:1],
                                                              features_convlstm_test[0+i:1+i,:,:,:,1:2],
                                                              features_convlstm_test[0+i:1+i,:,:,:,2:3],
                                                              features_convlstm_test[0+i:1+i,:,:,:,3:4]])   
            elif ConvLSTM_arch == 'stacked_2':
                model_stacked_2 = keras.models.load_model('saved_models/model_stacked_2.h5', compile = False)
                prediction = model_stacked_2.predict(x = features_convlstm_test[0+i:1+i])
                
            prediction = prediction[0,:,:,0]
            prediction[mask] = -9999
            state_ust_0 = numpy_operations.numpy2pcr(Scalar, prediction, -9999)
            report(state_ust_0, ('wflow_sbm/Nahe/instate/UStoreLayerDepth_0.map'))


        subprocess.run(['.../wflow_sbm.py', '-C',
                        'wflow_sbm/Nahe', '-R', 'da_run', '-f'])
                        
        
        
        states_act = '.../outmaps.nc'
        states_act = nc.Dataset(states_act)
        states_act = states_act['ust_0_'][:]
        states_act = np.ma.getdata(states_act)
        states[i] = states_act
        
        q_modeled = pd.read_csv('.../run.csv')
        
        
        q_val_modeled_6335115.append(q_modeled.loc[0]['6335115'])
        q_val_modeled_6335117.append(q_modeled.loc[0]['6335117'])
        q_val_modeled_9316159.append(q_modeled.loc[0]['9316159'])
        q_val_modeled_9316160.append(q_modeled.loc[0]['9316160'])
        q_val_modeled_9316161.append(q_modeled.loc[0]['9316161'])
        q_val_modeled_9316163.append(q_modeled.loc[0]['9316163'])
        q_val_modeled_9316166.append(q_modeled.loc[0]['9316166'])
        q_val_modeled_9316168.append(q_modeled.loc[0]['9316168'])
        q_val_modeled_9316170.append(q_modeled.loc[0]['9316170'])
        date_list.append(print_time)
    
    q_val_modeled_df = pd.DataFrame(columns = ['6335115', '6335117', '9316159', '9316160', '9316161', '9316163', '9316166', '9316168', '9316170'])
    q_val_modeled_df['6335115'] = q_val_modeled_6335115
    q_val_modeled_df['6335117'] = q_val_modeled_6335117
    q_val_modeled_df['9316159'] = q_val_modeled_9316159
    q_val_modeled_df['9316160'] = q_val_modeled_9316160
    q_val_modeled_df['9316161'] = q_val_modeled_9316161
    q_val_modeled_df['9316163'] = q_val_modeled_9316163
    q_val_modeled_df['9316166'] = q_val_modeled_9316166
    q_val_modeled_df['9316168'] = q_val_modeled_9316168
    q_val_modeled_df['9316170'] = q_val_modeled_9316170
    q_val_modeled_df['date'] = date_list

    np.savetxt(".../run_all.csv", q_val_modeled_df, delimiter=",", header = "6335115, 6335117, 9316159, 9316160, 9316161, 9316163, 9316166, 9316168, 9316170, date", comments = "", fmt="%s")
    
    np.save('.../statefile.npy', states)

