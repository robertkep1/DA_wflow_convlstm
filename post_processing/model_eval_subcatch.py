#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:29:24 2021

@author: robert
"""

import spotpy
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import netCDF4 as nc
import numpy.ma as ma

# change wd
os.chdir('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM')
dictionary_ol = {'6335117':'Altenbamberg_ol', '9316161':'Argenschwang_ol', '9316163':'Imsweiler_ol', '9316166':'Eschenau_ol', '9316170':'Boos_ol'}
dictionary_ss1 = {'6335117':'Altenbamberg_ss1', '9316161':'Argenschwang_ss1', '9316163':'Imsweiler_ss1', '9316166':'Eschenau_ss1', '9316170':'Boos_ss1'}
dictionary_s2 = {'6335117':'Altenbamberg_s2', '9316161':'Argenschwang_s2', '9316163':'Imsweiler_s2', '9316166':'Eschenau_s2', '9316170':'Boos_s2'}

# import simulated data
q_assim_ss1 = pd.read_csv('syn_exp/final/run_assim_ss1_with_q_obs_pert.csv', sep=r'\s*,\s*', index_col='date', engine='python')
q_assim_ss1 = q_assim_ss1.rename(columns = dictionary_ss1)
q_assim_s2 = pd.read_csv('syn_exp/final/run_assim_s2_with_q_obs_pert.csv', sep=r'\s*,\s*', index_col='date', engine='python')
q_assim_s2 = q_assim_s2.rename(columns = dictionary_s2)
q_sim_ol = pd.read_csv('syn_exp/final/q_true/q_true.csv',sep=r'\s*,\s*', engine='python')
q_sim_ol = q_sim_ol.rename(columns = dictionary_ol)
q_sim_ol = q_sim_ol[13514:-1]
q_sim_ol.index = q_assim_s2.index

# import observed data
q_obs_altenbamberg = pd.read_excel('q_obs/discharge_obs_distributed/Altenbamberg.xls', index_col = 'Datum')
q_obs_boos = pd.read_excel('q_obs/discharge_obs_distributed/Boos.xls', index_col = 'Datum')
q_obs_eschenau = pd.read_excel('q_obs/discharge_obs_distributed/Eschenau.xls', index_col = 'Datum')
q_obs_imsweiler = pd.read_excel('q_obs/discharge_obs_distributed/Imsweiler.xls', index_col = 'Datum')

# merge dataframe based on existent dates
q_obs_sim_altenbamberg = pd.merge(q_obs_altenbamberg, q_sim_ol['Altenbamberg_ol'], right_index=True, left_index=True)
q_obs_sim_altenbamberg  = pd.merge(q_obs_sim_altenbamberg, q_assim_s2['Altenbamberg_s2'], right_index=True, left_index=True)
q_obs_sim_altenbamberg  = pd.merge(q_obs_sim_altenbamberg, q_assim_ss1['Altenbamberg_ss1'], right_index=True, left_index=True)

q_obs_sim_imsweiler = pd.merge(q_obs_imsweiler, q_sim_ol['Imsweiler_ol'], right_index=True, left_index=True)
q_obs_sim_imsweiler  = pd.merge(q_obs_sim_imsweiler, q_assim_s2['Imsweiler_s2'], right_index=True, left_index=True)
q_obs_sim_imsweiler  = pd.merge(q_obs_sim_imsweiler, q_assim_ss1['Imsweiler_ss1'], right_index=True, left_index=True)

q_obs_sim_eschenau = pd.merge(q_obs_eschenau, q_sim_ol['Eschenau_ol'], right_index=True, left_index=True)
q_obs_sim_eschenau = pd.merge(q_obs_sim_eschenau, q_assim_s2['Eschenau_s2'], right_index=True, left_index=True)
q_obs_sim_eschenau = pd.merge(q_obs_sim_eschenau, q_assim_ss1['Eschenau_ss1'], right_index=True, left_index=True)

q_obs_sim_boos = pd.merge(q_obs_boos, q_sim_ol['Boos_ol'], right_index=True, left_index=True)
q_obs_sim_boos  = pd.merge(q_obs_sim_boos, q_assim_s2['Boos_s2'], right_index=True, left_index=True)
q_obs_sim_boos  = pd.merge(q_obs_sim_boos, q_assim_ss1['Boos_ss1'], right_index=True, left_index=True)

# calculate error metrics
error_metrics = pd.DataFrame(columns = ['Altenbamberg_kge', 'Altenbamberg_nse', 'Altenbamberg_lognse', 'Altenbamberg_rmse',
                                        'Imsweiler_kge', 'Imsweiler_nse', 'Imsweiler_lognse', 'Imsweiler_rmse',
                                        'Eschenau_kge', 'Eschenau_nse', 'Eschenau_lognse', 'Eschenau_rmse',
                                        'Boos_kge', 'Boos_nse', 'Boos_lognse', 'Boos_rmse'], index= ('OL', 'SS1', 'S2'))

merge = [q_obs_sim_altenbamberg, q_obs_sim_imsweiler, q_obs_sim_eschenau, q_obs_sim_boos]

name_list = ['Altenbamberg', 'Imsweiler', 'Eschenau', 'Boos']
for i in np.arange(4):
    df = merge[i]
    columns = df.columns
    location = name_list[i]
    
    kge_ol = spotpy.objectivefunctions.kge(df['q_obs'], df[columns[1]])
    kge_s2 = spotpy.objectivefunctions.kge(df['q_obs'], df[columns[2]])
    kge_ss1 = spotpy.objectivefunctions.kge(df['q_obs'], df[columns[3]])
    
    nse_ol = spotpy.objectivefunctions.nashsutcliffe(df['q_obs'], df[columns[1]])
    nse_s2 = spotpy.objectivefunctions.nashsutcliffe(df['q_obs'], df[columns[2]])
    nse_ss1 = spotpy.objectivefunctions.nashsutcliffe(df['q_obs'], df[columns[3]])
    
    lognse_ol = spotpy.objectivefunctions.lognashsutcliffe(df['q_obs'], df[columns[1]])
    lognse_s2 = spotpy.objectivefunctions.lognashsutcliffe(df['q_obs'], df[columns[2]])
    lognse_ss1 = spotpy.objectivefunctions.lognashsutcliffe(df['q_obs'], df[columns[3]])
    
    rmse_ol = spotpy.objectivefunctions.rmse(df['q_obs'], df[columns[1]])
    rmse_s2 = spotpy.objectivefunctions.rmse(df['q_obs'], df[columns[2]])
    rmse_ss1 = spotpy.objectivefunctions.rmse(df['q_obs'], df[columns[3]])
    
    error_metrics[location + '_kge']['OL'] = kge_ol
    error_metrics[location + '_kge']['S2'] = kge_s2
    error_metrics[location + '_kge']['SS1'] = kge_ss1
    
    error_metrics[location + '_nse']['OL'] = nse_ol
    error_metrics[location + '_nse']['S2'] = nse_s2
    error_metrics[location + '_nse']['SS1'] = nse_ss1
    
    error_metrics[location + '_lognse']['OL'] = lognse_ol
    error_metrics[location + '_lognse']['S2'] = lognse_s2
    error_metrics[location + '_lognse']['SS1'] = lognse_ss1
    
    error_metrics[location + '_rmse']['OL'] = rmse_ol
    error_metrics[location + '_rmse']['S2'] = rmse_s2
    error_metrics[location + '_rmse']['SS1'] = rmse_ss1