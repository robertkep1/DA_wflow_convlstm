
# coding: utf-8


import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy.ma as ma
import os
import joblib


# # # PRE-PROCESS SIMULATED DISCHARGE # # #

# load simulated discharge
dataset_q_sim = pd.read_csv('run.csv')

q_sim_orig = np.array(dataset_q_sim[['6335115']])[:]
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(q_sim_orig[1:11324])

# normalize simulated discharge
q_sim = min_max_scaler.transform(q_sim_orig[1:11324])
q_sim = np.expand_dims(q_sim, 1)

mask = np.load('mask.npy')
mask_inv = np.invert(mask)

# create min max scaler and save to be used later
scaler_filename = "scaler.save"
joblib.dump(min_max_scaler, scaler_filename) 


# # # PRE-PROCESS FORCING # # #

# load forcing data
forcing_data = 'forcing-1979_2019.nc'
forcing = nc.Dataset(forcing_data)

evap_orig = forcing['evapotranspiration'][:]
temp_orig = forcing['temperature'][:]
precip_orig = forcing['precipitation'][:]

mask = np.invert(ma.getmask(precip_orig))

# select forcing for train period
evap = ma.getdata(evap_orig[:11323])
temp = ma.getdata(temp_orig[:11323])
precip = ma.getdata(precip_orig[:11323])

# normalize and pad with median value at corresponding time step
evap_min = np.nanmin(evap, axis=(1, 2), keepdims=True)
evap_max = np.nanmax(evap, axis=(1, 2), keepdims=True)
evap_min_overall = np.nanmin(evap_min) 
evap_max_overall = np.nanmax(evap_max)
for i in np.arange(len(evap_min)):
    evap_min[i] = evap_min_overall
    evap_max[i] = evap_max_overall
evap = (evap - evap_min)/(evap_max-evap_min)
for i in np.arange(evap.shape[0]):
    median = np.nanmedian(evap[i])
    evap[i] = np.nan_to_num(evap[i], nan=median)

temp_min = np.nanmin(temp, axis=(1, 2), keepdims=True)
temp_max = np.nanmax(temp, axis=(1, 2), keepdims=True)
temp_min_overall = np.nanmin(temp_min)
temp_max_overall = np.nanmax(temp_max)
for i in np.arange(len(temp_min)):
    temp_min[i] = temp_min_overall
    temp_max[i] = temp_max_overall
temp = (temp - temp_min)/(temp_max - temp_min)
for i in np.arange(temp.shape[0]):
    median = np.nanmedian(temp[i])
    temp[i] = np.nan_to_num(temp[i], nan=median)

precip_min = np.nanmin(precip, axis=(1, 2), keepdims=True)
precip_max = np.nanmax(precip, axis=(1, 2), keepdims=True)
precip_min_overall = np.nanmin(precip_min)
precip_max_overall = np.nanmax(precip_max)
for i in np.arange(len(precip_min)):
    precip_min[i] = precip_min_overall
    precip_max[i] = precip_max_overall

num_precip = (precip - precip_min)
den_precip = (precip_max - precip_min)
precip = np.divide(num_precip, den_precip, out = np.zeros_like(num_precip), where = den_precip != 0)
precip[precip == mask] = np.nan
for i in np.arange(precip.shape[0]):
    median = np.nanmedian(precip[i])
    precip[i] = np.nan_to_num(precip[i], nan=median)

# save scaling values for later
min_max_scaling_values = pd.DataFrame(data={'evap':(evap_min_overall, evap_max_overall), 'temp':(temp_min_overall, temp_max_overall), 'precip':(precip_min_overall, precip_max_overall)}, index=('min','max'))
min_max_scaling_values.to_csv("min_max_etp.csv", sep=',',index=True)


q_sim_2d = np.zeros_like(evap)
for i in np.arange(len(q_sim)):
    q_sim_2d[i] = np.where(0, q_sim_2d[i], q_sim[i])


# # # MERGE ALL INPUT INTO ONE DATAFRAME # # #

# merge forcing and sim discharge into one dataframe
stack = np.stack((evap, temp, precip, q_sim_2d), axis = 3)

for j in np.arange(7, 0, -1):# select time window
    
    lag = j
    samples = len(stack)
    time_steps = j + 1 #choose time lag
    rows = 91
    cols = 134
    channels = 4
    
    feature_tensor_5d = np.zeros(shape = ((samples - lag), time_steps, rows, cols, channels))
    for i in np.arange(samples - lag):
            for k in np.arange(time_steps):
                window = stack[i:i+time_steps]
                feature_tensor_5d[i] = window

    np.save(os.path.join('PATH' + str(k)), feature_tensor_5d)

