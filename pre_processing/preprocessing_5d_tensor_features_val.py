
# coding: utf-8


import netCDF4 as nc
import numpy as np
import pandas as pd
import numpy.ma as ma
import os
import joblib


mask = np.load('mask.npy')
mask_inv = np.invert(mask)

# # # LOAD MIN MAX FROM TRAINING PREPROCESSING TO BE USED FOR VALIDATION # # #
scaler_q_sim = joblib.load('scaler.save')
etp_min_max = pd.read_csv('min_max_etp.csv', index_col = 0)
evap_min_overall = etp_min_max['evap']['min']
evap_max_overall = etp_min_max['evap']['max']
temp_min_overall = etp_min_max['temp']['min']
temp_max_overall = etp_min_max['temp']['max']
precip_min_overall = etp_min_max['precip']['min']
precip_max_overall = etp_min_max['precip']['max']

# load simulated discharge
dataset_q_sim = pd.read_csv('run.csv')
q_sim_orig = np.array(dataset_q_sim[['6335115']])[:]
q_sim = scaler_q_sim.transform(q_sim_orig[13504:])
q_sim = np.expand_dims(q_sim, 1)


# # # PREPROCESS FORCING # # #
forcing_data = 'forcing-1979_2019.nc'
forcing = nc.Dataset(forcing_data)

evap_orig = forcing['evapotranspiration'][:]
temp_orig = forcing['temperature'][:]
precip_orig = forcing['precipitation'][:]

mask = np.invert(ma.getmask(precip_orig))

evap = ma.getdata(evap_orig[13503:])
temp = ma.getdata(temp_orig[13503:])
precip = ma.getdata(precip_orig[13503:])


evap_min = np.nanmin(evap, axis=(1, 2), keepdims=True)
evap_max = np.nanmax(evap, axis=(1, 2), keepdims=True)
for i in np.arange(len(evap_min)):
    evap_min[i] = evap_min_overall
    evap_max[i] = evap_max_overall
evap = (evap - evap_min)/(evap_max-evap_min)
for i in np.arange(evap.shape[0]):
    median = np.nanmedian(evap[i])
    evap[i] = np.nan_to_num(evap[i], nan=median)

temp_min = np.nanmin(temp, axis=(1, 2), keepdims=True)
temp_max = np.nanmax(temp, axis=(1, 2), keepdims=True)
for i in np.arange(len(temp_min)):
    temp_min[i] = temp_min_overall
    temp_max[i] = temp_max_overall
temp = (temp - temp_min)/(temp_max - temp_min)
for i in np.arange(temp.shape[0]):
    median = np.nanmedian(temp[i])
    temp[i] = np.nan_to_num(temp[i], nan=median)

precip_min = np.nanmin(precip, axis=(1, 2), keepdims=True)
precip_max = np.nanmax(precip, axis=(1, 2), keepdims=True)
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


# create grid 91x134 with sim discharge
q_sim_2d = np.zeros_like(evap)
for i in np.arange(len(q_sim)):
    q_sim_2d[i] = np.where(0, q_sim_2d[i], q_sim[i])

# merge all data
stack = np.stack((evap, temp, precip, q_sim_2d), axis = 3)


for j in np.arange(10, 0, -1):

    lag = j
    samples = len(stack)
    time_steps = j + 1
    rows = 91
    cols = 134
    channels = 4
    
    feature_tensor_5d = np.zeros(shape = ((samples- lag), time_steps, rows, cols, channels))
    print(feature_tensor_5d.shape)
    for i in np.arange(samples - lag):
            for k in np.arange(time_steps):
                window = stack[i:i+time_steps]
                feature_tensor_5d[i] = window

    np.save(os.path.join('PATH' + str(k)), feature_tensor_5d)
    print(j)

