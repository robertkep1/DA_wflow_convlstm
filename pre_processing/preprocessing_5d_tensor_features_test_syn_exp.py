
# coding: utf-8

# In[2]:


import netCDF4 as nc
from numpy import save
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling3D, LSTM, Reshape, ConvLSTM2D, concatenate,ZeroPadding2D
from keras import backend as K
from keras import utils
from tensorflow import keras
from keras.callbacks import History 
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model
import numpy.ma as ma
import matplotlib as mpl
import os
import joblib


# In[3]:


mask = np.load('mask.npy')
mask_inv = np.invert(mask)


# In[4]:


scaler_q_sim = joblib.load('scaler.save')
etp_min_max = pd.read_csv('min_max_etp.csv', index_col = 0)
evap_min_overall = etp_min_max['evap']['min']
evap_max_overall = etp_min_max['evap']['max']
temp_min_overall = etp_min_max['temp']['min']
temp_max_overall = etp_min_max['temp']['max']
precip_min_overall = etp_min_max['precip']['min']
precip_max_overall = etp_min_max['precip']['max']


# In[5]:


etp_min_max


# In[10]:


q_sim_biased = pd.read_csv('q_sim_biased.csv')
q_sim_biased = q_sim_biased[:]['6335115']
q_sim_biased = q_sim_biased[1:]
q_obs_biased = np.zeros(len(q_sim_biased))


# In[11]:


q_sim_biased.shape


# In[117]:


q_true = pd.read_csv('run.csv')
q_true = q_true['6335115'][13503:]
q_true = q_true.reset_index()


# In[128]:


plt.plot(q_measured[13504:13550],color='r')
plt.plot(q_true['6335115'][:50])


# In[127]:


q_measured[13504:].shape


# In[52]:


q_true


# In[71]:


q_true = q_true[13503:13650].reset_index()
q_true['6335115'].plot()
q_obs = q_obs[22858:23000].reset_index()
q_obs['Wert'].plot()


# In[29]:


q_sim_biased = pd.read_csv('q_sim_biased.csv')
q_sim_biased = q_sim_biased[:]['6335115']
q_sim_biased = q_sim_biased[1:]
q_obs_biased = np.zeros(len(q_sim_biased))
delta_q_list = []
for i in np.arange(1,len(q_sim_biased)):
    q_true_sq = (q_sim_biased[i])**2
    delta_q = np.random.normal(loc = 0, scale = (np.sqrt(0.0001*q_true_sq)))
    q_obs_biased[i] = q_sim_biased[i] + delta_q
    delta_q_list.append(delta_q)
np.savetxt('delta_q_sn.csv', delta_q_list)
q_obs_biased = np.expand_dims(q_obs_biased, 1)
q_obs_biased_sn = scaler_q_sim.transform(q_obs_biased)


# In[134]:


q_measured = pd.read_csv('grolsheim_dietersheim_1.csv')
q_measured = q_measured[['Datum', 'Wert']]
q_measured['Datum'] = pd.to_datetime(q_measured['Datum'], format = '%-m/%-d/%Y', infer_datetime_format = True)
q_measured = q_measured.set_index(['Datum'])
q_measured = q_measured.loc['1979-01-01':'2020-01-01']
q_true = q_measured['Wert'][13504:].to_numpy()


# In[135]:


#q_true = pd.read_csv('grolsheim_dietersheim_1.csv')
#q_true = q_true[13503:]['6335115']
#q_true = q_true.reset_index()
#q_true = q_true.drop(columns = 'index')
#q_true = q_true['6335115'].to_numpy()
q_obs_pert = np.zeros(len(q_true))
delta_q_list = []
for i in np.arange(len(q_true)):
    q_day = q_true[i]*0.1
    delta_q = np.random.normal(loc = 0, scale = q_day)
    q_obs_pert[i] = q_true[i] + delta_q
    delta_q_list.append(delta_q)
np.savetxt('delta_q_real_final.csv', delta_q_list)
np.savetxt('q_real_obs_pert.csv', q_obs_pert)
q_obs_pert = np.expand_dims(q_obs_pert, 1)
q_obs_pert = scaler_q_sim.transform(q_obs_pert)


# In[136]:


plt.plot(q_obs_pert,alpha=0.5)
#plt.plot(q_true,alpha=0.5)


# In[9]:


plt.plot(q_sim_biased)


# In[138]:


forcing_data = 'forcing-1979_2019_biased.nc'
forcing = nc.Dataset(forcing_data)

evap_orig = forcing['evapotranspiration'][:]
temp_orig = forcing['temperature'][:]
precip_orig = forcing['precipitation'][:]

mask = np.invert(ma.getmask(precip_orig))

evap = ma.getdata(evap_orig[13503:])
temp = ma.getdata(temp_orig[13503:])
precip = ma.getdata(precip_orig[13503:])

# # # ADD BIAS TO PRECIP
#precip = precip*1.1

evap_min = np.nanmin(evap, axis=(1, 2), keepdims=True)
evap_max = np.nanmax(evap, axis=(1, 2), keepdims=True)
#evap_min_overall = np.nanmin(evap_min) 
#evap_max_overall = np.nanmax(evap_max)
for i in np.arange(len(evap_min)):
    evap_min[i] = evap_min_overall
    evap_max[i] = evap_max_overall
evap = (evap - evap_min)/(evap_max-evap_min)
#evap = np.nan_to_num(evap, nan = fill_value)
for i in np.arange(evap.shape[0]):
    median = np.nanmedian(evap[i])
    evap[i] = np.nan_to_num(evap[i], nan=median)

temp_min = np.nanmin(temp, axis=(1, 2), keepdims=True)
temp_max = np.nanmax(temp, axis=(1, 2), keepdims=True)
#temp_min_overall = np.nanmin(temp_min)
#temp_max_overall = np.nanmax(temp_max)
for i in np.arange(len(temp_min)):
    temp_min[i] = temp_min_overall
    temp_max[i] = temp_max_overall
temp = (temp - temp_min)/(temp_max - temp_min)
#temp = np.nan_to_num(temp, nan = fill_value)
for i in np.arange(temp.shape[0]):
    median = np.nanmedian(temp[i])
    temp[i] = np.nan_to_num(temp[i], nan=median)

precip_min = np.nanmin(precip, axis=(1, 2), keepdims=True)
precip_max = np.nanmax(precip, axis=(1, 2), keepdims=True)
#precip_min_overall = np.nanmin(precip_min)
#precip_max_overall = np.nanmax(precip_max)
for i in np.arange(len(precip_min)):
    precip_min[i] = precip_min_overall
    precip_max[i] = precip_max_overall

num_precip = (precip - precip_min)
den_precip = (precip_max - precip_min)
precip = np.divide(num_precip, den_precip, out = np.zeros_like(num_precip), where = den_precip != 0)
precip[precip == mask] = np.nan
#precip = np.nan_to_num(precip, nan = fill_value)
for i in np.arange(precip.shape[0]):
    median = np.nanmedian(precip[i])
    precip[i] = np.nan_to_num(precip[i], nan=median)


# In[37]:


q_obs_biased_2d_sn = np.zeros_like(evap)
for i in np.arange(len(q_obs_biased_2d_sn)):
    q_obs_biased_2d_sn[i] = np.where(0, q_obs_biased_2d_sn[i], q_obs_biased_sn[i])


# In[11]:


q_obs_biased_2d_ln = np.zeros_like(evap)
for i in np.arange(len(q_obs_biased_2d_ln)):
    q_obs_biased_2d_ln[i] = np.where(0, q_obs_biased_2d_ln[i], q_obs_biased_ln[i])


# In[139]:


q_obs_biased_2d_final = np.zeros_like(evap)
for i in np.arange(len(q_obs_biased_2d_final)):
    q_obs_biased_2d_final[i] = np.where(0, q_obs_biased_2d_final[i], q_obs_pert[i])


# In[140]:


plt.plot(q_obs_biased_2d_final[:,0,60])


# In[141]:


stack = np.stack((evap, temp, precip, q_obs_biased_2d_final), axis = 3)


# In[142]:


for j in np.arange(10, 0, -1):
    #stack = np.stack((evap, temp, precip, q_measured_2d), axis = 3)
    #stack = stack[13514-j:,:,:,:]
    
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

    np.save(os.path.join('/lustre/backup/INDIVIDUAL/keppl001/feature_data_lags/test_long_all',
                         '5d_feature_tensor_lag_median_biased_ln' + str(k)), feature_tensor_5d)
    print(j)

