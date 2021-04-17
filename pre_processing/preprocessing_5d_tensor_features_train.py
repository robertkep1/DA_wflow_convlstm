
# coding: utf-8

# In[1]:


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


# In[10]:


dataset_q_sim = pd.read_csv('run.csv')

q_sim_orig = np.array(dataset_q_sim[['6335115']])[:]
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(q_sim_orig[:11323])

mask = np.load('mask.npy')
mask_inv = np.invert(mask)


# In[11]:


scaler_filename = "scaler.save"
joblib.dump(min_max_scaler, scaler_filename) 


# In[12]:


q_measured = pd.read_csv('grolsheim_dietersheim_1.csv')
q_measured = q_measured[['Datum', 'Wert']]
q_measured['Datum'] = pd.to_datetime(q_measured['Datum'], format = '%-m/%-d/%Y', infer_datetime_format = True)
q_measured = q_measured.set_index(['Datum'])
q_measured = q_measured.loc['1979-01-01':'2019-12-31']
q_measured = q_measured['Wert'].to_numpy()


# In[13]:


q_measured = np.expand_dims(q_measured, 1)
q_measured = min_max_scaler.transform(q_measured)


# In[14]:


q_sim = min_max_scaler.transform(q_sim_orig[:11323])
q_sim = np.expand_dims(q_sim, 1)


# In[15]:


forcing_data = 'forcing-1979_2019.nc'
forcing = nc.Dataset(forcing_data)

evap_orig = forcing['evapotranspiration'][:]
temp_orig = forcing['temperature'][:]
precip_orig = forcing['precipitation'][:]

mask = np.invert(ma.getmask(precip_orig))

evap = ma.getdata(evap_orig[:11323])
temp = ma.getdata(temp_orig[:11323])
precip = ma.getdata(precip_orig[:11323])

evap_min = np.nanmin(evap, axis=(1, 2), keepdims=True)
evap_max = np.nanmax(evap, axis=(1, 2), keepdims=True)
evap_min_overall = np.nanmin(evap_min) 
evap_max_overall = np.nanmax(evap_max)
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
temp_min_overall = np.nanmin(temp_min)
temp_max_overall = np.nanmax(temp_max)
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
precip_min_overall = np.nanmin(precip_min)
precip_max_overall = np.nanmax(precip_max)
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


# In[16]:


min_max_scaling_values = pd.DataFrame(data={'evap':(evap_min_overall, evap_max_overall), 'temp':(temp_min_overall, temp_max_overall), 'precip':(precip_min_overall, precip_max_overall)}, index=('min','max'))
min_max_scaling_values.to_csv("min_max_etp.csv", sep=',',index=True)


# In[26]:


q_measured_2d = np.zeros_like(evap)
for i in np.arange(len(q_measured)):
    q_measured_2d[i] = np.where(0, q_measured_2d[i], q_measured[i])


# In[17]:


q_sim_2d = np.zeros_like(evap)
for i in np.arange(len(q_sim)):
    q_sim_2d[i] = np.where(0, q_sim_2d[i], q_sim[i])


# In[34]:


plt.plot(q_sim_2d[:,0,60])
#plt.ylim((0,1))


# In[70]:


temp_orig[13513:13514,0,60]


# In[57]:


temp_orig.shape


# In[58]:


(train_features[0,11:12,0,60,1]*(temp_max_overall-temp_min_overall))+temp_min_overall


# In[40]:


min_max_scaler.inverse_transform(q_sim


# In[35]:


plt.plot(q_sim_orig[11:12])
#plt.ylim(20,26)


# In[30]:


stack = np.stack((evap, temp, precip, q_sim_2d), axis = 3)


# In[60]:


for j in np.arange(7, 0, -1):
    #stack = np.stack((evap, temp, precip, q_measured_2d), axis = 3)
    #stack = stack[13514-j:,:,:,:]
    
    lag = j
    samples = len(stack)
    time_steps = j + 1
    rows = 91
    cols = 134
    channels = 4
    
    feature_tensor_5d = np.zeros(shape = ((samples - lag), time_steps, rows, cols, channels))
    print(feature_tensor_5d.shape)
    for i in np.arange(samples - lag):
            for k in np.arange(time_steps):
                window = stack[i:i+time_steps]
                feature_tensor_5d[i] = window

    np.save(os.path.join('/lustre/backup/INDIVIDUAL/keppl001/feature_data_lags/train_long',
                         '5d_feature_tensor_lag_median_' + str(k)), feature_tensor_5d)
    print(j)


# In[32]:


train_features = np.load('/5d_feature_tensor_lag_median_11.npy')


# In[33]:


plt.plot(train_features[:,3,0,60,3])


# In[69]:


train_features[11311,:,0,60,0]


# In[67]:


train_features.shape


# In[50]:


min_max_scaler.inverse_transform(train_features[0:1,:,0,60,0])

