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
import sys
from scipy.stats import boxcox
from scipy.special import boxcox1p
from sklearn.preprocessing import PowerTransformer
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

mask = np.load('mask.npy')
mask_inv = np.invert(mask)


fn = '/home/WUR/keppl001/MScThesis_env/data/outmaps.nc'
label_data = nc.Dataset(fn)
river_runoff = label_data['ust_0_']
river_runoff = np.ma.filled(river_runoff, fill_value = np.nan)
river_runoff = np.ma.getdata(river_runoff)
#print(river_runoff)
river_runoff[mask_inv] = np.nan
#print(river_runoff)
for i in np.arange(len(river_runoff)):
    median = np.nanmedian(river_runoff[i])
    river_runoff[i] = np.nan_to_num(river_runoff[i], copy=False, nan = median)
labels = np.expand_dims(river_runoff, axis = 1)

train_window = [0, 11323]
val_window = [11323, 13514]

lag = 10
    
features_train = '/lustre/backup/INDIVIDUAL/keppl001/feature_data_lags/train_long/5d_feature_tensor_lag_median_10.npy'
features_val = '/lustre/backup/INDIVIDUAL/keppl001/feature_data_lags/val_long/5d_feature_tensor_lag_median_10.npy' 
features_train = np.load(features_train)
features_val = np.load(features_val)

labels_train = labels[train_window[0] + lag:train_window[1]]
labels_val = labels[val_window[0] + lag:val_window[1]]
    
labels_train = np.reshape(labels_train, (features_train.shape[0], 91, 134, 1))
labels_val = np.reshape(labels_val, (features_val.shape[0], 91, 134, 1))

print(features_train.shape)
print(labels_train.shape)

model=load_model('models/model_stacked_1')

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss = 'mean_squared_error', optimizer = opt, metrics = 'mse')

mc = ModelCheckpoint('model_stacked_1.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

history = model.fit(x = features_train, y = labels_train, validation_data = (features_val, labels_val), epochs = 250,
                    batch_size = 1, verbose = 2, callbacks=[es, mc], shuffle = False)

array_hist = numpy.array(list(history.history.values())).transpose()
numpy.save('history_stacked_1.csv', array_hist)