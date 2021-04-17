import netCDF4 as nc
from tensorflow import keras
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


# import mask to define catchment boundaries
mask = np.load('mask.npy')
mask_inv = np.invert(mask)


# # # PREPROCESS LABELS, UPPER ZONE SOIL MOISTURE # # #

# load label data
fn = '/home/WUR/keppl001/MScThesis_env/data/outmaps.nc'
label_data = nc.Dataset(fn)

# select labels and apply mask to fill area outside catchment with nan
soil_moisture = label_data['ust_0_']
soil_moisture = np.ma.filled(soil_moisture, fill_value = np.nan)
soil_moisture = np.ma.getdata(soil_moisture)
soil_moisture[mask_inv] = np.nan

# loop through dataset to fill values outside catchment with median per timestep
for i in np.arange(len(soil_moisture)):
    median = np.nanmedian(soil_moisture[i])
    soil_moisture[i] = np.nan_to_num(soil_moisture[i], copy=False, nan = median)
labels = np.expand_dims(soil_moisture, axis = 1)

# define train and validation period
train_window = [0, 11323]
val_window = [11323, 13514]

lag = 10
    
# load train and validation features
features_train = 'PATH'
features_val = 'PATH' 
features_train = np.load(features_train)
features_val = np.load(features_val)

# reshape labels to fit shape (BATCH, HEIGHT, WIDTH, CHANNEL)
labels_train = labels[train_window[0] + lag:train_window[1]]
labels_val = labels[val_window[0] + lag:val_window[1]]
    
labels_train = np.reshape(labels_train, (features_train.shape[0], 91, 134, 1))
labels_val = np.reshape(labels_val, (features_val.shape[0], 91, 134, 1))


# # # MODEL TRAINING # # #

# load S2 model
model = load_model('models/model_stacked_2')

# define optimizer and lr
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss = 'mean_squared_error', optimizer = opt, metrics = 'mse')

# add model checkpoints to save model after improvement
mc = ModelCheckpoint('model_stacked_2.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

# train model
history = model.fit(x = features_train, y = labels_train, validation_data = (features_val, labels_val), epochs = 250,
                    batch_size = 2, verbose = 2, callbacks=[es, mc], shuffle = False, initial_epoch=29)

# save learning
array_hist = np.array(list(history.history.values())).transpose()
np.save('history_stacked_2.csv', array_hist)