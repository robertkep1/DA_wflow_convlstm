#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:22:46 2021

@author: robert
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# load history 
data_stacked_sep_1 = pd.read_csv('.../loss_hist_stacked_sep_1.csv', sep=',', names = ['loss', 'val_loss'])
data_stacked_sep_1 = data_stacked_sep_1.dropna()
data_stacked_sep_1 = data_stacked_sep_1.reset_index().drop(['index'], axis=1)

data_s_2 = pd.read_csv('.../loss_hist_stacked_2.csv', sep=',',names = ['loss', 'val_loss'])
data_s_2 = data_s_2.dropna()
data_s_2 = data_s_2.reset_index().drop(['index'], axis=1)


# define smoothing function for learning curve
def smoothing(data, weight = 0.9):
    scalar = data
    last = scalar[0]
    smoothed_list = []
    for point in scalar:
        smoothed_value = last*weight + (1-weight) *point
        smoothed_list.append(smoothed_value)
        last = smoothed_value
    return smoothed_list


# apply smoothing to loss
ss_1_train_loss = smoothing(data_stacked_sep_1['loss'].values)
ss_1_val_loss = smoothing(data_stacked_sep_1['val_loss'].values)

ss_1_train_loss = np.sqrt(ss_1_train_loss)
ss_1_val_loss = np.sqrt(ss_1_val_loss)

s_2_train_loss = smoothing(data_s_2['loss'].values)
s_2_val_loss = smoothing(data_s_2['val_loss'].values)

s_2_train_loss = np.sqrt(s_2_train_loss)
s_2_val_loss = np.sqrt(s_2_val_loss)


# create plots
fig, (ax1, ax2) = plt.subplots(2, figsize=(5,5), sharex=True)
plt.subplots_adjust(hspace =0.1)

ax1.plot(s_2_train_loss, label='training')
ax1.plot(s_2_val_loss, label='validation')
plt.legend()

ax2.plot(ss_1_train_loss, label='training')
ax2.plot(ss_1_val_loss, label='validation')
ax1.set_ylabel('RMSE Loss [mm]')
ax2.set_ylabel('RMSE Loss [mm]')
ax2.set_xlabel('Epoch [-]')

ax1.set_xlim(-1,100)

plt.legend()
plt.text(0.5, 0.85, 'S2', fontsize = 12, transform = ax1.transAxes)
plt.text(0.5, 0.85, 'SS1', fontsize = 12, transform = ax2.transAxes)


plt.savefig(".../test.png",bbox_inches='tight')