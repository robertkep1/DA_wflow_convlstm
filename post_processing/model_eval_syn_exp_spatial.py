#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:34:23 2021

@author: robert
"""

import numpy as np
from tensorflow import keras
import pandas as pd
import datetime
import subprocess
import configparser
import netCDF4 as nc
import os
from tqdm import tqdm
from pcraster import report, setclone, numpy_operations, Scalar, aguila
from datetime import timedelta
import matplotlib.pyplot as plt
import spotpy
import seaborn as sns
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pcraster import readmap, pcr2numpy
import sklearn
import matplotlib.dates as datesplot
import matplotlib.ticker as ticker



# load synthetic exp data
q_true = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/q_true/q_true.csv', sep=r'\s*,\s*', engine = 'python')
q_true = q_true[13514:-1].reset_index()

q_obs_pert = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/q_obs_pert.csv', sep=r'\s*,\s*', engine = 'python', header=None)
q_obs_pert = q_obs_pert[11:-1]

q_assim_syn_ss1 = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/run_assim_ss1_with_qtrue.csv', index_col = 'date', sep=r'\s*,\s*', engine = 'python')
q_assim_syn_s2 = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/run_assim_s2_with_qtrue.csv', index_col = 'date', sep=r'\s*,\s*', engine = 'python')

# load syn exp states
states_true = nc.Dataset('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/q_true/states_true.nc')
states_true = states_true['ust_0_'][13515:]
mask = np.ma.getmask(states_true[0])
states_true = np.ma.getdata(states_true)

states_ass_syn_ss1 = np.load('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/statefile_assim_ss1_with_qtrue.npy')
states_ass_syn_s2 = np.load('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/statefile_assim_s2_with_qtrue.npy')

# load real world exp data
q_real_obs_pert = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/q_real_obs_pert.csv', sep=r'\s*,\s*', engine = 'python', header=None)
q_real_obs_pert = q_real_obs_pert[11:-1]

q_assim_realworld_ss1 = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/run_assim_ss1_with_q_obs_pert.csv', sep=r'\s*,\s*', engine = 'python')
q_assim_realworld_s2 = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/run_assim_s2_with_q_obs_pert.csv', sep=r'\s*,\s*', engine = 'python')

q_real_obs = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/q_obs/grolsheim_dietersheim_1.csv')
q_real_obs = q_real_obs[['Datum', 'Wert']]
q_real_obs['Datum'] = pd.to_datetime(q_real_obs['Datum'], format = '%-m/%-d/%Y', infer_datetime_format = True)
q_real_obs = q_real_obs.set_index(['Datum'])
q_real_obs = q_real_obs.loc['2016-01-02':'2019-12-31']
q_real_obs = q_real_obs['Wert'].to_numpy()

# load pert forcing data
forcing_pert = nc.Dataset('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/wflow_sbm/Nahe/inmaps/forcing-1979_2019_biased.nc')
temperature = forcing_pert['temperature'][13515:]
mask_long = np.ma.getmask(temperature)
temperature = np.ma.getdata(temperature)
precip = forcing_pert['precipitation'][13515:]
precip = np.ma.getdata(precip)

# analyse results
results_grolsheim = pd.DataFrame(index = q_assim_syn_ss1.index)
results_grolsheim['ass_ss1_syn'] = q_assim_syn_ss1['6335115']
results_grolsheim['ass_s2_syn'] = q_assim_syn_s2['6335115']
results_grolsheim['q_true'] = q_true['6335115'].values
results_grolsheim['q_obs_pert'] = q_obs_pert.values
results_grolsheim['q_real_obs_pert'] = q_real_obs_pert.values
results_grolsheim['ass_ss1_real'] = q_assim_realworld_ss1['6335115'].values
results_grolsheim['ass_s2_real'] = q_assim_realworld_s2['6335115'].values
results_grolsheim['q_real_obs'] = q_real_obs

res = results_grolsheim.sort_values(by = 'q_true')

# evaluate syn experiment
rmse_syn_ol = spotpy.objectivefunctions.rmse(results_grolsheim.q_true, results_grolsheim.q_obs_pert)
rmse_syn_ss1 = spotpy.objectivefunctions.rmse(results_grolsheim.q_true, results_grolsheim.ass_ss1_syn)
rmse_syn_s2 = spotpy.objectivefunctions.rmse(results_grolsheim.q_true, results_grolsheim.ass_s2_syn)

kge_syn_ol, r_syn_ol, alpha_syn_ol, beta_syn_ol = spotpy.objectivefunctions.kge(results_grolsheim.q_true, results_grolsheim.q_obs_pert, return_all=True)
kge_syn_ss1, r_syn_ss1, alpha_syn_ss1, beta_syn_ss1 = spotpy.objectivefunctions.kge(results_grolsheim.q_true, results_grolsheim.ass_ss1_syn, return_all=True)
kge_syn_s2, r_syn_s2, alpha_syn_s2, beta_syn_s2 = spotpy.objectivefunctions.kge(results_grolsheim.q_true, results_grolsheim.ass_s2_syn, return_all=True)

nse_syn_ol= spotpy.objectivefunctions.nashsutcliffe(results_grolsheim.q_true, results_grolsheim.q_obs_pert)
nse_syn_ss1 = spotpy.objectivefunctions.nashsutcliffe(results_grolsheim.q_true, results_grolsheim.ass_ss1_syn)
nse_syn_s2 = spotpy.objectivefunctions.nashsutcliffe(results_grolsheim.q_true, results_grolsheim.ass_s2_syn)

lognse_syn_ol= spotpy.objectivefunctions.lognashsutcliffe(results_grolsheim.q_true, results_grolsheim.q_obs_pert)
lognse_syn_ss1 = spotpy.objectivefunctions.lognashsutcliffe(results_grolsheim.q_true, results_grolsheim.ass_ss1_syn)
lognse_syn_s2 = spotpy.objectivefunctions.lognashsutcliffe(results_grolsheim.q_true, results_grolsheim.ass_s2_syn)

def pct_bias_fhv(q_obs, q_sim):
    return np.sum((q_sim - q_obs))/ np.sum(q_obs) *100

data_sorted_syn = results_grolsheim.sort_values(by='q_obs_pert')
data_sorted_80_syn = data_sorted_syn[1168:]
fhv_ol_syn = pct_bias_fhv(data_sorted_80_syn.q_true, data_sorted_80_syn.q_obs_pert)
fhv_s2_syn = pct_bias_fhv(data_sorted_80_syn.q_true, data_sorted_80_syn.ass_s2_syn)
fhv_ss1_syn = pct_bias_fhv(data_sorted_80_syn.q_true, data_sorted_80_syn.ass_ss1_syn)


# evaluate real world experiment
rmse_real_world_ol = spotpy.objectivefunctions.rmse(results_grolsheim.q_real_obs, results_grolsheim.q_true)
rmse_real_world_ss1 = spotpy.objectivefunctions.rmse(results_grolsheim.q_real_obs, results_grolsheim.ass_ss1_real)
rmse_real_world_s2 = spotpy.objectivefunctions.rmse(results_grolsheim.q_true, results_grolsheim.ass_s2_real)

kge_real_world_ol, r_real_world_ol, alpha_real_world_ol, beta_real_world_ol = spotpy.objectivefunctions.kge(results_grolsheim.q_real_obs, results_grolsheim.q_true, return_all=True)
kge_real_world_ss1, r_real_world_ss1, alpha_real_world_ss1, beta_real_world_ss1 = spotpy.objectivefunctions.kge(results_grolsheim.q_real_obs, results_grolsheim.ass_ss1_real, return_all=True)
kge_real_world_s2, r_real_world_s2, alpha_real_world_s2, beta_real_world_s2 = spotpy.objectivefunctions.kge(results_grolsheim.q_true, results_grolsheim.ass_s2_real, return_all=True)

nse_real_world_ol = spotpy.objectivefunctions.nashsutcliffe(results_grolsheim.q_real_obs, results_grolsheim.q_true)
nse_real_world_ss1 = spotpy.objectivefunctions.nashsutcliffe(results_grolsheim.q_real_obs, results_grolsheim.ass_ss1_real)
nse_real_world_s2 = spotpy.objectivefunctions.nashsutcliffe(results_grolsheim.q_true, results_grolsheim.ass_s2_real)

lognse_real_world_ol = spotpy.objectivefunctions.lognashsutcliffe(results_grolsheim.q_real_obs, results_grolsheim.q_true)
lognse_real_world_ss1 = spotpy.objectivefunctions.lognashsutcliffe(results_grolsheim.q_real_obs, results_grolsheim.ass_ss1_real)
lognse_real_world_s2 = spotpy.objectivefunctions.lognashsutcliffe(results_grolsheim.q_true, results_grolsheim.ass_s2_real)



# analyse high flow
# lr_ol = linear_model.LinearRegression().fit(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_sim_ol.values.reshape(1, -1))
r2_ol = r2_score(results_grolsheim.q_real_obs_pert.values, results_grolsheim.q_true.values)
# r2_s2 = r2_score(results_grolsheim.q_real_obs_pert.values, _obs_sim_test.q_ass_s2.values)
r2_ss1 = r2_score(results_grolsheim.q_real_obs_pert.values, results_grolsheim.ass_ss1_real.values)


linregress_ol= stats.linregress(results_grolsheim.q_real_obs_pert.values, results_grolsheim.q_true.values)
# linregress_s2 = stats.linregress(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_ass_s2.values)
linregress_ss1 = stats.linregress(results_grolsheim.q_real_obs_pert.values, results_grolsheim.ass_ss1_real.values)


fig, axs = plt.subplots(1, 3, figsize = (10, 3))

axs[0].plot(results_grolsheim.q_real_obs_pert.values, results_grolsheim.q_true.values, '.')
axs[0].plot(results_grolsheim.q_real_obs_pert.values, linregress_ol.intercept + linregress_ol.slope*results_grolsheim.q_real_obs_pert.values, 'r')
axs[0].plot([0,450], [0,450], linewidth=0.8, color = 'k', linestyle = '--')
axs[0].set_ylim((0, 450))
axs[0].set_xlim((0, 450))
axs[0].text(360, 340, '1:1 line', rotation=45)
axs[0].text(20, 390, 'Open Loop', weight = 'bold', fontsize = 12)
axs[0].annotate('$k_0 = 0.73$', xy=(350, 250), xytext=(320, 160), fontsize = 9,
            arrowprops=dict(facecolor='black', arrowstyle = '->'))


axs[1].plot(results_grolsheim.q_real_obs_pert.values, results_grolsheim.ass_ss1_real.values, '.')
axs[1].plot(results_grolsheim.q_real_obs_pert.values, linregress_ss1.intercept + linregress_ss1.slope * results_grolsheim.q_real_obs_pert.values, 'r', linestyle ='solid')
axs[1].plot([0,450], [0,450], linewidth=0.8, color = 'k', linestyle = '--')
axs[1].set_ylim((0, 450))
axs[1].set_xlim((0, 450))
axs[1].text(360, 340, '1:1 line', rotation=45)
axs[1].text(20, 390, 'SS1', weight = 'bold', fontsize = 12)
axs[1].annotate('$k_0 = 0.83$', xy=(340, 380), xytext=(150, 410), fontsize = 9,
            arrowprops=dict(facecolor='black', arrowstyle = '->'))

# axs[2].plot(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_ass_ss1.values, '.')
# axs[2].plot(q_obs_sim_test.q_obs.values, linregress_ss1.intercept + linregress_ss1.slope * q_obs_sim_test.q_obs.values, 'r')
# axs[2].plot([0,450], [0,450], linewidth=0.8, color = 'k', linestyle = '--')
# axs[2].set_ylim((0, 450))
# axs[2].set_xlim((0, 450))
# axs[2].text(360, 340, '1:1 line', rotation=45)
# axs[2].text(20, 390, 'SS1', weight = 'bold', fontsize = 12)
# axs[2].annotate('$k_0 = 0.8$', xy=(360, 283), xytext=(340, 180), fontsize = 9,
            # arrowprops=dict(facecolor='black', arrowstyle = '->'))


plt.setp(axs[-3], ylabel = 'simulated runoff [m³ s⁻¹]')
for ax in axs.flat:
    ax.set(xlabel = 'observed runoff [m³ s⁻¹]')


data_sorted_real_world = results_grolsheim.sort_values(by='q_real_obs_pert')
data_sorted_80_real_world = data_sorted_syn[1168:]
fhv_ol_real = pct_bias_fhv(data_sorted_80_real_world.q_real_obs_pert, data_sorted_80_real_world.q_true)
fhv_s2_real = pct_bias_fhv(data_sorted_80_real_world.q_real_obs_pert, data_sorted_80_real_world.ass_s2_real)
fhv_ss1_real = pct_bias_fhv(data_sorted_80_real_world.q_real_obs_pert, data_sorted_80_real_world.ass_ss1_real)


# analyse specific events for state updating

# first event: analysis on 10-02-2016
flood_event_2016_02_10 = results_grolsheim['2016-01-01':'2016-02-25']
flood_event_2016_02_10_syn = flood_event_2016_02_10[['q_true','ass_ss1_syn','ass_s2_syn']]
states_event_2016_02_10_true = states_true[38]
states_event_2016_02_10_ass_ss1 = states_ass_syn_ss1[38]
states_event_2016_02_10_ass_s2 = states_ass_syn_s2[38]

delta_state_event_2016_02_10_ss1 = states_event_2016_02_10_true - states_event_2016_02_10_ass_ss1
delta_state_event_2016_02_10_ss1[mask] = np.nan

delta_state_event_2016_02_10_s2 = states_event_2016_02_10_true - states_event_2016_02_10_ass_s2
delta_state_event_2016_02_10_s2[mask] = np.nan


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (14, 4.5), sharey = False, gridspec_kw={'width_ratios':[1.2,1,1,0.05]})
ax1.plot(flood_event_2016_02_10_syn.index, flood_event_2016_02_10.q_true, 'b',
         flood_event_2016_02_10_syn.index, flood_event_2016_02_10.ass_ss1_syn, 'r',
         flood_event_2016_02_10_syn.index, flood_event_2016_02_10.ass_s2_syn, 'forestgreen', linewidth = 1.2)
ax1.axvspan(28,38,alpha=0.2)
date = flood_event_2016_02_10.index.astype('O')
ax1.xaxis.set_major_locator(datesplot.MonthLocator())
ax1.xaxis.set_minor_locator(datesplot.MonthLocator(bymonthday=16))
ax1.xaxis.set_major_formatter(ticker.NullFormatter())
ax1.xaxis.set_minor_formatter(datesplot.DateFormatter('%b'))

for tick in ax1.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')
ax1.axvline(38, linestyle = '--', color = 'k', linewidth = 1.1)
ax1.set_ylabel('Discharge [m³s$^{-1}$]', fontsize = 12)
ax1.set_xlabel('Date')
ax1.set_xlim(0, 52)
ax1.text(43, 420, '2016', fontsize = 13)
ax1.text(38.5, 25, '09-02-2016')
ax1.legend(['$Q_{true}$', 'Assim SS1','Assim S2'])
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='right', fontsize = 14)


ax2.get_shared_y_axes().join(ax3)
g1 = sns.heatmap(delta_state_event_2016_02_10_ss1 , cmap = 'coolwarm', center = 0, ax = ax2, vmin = np.nanmin((np.nanmin(delta_state_event_2016_02_10_ss1), np.nanmin(delta_state_event_2016_02_10_s2))), vmax = np.nanmax((np.nanmax(delta_state_event_2016_02_10_ss1), np.nanmax(delta_state_event_2016_02_10_s2))), cbar=False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('SS1', fontweight = 'bold', fontsize = 18)
g2 = sns.heatmap(delta_state_event_2016_02_10_s2 , cmap = 'coolwarm', center=0,  ax = ax3, vmin = np.nanmin((np.nanmin(delta_state_event_2016_02_10_ss1), np.nanmin(delta_state_event_2016_02_10_s2))), vmax = np.nanmax((np.nanmax(delta_state_event_2016_02_10_ss1), np.nanmax(delta_state_event_2016_02_10_s2))), cbar_ax = ax4)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel('S2', fontweight = 'bold', fontsize = 18)
ax4.set_xlabel('[mm]', fontsize = 10)
plt.tight_layout()


# second event: analysis on 05-01-2018
flood_event_2018_01_05 = results_grolsheim['2017-12-05':'2018-01-20']
flood_event_2018_01_05_syn = flood_event_2018_01_05[['q_true','ass_ss1_syn','ass_s2_syn']]
states_event_2018_01_05_true = states_true[730] 
states_event_2018_01_05_ass_ss1 = states_ass_syn_ss1[730]
states_event_2018_01_05_ass_s2 = states_ass_syn_s2[730]

delta_state_event_2018_01_05_ss1 = states_event_2018_01_05_true - states_event_2018_01_05_ass_ss1
delta_state_event_2018_01_05_ss1[mask] = np.nan

delta_state_event_2018_01_05_s2 = states_event_2018_01_05_true - states_event_2018_01_05_ass_s2
delta_state_event_2018_01_05_s2[mask] = np.nan

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (14, 4.5), sharey = False, gridspec_kw={'width_ratios':[1.2,1,1,0.05]})
ax1.plot(flood_event_2018_01_05_syn.index, flood_event_2018_01_05_syn.q_true, 'b',
         flood_event_2018_01_05_syn.index, flood_event_2018_01_05_syn.ass_ss1_syn, 'r',
         flood_event_2018_01_05_syn.index, flood_event_2018_01_05_syn.ass_s2_syn, 'forestgreen', linewidth = 1.2)
date = flood_event_2018_01_05_syn.index.astype('O')
ax1.xaxis.set_major_locator(datesplot.MonthLocator())
ax1.xaxis.set_minor_locator(datesplot.MonthLocator(bymonthday=8))
ax1.xaxis.set_major_formatter(ticker.NullFormatter())
ax1.xaxis.set_minor_formatter(datesplot.DateFormatter('%b'))
ax1.set_xticklabels(['Dec', 'Jan'])

ax1.axvspan(17,27,alpha=0.2)
ax1.axvline(27, linestyle = '--', color = 'k', linewidth = 1.1)
ax1.set_ylabel('Discharge [m³s$^{-1}$]', fontsize = 12)
ax1.set_xlabel('Date')
ax1.set_xlim(0, 44)
ax1.text(32.5, 345, '2017-2018', fontsize = 13)
ax1.text(28, 40, '01-01-2018')
ax1.legend(['$Q_{true}$', 'Assim SS1','Assim S2'])
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='right', fontsize = 14)


ax2.get_shared_y_axes().join(ax3)
g1 = sns.heatmap(delta_state_event_2018_01_05_ss1, cmap = 'coolwarm', center = 0, ax = ax2, vmin = np.nanmin((np.nanmin(delta_state_event_2018_01_05_ss1), np.nanmin(delta_state_event_2018_01_05_s2))), vmax = np.nanmax((np.nanmax(delta_state_event_2018_01_05_ss1), np.nanmax(delta_state_event_2018_01_05_s2))), cbar=False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('SS1', fontweight = 'bold', fontsize = 18)
g2 = sns.heatmap(delta_state_event_2018_01_05_s2 , cmap = 'coolwarm', center=0,  ax = ax3, vmin = np.nanmin((np.nanmin(delta_state_event_2018_01_05_ss1), np.nanmin(delta_state_event_2018_01_05_s2))), vmax = np.nanmax((np.nanmax(delta_state_event_2018_01_05_ss1), np.nanmax(delta_state_event_2018_01_05_s2))), cbar_ax = ax4)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel('S2', fontweight = 'bold', fontsize = 18)
ax4.set_xlabel('[mm]', fontsize = 10)
plt.tight_layout()


# evaluate correlation between state difference and temperature

diff_syn_ss1 = states_true - states_ass_syn_ss1
diff_syn_ss1_1 = np.nanmean(diff_syn_ss1, axis = 0)
diff_syn_ss1 = diff_syn_ss1**2
diff_syn_ss1_mean = diff_syn_ss1.mean(axis = 1).mean(axis = 1)
states_true[mask_long] = np.nan
states_ass_syn_ss1[mask_long] = np.nan

def spatial_rmse(actual, pred):
    rmse_spatial = np.zeros((91, 134))
    actual[mask_long] = 0
    pred[mask_long] = 0
    for i in np.arange(134):
        for j in np.arange(91):
            rmse_spatial[j,i] = sklearn.metrics.mean_squared_error(actual[:,j,i], pred[:,j,i], squared = False)
    return rmse_spatial

spatial_rmse_syn_ss1 = spatial_rmse(states_true, diff_syn_ss1)

length = 1455
preceding_state = np.zeros(length)
for i in np.arange(length):
      preceding_state[i] = np.nanmean(precip[i:i+1])

dem = readmap('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/wflow_sbm/Nahe/staticmaps/wflow_dem.map')
dem = pcr2numpy(dem, 0)
# dem[mask] = 0

score = r2_score(dem.flatten(), diff_syn_ss1_1.flatten())
print(score)

min_temp_2016 = []
min_temp_2018 = []
for i in range(35):
    min_temp_2016.append(np.nanmin(temperature[38-35+i]))
    min_temp_2018.append(np.nanmin(temperature[730-35+i]))