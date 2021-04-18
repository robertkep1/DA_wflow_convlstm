#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:02:49 2021

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
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
# import matplotlib.gridspec as gridspec

# import data subcatchments
os.chdir('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM')
dictionary_ol = {'6335115':'Grolsheim_q_true', '6335117':'Altenbamberg_q_true', '9316161':'Argenschwang_q_true', '9316163':'Imsweiler_q_true', '9316166':'Eschenau_q_true', '9316170':'Boos_q_true'}
dictionary_ss1 = {'6335115':'Grolsheim_ss1','6335117':'Altenbamberg_ss1', '9316161':'Argenschwang_ss1', '9316163':'Imsweiler_ss1', '9316166':'Eschenau_ss1', '9316170':'Boos_ss1'}
dictionary_s2 = {'6335115':'Grolsheim_s2','6335117':'Altenbamberg_s2', '9316161':'Argenschwang_s2', '9316163':'Imsweiler_s2', '9316166':'Eschenau_s2', '9316170':'Boos_s2'}


# import simulated data for real world experiment
q_assim_ss1_q_real_obs = pd.read_csv('syn_exp/final/run_assim_ss1_with_q_obs_pert.csv', sep=r'\s*,\s*', index_col='date', engine='python')
q_assim_ss1_q_real_obs = q_assim_ss1_q_real_obs.rename(columns = dictionary_ss1)
q_assim_s2_q_real_obs = pd.read_csv('syn_exp/final/run_assim_s2_with_q_obs_pert.csv', sep=r'\s*,\s*', index_col='date', engine='python')
q_assim_s2_q_real_obs = q_assim_s2_q_real_obs.rename(columns = dictionary_s2)

# load syn exp states
states_true = nc.Dataset('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/q_true/states_true.nc')
states_true = states_true['ust_0_'][13515:]
mask = np.ma.getmask(states_true)
states_true = np.ma.getdata(states_true)
states_true[mask] = np.nan

states_ass_syn_ss1 = np.load('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/statefile_assim_ss1_with_qtrue.npy')
states_ass_syn_s2 = np.load('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/syn_exp/final/statefile_assim_s2_with_qtrue.npy')
states_ass_syn_ss1[mask] = np.nan
states_ass_syn_s2[mask] = np.nan

states = pd.DataFrame()

mean_ss1 = []
var_ss1 = []
mean_s2 = []
var_s2= []
mean_true = []
var_true = []

for i in np.arange(1460):
    mean1 = np.nanmean(states_true[i])
    mean_true.append(mean1)
    act_std1 = np.nanvar(states_true[i])
    var_true.append(act_std1)
for i in np.arange(1460):
    mean2 = np.nanmean(states_ass_syn_ss1[i])
    act_std2 = np.nanvar(states_ass_syn_ss1[i])
    mean_ss1.append(mean2)
    var_ss1.append(act_std2)
for i in np.arange(1460):
    mean3 = np.nanmean(states_ass_syn_s2[i])
    act_std3 = np.nanvar(states_ass_syn_s2[i])
    mean_s2.append(mean3)
    var_s2.append(act_std3)

states['true_mean'] = mean_true  
states['true_var'] = var_true
states['ss1_var'] = var_ss1
states['ss1_mean'] = mean_ss1
states['s2_var'] = var_s2
states['s2_mean'] = mean_s2

def percentage_change(col1,col2):
    return ((col2 - col1) / col1) * 100

def std_error_pct_chg(x_mean,x_var,y_mean,y_var):
    return 100* np.sqrt((y_var*x_mean**2 + x_var*y_mean**2)/x_mean**4)

states['pct_chg_ss1'] = percentage_change(states.true_mean, states.ss1_mean)
states['std_err_pct_chg_ss1'] = std_error_pct_chg(states.true_mean, states.true_var, states.ss1_mean, states.ss1_var)
states['pct_chg_s2'] = percentage_change(states.true_mean, states.s2_mean)
states['std_err_pct_chg_s2'] = std_error_pct_chg(states.true_mean, states.true_var, states.s2_mean, states.s2_var)


#plt.plot(np.arange(1460), states.pct_chg_s2, 'forestgreen', linewidth = 0.5)
#plt.fill_between(np.arange(1460), (states.pct_chg_s2-states.std_err_pct_chg_s2),
#                 (states.pct_chg_s2+states.std_err_pct_chg_s2), color='forestgreen', alpha=.1)
#plt.plot(np.arange(1460), states.pct_chg_ss1, 'r', linewidth = 0.5)
#plt.fill_between(np.arange(1460), (states.pct_chg_ss1-states.std_err_pct_chg_ss1),
#                 (states.pct_chg_ss1+states.std_err_pct_chg_ss1), color='r', alpha=.1)
#plt.ylim(-200,200)
plt.show()


# import observed data
q_obs_altenbamberg = pd.read_excel('q_obs/discharge_obs_distributed/Altenbamberg.xls', index_col = 'Datum')
q_obs_boos = pd.read_excel('q_obs/discharge_obs_distributed/Boos.xls', index_col = 'Datum')
q_obs_eschenau = pd.read_excel('q_obs/discharge_obs_distributed/Eschenau.xls', index_col = 'Datum')
q_obs_imsweiler = pd.read_excel('q_obs/discharge_obs_distributed/Imsweiler.xls', index_col = 'Datum')
q_real_obs_grolsheim = pd.read_csv('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM/q_obs/grolsheim_dietersheim_1.csv')
q_real_obs_grolsheim = q_real_obs_grolsheim[['Datum', 'Wert']]
q_real_obs_grolsheim['Datum'] = pd.to_datetime(q_real_obs_grolsheim['Datum'], format = '%-m/%-d/%Y', infer_datetime_format = True)
q_real_obs_grolsheim = q_real_obs_grolsheim.set_index(['Datum'])
q_real_obs_grolsheim = q_real_obs_grolsheim.loc['2016-01-02':'2019-12-31']
q_real_obs_grolsheim = q_real_obs_grolsheim.rename(columns = {'Wert':'q_obs'})
date = q_real_obs_grolsheim.index

q_true = pd.read_csv('syn_exp/final/q_true/q_true.csv', sep=r'\s*,\s*', engine='python')
q_true = q_true[13514:-1]
q_true['Date'] = date
q_true = q_true.set_index('Date')
q_true = q_true.rename(columns = dictionary_ol)

# merge dataframe based on existent dates
q_obs_sim_altenbamberg = pd.merge(q_obs_altenbamberg, q_true['Altenbamberg_q_true'], right_index=True, left_index=True)
q_obs_sim_altenbamberg  = pd.merge(q_obs_sim_altenbamberg, q_assim_s2_q_real_obs['Altenbamberg_s2'], right_index=True, left_index=True)
q_obs_sim_altenbamberg  = pd.merge(q_obs_sim_altenbamberg, q_assim_ss1_q_real_obs['Altenbamberg_ss1'], right_index=True, left_index=True)

q_obs_sim_imsweiler = pd.merge(q_obs_imsweiler, q_true['Imsweiler_q_true'], right_index=True, left_index=True)
q_obs_sim_imsweiler  = pd.merge(q_obs_sim_imsweiler, q_assim_s2_q_real_obs['Imsweiler_s2'], right_index=True, left_index=True)
q_obs_sim_imsweiler  = pd.merge(q_obs_sim_imsweiler, q_assim_ss1_q_real_obs['Imsweiler_ss1'], right_index=True, left_index=True)

q_obs_sim_eschenau = pd.merge(q_obs_eschenau, q_true['Eschenau_q_true'], right_index=True, left_index=True)
q_obs_sim_eschenau = pd.merge(q_obs_sim_eschenau, q_assim_s2_q_real_obs['Eschenau_s2'], right_index=True, left_index=True)
q_obs_sim_eschenau = pd.merge(q_obs_sim_eschenau, q_assim_ss1_q_real_obs['Eschenau_ss1'], right_index=True, left_index=True)

q_obs_sim_boos = pd.merge(q_obs_boos, q_true['Boos_q_true'], right_index=True, left_index=True)
q_obs_sim_boos  = pd.merge(q_obs_sim_boos, q_assim_s2_q_real_obs['Boos_s2'], right_index=True, left_index=True)
q_obs_sim_boos  = pd.merge(q_obs_sim_boos, q_assim_ss1_q_real_obs['Boos_ss1'], right_index=True, left_index=True)

q_obs_sim_grolsheim = pd.merge(q_real_obs_grolsheim, q_true['Grolsheim_q_true'], right_index=True, left_index=True)
q_obs_sim_grolsheim  = pd.merge(q_obs_sim_grolsheim, q_assim_s2_q_real_obs['Grolsheim_s2'], right_index=True, left_index=True)
q_obs_sim_grolsheim  = pd.merge(q_obs_sim_grolsheim, q_assim_ss1_q_real_obs['Grolsheim_ss1'], right_index=True, left_index=True)
#q_obs_sim_grolsheim = q_obs_sim_grolsheim.drop(q_obs_sim_grolsheim.index[1095:1134])


# # # SYNTHETIC EXPERIMENT


# calculate percentage difference

q_obs_sim_altenbamberg['pct_chg_qtrue'] = percentage_change(q_obs_sim_altenbamberg['q_obs'], q_obs_sim_altenbamberg['Altenbamberg_q_true'])
q_obs_sim_altenbamberg['pct_chg_ss1'] = percentage_change(q_obs_sim_altenbamberg['q_obs'], q_obs_sim_altenbamberg['Altenbamberg_ss1'])
q_obs_sim_altenbamberg['pct_chg_s2'] = percentage_change(q_obs_sim_altenbamberg['q_obs'], q_obs_sim_altenbamberg['Altenbamberg_s2'])

q_obs_sim_imsweiler['pct_chg_qtrue'] = percentage_change(q_obs_sim_imsweiler['q_obs'], q_obs_sim_imsweiler['Imsweiler_q_true'])
q_obs_sim_imsweiler['pct_chg_ss1'] = percentage_change(q_obs_sim_imsweiler['q_obs'], q_obs_sim_imsweiler['Imsweiler_ss1'])
q_obs_sim_imsweiler['pct_chg_s2'] = percentage_change(q_obs_sim_imsweiler['q_obs'], q_obs_sim_imsweiler['Imsweiler_s2'])

q_obs_sim_eschenau['pct_chg_qtrue'] = percentage_change(q_obs_sim_eschenau['q_obs'], q_obs_sim_eschenau['Eschenau_q_true'])
q_obs_sim_eschenau['pct_chg_ss1'] = percentage_change(q_obs_sim_eschenau['q_obs'], q_obs_sim_eschenau['Eschenau_ss1'])
q_obs_sim_eschenau['pct_chg_s2'] = percentage_change(q_obs_sim_eschenau['q_obs'], q_obs_sim_eschenau['Eschenau_s2'])

q_obs_sim_boos['pct_chg_qtrue'] = percentage_change(q_obs_sim_boos['q_obs'], q_obs_sim_boos['Boos_q_true'])
q_obs_sim_boos['pct_chg_ss1'] = percentage_change(q_obs_sim_boos['q_obs'], q_obs_sim_boos['Boos_ss1'])
q_obs_sim_boos['pct_chg_s2'] = percentage_change(q_obs_sim_boos['q_obs'], q_obs_sim_boos['Boos_s2'])

q_obs_sim_grolsheim['pct_chg_qtrue'] = percentage_change(q_obs_sim_grolsheim['q_obs'], q_obs_sim_grolsheim['Grolsheim_q_true'])
q_obs_sim_grolsheim['pct_chg_ss1'] = percentage_change(q_obs_sim_grolsheim['q_obs'], q_obs_sim_grolsheim['Grolsheim_ss1'])
q_obs_sim_grolsheim['pct_chg_s2'] = percentage_change(q_obs_sim_grolsheim['q_obs'], q_obs_sim_grolsheim['Grolsheim_s2'])


states.index = q_obs_sim_grolsheim.index

# create plots

idx = pd.date_range(min(q_obs_sim_altenbamberg.index.date), max(q_obs_sim_altenbamberg.index.date))
q_obs_sim_altenbamberg = q_obs_sim_altenbamberg.reindex(idx, fill_value = np.nan)
q_obs_sim_boos = q_obs_sim_boos.reindex(idx, fill_value = np.nan)
q_obs_sim_eschenau = q_obs_sim_eschenau.reindex(idx, fill_value = np.nan)
q_obs_sim_imsweiler = q_obs_sim_imsweiler.reindex(idx, fill_value = np.nan)

date_form = DateFormatter("%b %y")
# q_obs_sim_grolsheim.index = pd.to_datetime(q_assim_s2.index)
#dates = q_assim_s2.index

gridspec1 = dict(hspace=0.0, wspace =0.13, height_ratios=[0.6, 1.1, 0.05, 0.6, 1.1, 0.2, 0.6, 1.1])
fig, axes = plt.subplots(nrows=8, ncols=2, gridspec_kw=gridspec1, figsize=(7,9))
plt.tight_layout(h_pad = 10, w_pad = 5)

axes[2,0].set_visible(False)
axes[2,1].set_visible(False)
axes[5,0].set_visible(False)
axes[5,1].set_visible(False)



axes[0,0].plot(q_obs_sim_altenbamberg.index, q_obs_sim_altenbamberg.q_obs, linewidth = 0.5, color = 'k')
axes[0,0].set_xlim(q_obs_sim_altenbamberg.index[0], q_obs_sim_altenbamberg.index[-1])
axes[0,0].set_ylim(-2.9, 35)
axes[0,0].tick_params(axis = 'y', labelsize = 6)
axes[0,0].set_ylabel('Discharge [m³s⁻¹]', fontsize = 6, labelpad = 13)
# axes[0,0].text(400, 20, 'a', fontsize = 12)
axes[0,0].text(0.65, 0.8, 'Altenbamberg', fontsize = 8,transform=axes[0,0].transAxes)


axes[1,0].plot(#q_obs_sim_altenbamberg.index, q_obs_sim_altenbamberg.pct_chg_qtrue, 'b',
               q_obs_sim_altenbamberg.index, q_obs_sim_altenbamberg.pct_chg_ss1, 'r',
               q_obs_sim_altenbamberg.index, q_obs_sim_altenbamberg.pct_chg_s2, 'forestgreen', linewidth = 0.3)
axes[1,0].xaxis.set_major_formatter(date_form)
axes[1,0].set_xlim(q_obs_sim_altenbamberg.index[0], q_obs_sim_altenbamberg.index[-1])
axes[1,0].xaxis.set_major_locator(mdates.YearLocator())
axes[1,0].xaxis.set_minor_locator(mdates.MonthLocator())
axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
axes[1,0].xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
for label in axes[1,0].xaxis.get_ticklabels(which = 'minor')[::1]:
    label.set_visible(False)
axes[1,0].tick_params(axis = 'x', direction='in', which = 'minor', labelsize = 6)
axes[1,0].tick_params(axis = 'x', direction='in', which = 'major', labelsize = 8, pad = 1)
axes[1,0].tick_params(axis = 'y', labelsize = 6)
for tick in axes[1,0].xaxis.get_minor_ticks():
    tick.label1.set_horizontalalignment('left')
axes[1,0].set_ylabel('Normalized difference in Q [%]', fontsize = 6, labelpad = 9)


axes[0,1].plot(q_obs_sim_boos.index, q_obs_sim_boos.q_obs, linewidth = 0.5, color = 'k')
axes[0,1].set_xlim(q_obs_sim_boos.index[0], q_obs_sim_boos.index[-1])
#axes[0,1].set_ylim(-80, 320)
axes[0,1].tick_params(axis = 'y', labelsize = 6)
axes[0,1].text(0.85, 0.8, 'Boos', fontsize = 8,transform=axes[0,1].transAxes)


axes[1,1].plot(#q_obs_sim_boos.index, q_obs_sim_boos.pct_chg_qtrue, 'b',
               q_obs_sim_boos.index, q_obs_sim_boos.pct_chg_ss1, 'r',
               q_obs_sim_boos.index, q_obs_sim_boos.pct_chg_s2, 'forestgreen', linewidth = 0.5)
axes[1,1].annotate('=488', xy=(168, 95), xycoords='axes points',
            xytext=(-18,3), textcoords='offset points',
            arrowprops=dict(facecolor='black', arrowstyle = '->'),
            horizontalalignment='right', verticalalignment='top', fontsize = 5.5)
axes[1,1].xaxis.set_major_formatter(date_form)
axes[1,1].set_xlim(q_obs_sim_boos.index[0], q_obs_sim_boos.index[-1])
axes[1,1].set_ylim(-90, 230)
axes[1,1].xaxis.set_major_locator(mdates.YearLocator())
axes[1,1].xaxis.set_minor_locator(mdates.MonthLocator())
axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
axes[1,1].xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
for label in axes[1,1].xaxis.get_ticklabels(which = 'minor')[::1]:
    label.set_visible(False)
axes[1,1].tick_params(axis = 'x', direction='in', which = 'minor', labelsize = 6)
axes[1,1].tick_params(axis = 'x', direction='in', which = 'major', labelsize = 8, pad = 1)
axes[1,1].tick_params(axis = 'y', labelsize = 6)
for tick in axes[1,1].xaxis.get_minor_ticks():
    tick.label1.set_horizontalalignment('left')


axes[3,0].plot(q_obs_sim_eschenau.index, q_obs_sim_eschenau.q_obs, linewidth = 0.5, color = 'k')
axes[3,0].set_xlim(q_obs_sim_eschenau.index[0], q_obs_sim_eschenau.index[-1])
axes[3,0].tick_params(axis = 'y', labelsize = 6)
axes[3,0].set_ylabel('Discharge [m³s⁻¹]', fontsize = 6, labelpad = 9)
axes[3,0].text(0.75, 0.8, 'Eschenau', fontsize = 8,transform=axes[3,0].transAxes)
axes[3,0].set_ylim(-5, 102)


axes[4,0].plot(#q_obs_sim_eschenau.index, q_obs_sim_eschenau.pct_chg_qtrue, 'b',
               q_obs_sim_eschenau.index, q_obs_sim_eschenau.pct_chg_ss1, 'r',
               q_obs_sim_eschenau.index, q_obs_sim_eschenau.pct_chg_s2, 'forestgreen', linewidth = 0.5)
axes[4,0].xaxis.set_major_formatter(date_form)
axes[4,0].set_xlim(q_obs_sim_eschenau.index[0], q_obs_sim_eschenau.index[-1])
axes[4,0].xaxis.set_major_locator(mdates.YearLocator())
axes[4,0].xaxis.set_minor_locator(mdates.MonthLocator())
axes[4,0].xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
axes[4,0].xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
for label in axes[4,0].xaxis.get_ticklabels(which = 'minor')[::1]:
    label.set_visible(False)
axes[4,0].tick_params(axis = 'x', direction='in', which = 'minor', labelsize = 7)
axes[4,0].tick_params(axis = 'x', direction='in', which = 'major', labelsize = 7, pad = -1)
axes[4,0].tick_params(axis = 'y', labelsize = 6)
for tick in axes[4,0].xaxis.get_minor_ticks():
    tick.label1.set_horizontalalignment('left')
axes[4,0].set_ylabel('Normalized difference in Q [%]', fontsize = 6)
axes[4,0].set_xlabel('Date', fontsize = 6)
axes[4,0].set_ylim(-100, 310)



axes[3,1].plot(q_obs_sim_imsweiler.index, q_obs_sim_imsweiler.q_obs, linewidth = 0.5, color = 'k')
axes[3,1].set_xlim(q_obs_sim_imsweiler.index[0], q_obs_sim_imsweiler.index[-1])
axes[3,1].tick_params(axis = 'y', labelsize = 6)
axes[3,1].text(0.75, 0.8, 'Imsweiler', fontsize = 8,transform=axes[3,1].transAxes)


axes[4,1].plot(#q_obs_sim_imsweiler.index, q_obs_sim_imsweiler.pct_chg_qtrue, 'b',
               q_obs_sim_imsweiler.index, q_obs_sim_imsweiler.pct_chg_ss1, 'r',
               q_obs_sim_imsweiler.index, q_obs_sim_imsweiler.pct_chg_s2, 'forestgreen', linewidth = 0.5)
axes[4,1].xaxis.set_major_formatter(date_form)
axes[4,1].set_xlim(q_obs_sim_imsweiler.index[0], q_obs_sim_imsweiler.index[-1])
axes[4,1].xaxis.set_major_locator(mdates.YearLocator())
axes[4,1].xaxis.set_minor_locator(mdates.MonthLocator())
axes[4,1].xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
axes[4,1].xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
for label in axes[4,1].xaxis.get_ticklabels(which = 'minor')[::1]:
    label.set_visible(False)
axes[4,1].tick_params(axis = 'x', direction='in', which = 'minor', labelsize = 7)
axes[4,1].tick_params(axis = 'x', direction='in', which = 'major', labelsize = 7, pad = -1)
axes[4,1].tick_params(axis = 'y', labelsize = 6)
for tick in axes[4,1].xaxis.get_minor_ticks():
    tick.label1.set_horizontalalignment('left')
axes[4,1].set_xlabel('Time', fontsize = 6)


axes[6,0].get_xaxis().set_visible(False)
axes[6,0].get_yaxis().set_visible(False)
gs1 = axes[6, 0].get_gridspec()
for ax in axes[6:7, 1]:
    ax.remove()
axbigtop = fig.add_subplot(gs1[6, 0:])
axbigtop.plot(q_obs_sim_grolsheim.index, q_obs_sim_grolsheim.q_obs, linewidth = 0.5, color = 'k')
axbigtop.set_xlim(q_obs_sim_altenbamberg.index[0], q_obs_sim_altenbamberg.index[-1])
axbigtop.set_ylim(-50, 450)
axbigtop.set_ylabel('Discharge [m³s⁻¹]', fontsize = 6, labelpad = 9)
axbigtop.text(0.45, 0.8, 'Grolsheim', fontsize = 8,transform=axbigtop.transAxes)
axbigtop.tick_params(axis = 'y', labelsize = 6)


axes[7,0].get_xaxis().set_visible(False)
axes[7,0].get_yaxis().set_visible(False)
gs = axes[7, 0].get_gridspec()
for ax in axes[7:, 1]:
    ax.remove()
axbigbot = fig.add_subplot(gs[7, 0:])
axbigbot.plot(#q_obs_sim_grolsheim.index, q_obs_sim_grolsheim.pct_chg_qtrue, 'b',
               q_obs_sim_grolsheim.index, q_obs_sim_grolsheim.pct_chg_ss1, 'r',
               q_obs_sim_grolsheim.index, q_obs_sim_grolsheim.pct_chg_s2, 'forestgreen', linewidth = 0.4)

axbigbot.xaxis.set_major_formatter(date_form)
axbigbot.set_xlim(q_obs_sim_altenbamberg.index[0], q_obs_sim_altenbamberg.index[-1])
axbigbot.xaxis.set_major_locator(mdates.YearLocator())
axbigbot.xaxis.set_minor_locator(mdates.MonthLocator())
axbigbot.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
axbigbot.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
for label in axbigbot.xaxis.get_ticklabels(which = 'minor')[::2]:
    label.set_visible(False)
axbigbot.tick_params(axis = 'x', direction='in', which = 'minor', labelsize = 6)
axbigbot.tick_params(axis = 'x', direction='in', which = 'major', labelsize = 8, pad = 2)
axbigbot.tick_params(axis = 'y', labelsize = 6)
for tick in axbigbot.xaxis.get_minor_ticks():
    tick.label1.set_horizontalalignment('left')
axbigbot.set_ylabel('Normalized difference in Q [%]', fontsize = 6, labelpad = 8)
axbigbot.set_xlabel('Time', fontsize = 9)



# temporal state analysis

fig, axs = plt.subplots(3, figsize = (8,4), gridspec_kw={'height_ratios': [0.5, 1, 1]}, sharex=True)
fig.subplots_adjust(hspace=0)

lim_min = -230
lim_max = 460

axs[0].plot(states.index, q_true.Grolsheim_q_true, linewidth = 0.65, color = 'k')
axs[0].set_xlim(states.index[0], states.index[-1])
axs[0].tick_params(axis = 'x', direction='in', which = 'both')
axs[0].tick_params(axis = 'y', labelsize = 5)
axs[0].set_ylabel('Discharge [m³s$^{-1}$]', fontsize = 4.6, labelpad = 5)
axs[0].set_ylim(-25,350)

axs[1].plot(states.index, states.pct_chg_ss1, 'r', 
            #states.index, states.pct_chg_s2, 'forestgreen',
            linewidth = 0.5)
axs[1].fill_between(states.index, (states.pct_chg_ss1-states.std_err_pct_chg_ss1), (states.pct_chg_ss1+states.std_err_pct_chg_ss1), color='r', alpha=.1)
axs[1].set_xlim(states.index[0], states.index[-1])
axs[1].tick_params(axis = 'x', direction='in', which = 'both')
axs[1].tick_params(axis = 'y', labelsize = 5)
axs[1].set_ylabel('Normalized difference in state [%]', fontsize = 4.6, labelpad = 2)
axs[1].set_ylim(lim_min, lim_max)

axs[2].plot(states.index, states.pct_chg_s2, 'forestgreen', linewidth = 0.5)
axs[2].fill_between(states.index, (states.pct_chg_s2-states.std_err_pct_chg_s2),
                 (states.pct_chg_s2+states.std_err_pct_chg_s2), color='forestgreen', alpha=.1)
axs[2].xaxis.set_major_formatter(date_form)
axs[2].set_xlim(states.index[0], states.index[-1])
axs[2].xaxis.set_major_locator(mdates.YearLocator())
axs[2].xaxis.set_minor_locator(mdates.MonthLocator())
axs[2].xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
axs[2].xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
for label in axs[2].xaxis.get_ticklabels(which = 'minor')[::2]:
    label.set_visible(False)
axs[2].tick_params(axis = 'x', direction='in', which = 'minor', labelsize = 6)
axs[2].tick_params(axis = 'x', direction='in', which = 'major', labelsize = 7)
axs[2].tick_params(axis = 'y', labelsize = 5)
for tick in axs[2].xaxis.get_minor_ticks():
    tick.label1.set_horizontalalignment('left')
axs[2].set_xlabel('Time', fontsize = 7, labelpad = 2)
axs[2].set_ylabel('Normalized difference in state [%]', fontsize = 4.6, labelpad = 2)
axs[2].set_ylim(lim_min, lim_max)

plt.text(0.55, 0.8, '$Q_{true}$', fontsize = 8,transform=axs[0].transAxes)
plt.text(0.55, 0.85, 'SS1', fontsize = 8,transform=axs[1].transAxes)
plt.text(0.55, 0.85, 'S2', fontsize = 8,transform=axs[2].transAxes)

#plt.text(0.9, 0.8, 'a', fontsize = 12,transform=axs[0].transAxes)
#plt.text(0.9, 0.8, 'b', fontsize = 12,transform=axs[1].transAxes)
#plt.text(0.9, 0.8, 'c', fontsize = 12,transform=axs[2].transAxes)

plt.show()

# calculate high flow percentage bias (Yilmaz, 2008)
def pct_bias_fhv(q_sim, q_obs):
    return np.sum((q_sim - q_obs))/ np.sum(q_obs) *100


data_sorted = q_obs_sim_test.sort_values(by='q_obs')
data_sorted_80 = data_sorted[1168:]
fhv_ol = pct_bias_fhv(data_sorted_80.q_sim_ol, data_sorted_80.q_obs)
fhv_s2 = pct_bias_fhv(data_sorted_80.q_ass_s2, data_sorted_80.q_obs)
fhv_ss1 = pct_bias_fhv(data_sorted_80.q_ass_ss1, data_sorted_80.q_obs)


# evaluate synthetic experiment
kge_q_true, r_q_true, alpha_q_true, beta_q_true = spotpy.objectivefunctions.kge(q_obs_sim_grolsheim.Grolsheim_q_true, q_obs_sim_grolsheim.Grolsheim_s2, return_all=True)
kge_q_true_s2, r_q_true_s2, alpha_q_true_s2, beta_q_true_s2 = spotpy.objectivefunctions.kge(q_obs_sim_grolsheim.Grolsheim_q_true, q_obs_sim_grolsheim.Grolsheim_s2, return_all=True)
kge_q_true_ss1, r_kge_q_ture_ss1, alpha_q_ture_ss1_ln, beta_q_ture_ss1 = spotpy.objectivefunctions.kge(q_obs_sim_grolsheim.Grolsheim_q_true, q_obs_sim_grolsheim.Grolsheim_ss1, return_all=True)

nse_q_true_s2 = spotpy.objectivefunctions.nashsutcliffe(q_obs_sim_grolsheim.Grolsheim_q_true, q_obs_sim_grolsheim.Grolsheim_s2)
nse_q_true_ss1 = spotpy.objectivefunctions.nashsutcliffe(q_obs_sim_grolsheim.Grolsheim_q_true, q_obs_sim_grolsheim.Grolsheim_ss1)

lognse_q_true_s2 = spotpy.objectivefunctions.lognashsutcliffe(q_obs_sim_grolsheim.Grolsheim_q_true, q_obs_sim_grolsheim.Grolsheim_s2)
lognse_q_true_ss1 = spotpy.objectivefunctions.lognashsutcliffe(q_obs_sim_grolsheim.Grolsheim_q_true, q_obs_sim_grolsheim.Grolsheim_ss1)


# evaluate real world experiment 2

