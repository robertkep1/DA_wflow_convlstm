#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:29:49 2021

@author: robert
"""

import spotpy
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
import matplotlib.ticker as ticker
import matplotlib.dates as datesplot
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


os.chdir('/home/robert/pCloudLocal/UNI/MEE/MScThesis/data/ConvLSTM')

# load observed q
q_obs = pd.read_csv('q_obs/grolsheim_dietersheim_1.csv', skiprows=1,sep=',', parse_dates=[0], header=None)
q_obs = q_obs.drop([0], axis=1)
q_obs = q_obs.drop([0], axis=0)
q_obs = q_obs.drop(columns=[1])
DTI1 = pd.to_datetime(q_obs.index.values, unit='D', origin=pd.Timestamp('1953-5-23'))
q_obs.index = DTI1

# load open loop q sim
open_loop_q_sim = pd.read_csv('q_sim/q_sim_int_run.csv', header = None)
begin_date = '2016-01-01'
dates = []

#load assim q for stacked_sep_1 model
q_assim_ss1 = pd.read_csv('q_assim/run_assim_stacked_sep_1.csv', header = None)

# load assim q for stacked 2 model
q_assim_s2 = pd.read_csv('q_assim/run_assim_stacked_2.csv', header = None)

for i in np.arange(0, 365*4):
    if i == 0:
        start_time = begin_date
    else:
        start_time = end_time
    
    end_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = end_time + timedelta(days = 1)
    dates.append(end_time)
    end_time = str(end_time.date())

open_loop_q_sim['dates'] = dates
open_loop_q_sim = open_loop_q_sim.set_index('dates')
open_loop_q_sim = open_loop_q_sim.rename(columns = {0:'q_sim_ol'})

q_obs_testp = q_obs.loc['2016-01-02':'2019-12-31']
q_obs_testp = q_obs_testp.rename(columns = {2:'q_obs'})

q_assim_ss1['dates'] = dates
q_assim_ss1 = q_assim_ss1.set_index('dates')
q_assim_ss1 = q_assim_ss1.rename(columns = {0:'q_ass_ss1'})

q_assim_s2['dates'] = dates
q_assim_s2 = q_assim_s2.set_index('dates')
q_assim_s2 = q_assim_s2.rename(columns = {0:'q_ass_s2'})

q_obs_sim_test = pd.concat([q_obs_testp, open_loop_q_sim, q_assim_ss1, q_assim_s2], axis = 1)


eval_df = pd.DataFrame(columns = ['kge_ol', 'nse_ol', 'lognse_ol', 'kge_ass_ss1', 'nse_ass_ss1', 'lognse_ass_ss1', 'kge_ass_s2', 'nse_ass_s2', 'lognse_ass_s2'],
                       index=('2016', '2017', '2018', '2019'))
for i in np.arange(4):
    start_date = '201' +str(6+i) + '-01-01'
    end_date = '201' + str(6+i) + '-12-31'
    year = datetime.strptime(start_date, '%Y-%m-%d')
    q_obs_sim_yearly = q_obs_sim_test.loc[start_date : end_date]
    
    # assess performance open loop simulation
    kge_wflow_ol,cc,alpha,beta=spotpy.objectivefunctions.kge(q_obs_sim_yearly['q_obs'], q_obs_sim_yearly['q_sim_ol'],return_all=True)
    
    nse_wflow_ol=spotpy.objectivefunctions.nashsutcliffe(q_obs_sim_yearly['q_obs'], q_obs_sim_yearly['q_sim_ol'])
    
    lognse_wflow_ol=spotpy.objectivefunctions.lognashsutcliffe(q_obs_sim_yearly['q_obs'], q_obs_sim_yearly['q_sim_ol'],)
    
    eval_df['kge_ol'][str(year.year)] = round(kge_wflow_ol, 2)
    eval_df['nse_ol'][str(year.year)] = round(nse_wflow_ol, 2)
    eval_df['lognse_ol'][str(year.year)] = round(lognse_wflow_ol, 2)
    
    # assess performance of assimilated simulation model 'stacked sep 1'
    kge_ass_ss1,cc,alpha,beta = spotpy.objectivefunctions.kge(q_obs_sim_yearly['q_obs'],q_obs_sim_yearly['q_ass_ss1'], return_all=True)
    
    nse_ass_ss1 = spotpy.objectivefunctions.nashsutcliffe(q_obs_sim_yearly['q_obs'], q_obs_sim_yearly['q_ass_ss1'])
    
    lognse_ass_ss1 = spotpy.objectivefunctions.lognashsutcliffe(q_obs_sim_yearly['q_obs'], q_obs_sim_yearly['q_ass_ss1'])
    
    eval_df['kge_ass_ss1'][str(year.year)] = round(kge_ass_ss1, 2)
    eval_df['nse_ass_ss1'][str(year.year)] = round(nse_ass_ss1, 2)
    eval_df['lognse_ass_ss1'][str(year.year)] = round(lognse_ass_ss1, 2)
    
    # assess performance of ass simulation model 'stacked 2'
    kge_ass_s2,cc,alpha,beta = spotpy.objectivefunctions.kge(q_obs_sim_yearly['q_obs'],q_obs_sim_yearly['q_ass_s2'], return_all=True)
    
    nse_ass_s2 = spotpy.objectivefunctions.nashsutcliffe(q_obs_sim_yearly['q_obs'], q_obs_sim_yearly['q_ass_s2'])
    
    lognse_ass_s2 = spotpy.objectivefunctions.lognashsutcliffe(q_obs_sim_yearly['q_obs'], q_obs_sim_yearly['q_ass_s2'])
    
    eval_df['kge_ass_s2'][str(year.year)] = round(kge_ass_s2, 2)
    eval_df['nse_ass_s2'][str(year.year)] = round(nse_ass_s2, 2)
    eval_df['lognse_ass_s2'][str(year.year)] = round(lognse_ass_s2, 2)

kge_testp_ol, r_kge_ol, alpha_kge_ol, beta_kge_ol = spotpy.objectivefunctions.kge(q_obs_sim_test.q_obs, q_obs_sim_test.q_sim_ol, return_all = True)
nse_testp_ol = spotpy.objectivefunctions.nashsutcliffe(q_obs_sim_test.q_obs, q_obs_sim_test.q_sim_ol)
lognse_testp_ol = spotpy.objectivefunctions.lognashsutcliffe(q_obs_sim_test.q_obs,q_obs_sim_test.q_sim_ol)

kge_testp_ass_ss1, r_kge_ss1, alpha_kge_ss1, beta_kge_ss1 = spotpy.objectivefunctions.kge(q_obs_sim_test.q_ass_ss1, q_obs_sim_test.q_obs, return_all = True)
nse_testp_ass_ss1 = spotpy.objectivefunctions.nashsutcliffe(q_obs_sim_test.q_ass_ss1, q_obs_sim_test.q_obs)
lognse_testp_ass_ss1 = spotpy.objectivefunctions.lognashsutcliffe(q_obs_sim_test.q_ass_ss1, q_obs_sim_test.q_obs)

kge_testp_ass_s2, r_kge_s2, alpha_kge_s2, beta_kge_s2 = spotpy.objectivefunctions.kge(q_obs_sim_test.q_ass_s2, q_obs_sim_test.q_obs, return_all = True)
nse_testp_ass_s2 = spotpy.objectivefunctions.nashsutcliffe(q_obs_sim_test.q_ass_s2, q_obs_sim_test.q_obs)
lognse_testp_ass_s2 = spotpy.objectivefunctions.lognashsutcliffe(q_obs_sim_test.q_ass_s2, q_obs_sim_test.q_obs)

eval_df.loc['overall'] = [round(kge_testp_ol, 2), round(nse_testp_ol, 2), round(lognse_testp_ol, 2),
                          round(kge_testp_ass_ss1, 2), round(nse_testp_ass_ss1, 2), round(lognse_testp_ass_ss1, 2),
                          round(kge_testp_ass_s2, 2), round(nse_testp_ass_s2, 2), round(lognse_testp_ass_s2, 2)]




y2016 = q_obs_sim_test['2016-01-01 00:00:00':'2016-12-31 00:00:00']#.plot(linewidth=0.8)
y2017 = q_obs_sim_test['2017-01-01 00:00:00':'2017-12-31 00:00:00']#.plot(linewidth=0.8)
y2018 = q_obs_sim_test['2018-01-01 00:00:00':'2018-12-31 00:00:00']#.plot(linewidth=0.8)
y2019 = q_obs_sim_test['2019-01-01 00:00:00':'2019-12-31 00:00:00']#.plot(linewidth=0.8)


fig, axes = plt.subplots(4, figsize = (16,12), sharex=False)
fig.subplots_adjust(hspace=0)
axes[0].plot(y2016.index, y2016['q_sim_ol'].values, 'b', y2016.index,
             y2016['q_ass_ss1'].values, 'r', y2016.index, y2016['q_ass_s2'].values,'forestgreen', y2016.index, y2016['q_obs'].values, 'k--', linewidth = 1.8)
axes[0].set_xlim(y2016.index[0], y2016.index[364])
axes[0].set_ylim(-10, np.max(y2016.max())+10)
axes[0].tick_params(axis='x', direction='in', labelsize=0)

axes[1].plot( y2017.index, y2017['q_sim_ol'].values, 'b', y2017.index,
             y2017['q_ass_ss1'].values, 'r', y2017.index, y2017['q_ass_s2'].values,'forestgreen', y2017.index, y2017['q_obs'].values, 'k--', linewidth = 1.8)
axes[1].set_xlim(y2017.index[0], y2017.index[364])
axes[1].set_ylim(-10, np.max(y2017.max())+10)
axes[1].tick_params(axis = 'x', direction='in', labelsize=0)

axes[2].plot( y2018.index, y2018['q_sim_ol'].values, 'b', y2018.index,
             y2018['q_ass_ss1'].values, 'r', y2018.index, y2018['q_ass_s2'].values,'forestgreen', y2018.index, y2018['q_obs'].values, 'k--', linewidth = 1.8)
axes[2].set_xlim(y2018.index[0], y2018.index[364])
axes[2].set_ylim(-10, np.max(y2018.max())+10)
axes[2].tick_params(axis = 'x', direction='in', labelsize = 0)

axes[3].plot( y2019.index, y2019['q_sim_ol'].values, 'b', y2019.index,
             y2019['q_ass_ss1'].values, 'r', y2019.index, y2019['q_ass_s2'].values,'forestgreen', y2019.index, y2019['q_obs'].values, 'k--', linewidth = 1.8)
axes[3].set_xlim(y2019.index[0], y2019.index[364])
axes[3].set_ylim(-10, np.max(y2019.max())+10)
axes[3].tick_params(axis = 'x', direction='in')
fig.text(0.09, 0.5, 'Discharge [m³s⁻¹]', va='center', rotation='vertical')

date = y2019.index.astype('O')
axes[3].xaxis.set_major_locator(datesplot.MonthLocator())
axes[3].xaxis.set_minor_locator(datesplot.MonthLocator(bymonthday=16))
axes[3].xaxis.set_major_formatter(ticker.NullFormatter())
axes[3].xaxis.set_minor_formatter(datesplot.DateFormatter('%b'))

for tick in axes[3].xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')
    
plt.text(0.5, 0.85, '2016', fontsize = 18,transform=axes[0].transAxes)
plt.text(0.5, 0.85, '2017', fontsize = 18,transform=axes[1].transAxes)
plt.text(0.5, 0.85, '2018', fontsize = 18,transform=axes[2].transAxes)
plt.text(0.5, 0.85, '2019', fontsize = 18,transform=axes[3].transAxes)

axes[0].legend(['Open Loop', 'Assim SS1','Assim S2', 'Observed'])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='right', fontsize = 13)
axes[3].set_xlabel('Time')


delta_err_fcts = pd.DataFrame(columns = ['kge_ass_ss1', 'nse_ass_ss1', 'lognse_ass_ss1', 'kge_ass_s2', 'nse_ass_s2', 'lognse_ass_s2'],
                       index=('2016', '2017', '2018', '2019', 'overall'))
delta_err_fcts['kge_ass_ss1'] = eval_df['kge_ass_ss1'] - eval_df['kge_ol']
delta_err_fcts['nse_ass_ss1'] = eval_df['nse_ass_ss1'] - eval_df['nse_ol']
delta_err_fcts['lognse_ass_ss1'] = eval_df['nse_ass_ss1'] - eval_df['nse_ol']

delta_err_fcts['kge_ass_s2'] = eval_df['kge_ass_s2'] - eval_df['kge_ol']
delta_err_fcts['nse_ass_s2'] = eval_df['nse_ass_s2'] - eval_df['nse_ol']
delta_err_fcts['lognse_ass_s2'] = eval_df['nse_ass_s2'] - eval_df['nse_ol']


fig, axes = plt.subplots(1, 3, sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
# (ax1, ax2, ax3) = axs
# fig.suptitle('Sharing x per column, y per row')
# ax1.bar(eval_df[['kge_ol', 'kge_ass_ss1', 'kge_ass_s2']].index.values.astype(str), eval_df[['kge_ol', 'kge_ass_ss1', 'kge_ass_s2']].values)
# ax2.plot(x, y**2, 'tab:orange')
# ax3.plot(x + 1, -y, 'tab:green')

# ax1.rc('axes', axisbelow = True)
# fig.grid(True, color='black', axis='y')

eval_df[['kge_ol', 'kge_ass_ss1', 'kge_ass_s2']].plot(kind='bar', ylim=[0,1.3], ax=axes[0])
eval_df[['nse_ol', 'nse_ass_ss1', 'nse_ass_s2']].plot(kind='bar', ylim=[0,1.3], ax=axes[1])
eval_df[['lognse_ol', 'lognse_ass_ss1', 'lognse_ass_s2']].plot(kind='bar', ylim=[0,1.3], ax=axes[2])

# .rc('axes', axisbelow = True)


fig, axes = plt.subplots(1, 3, sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
# (ax1, ax2, ax3) = axs
# fig.suptitle('Sharing x per column, y per row')
# ax1.bar(eval_df[['kge_ol', 'kge_ass_ss1', 'kge_ass_s2']].index.values.astype(str), eval_df[['kge_ol', 'kge_ass_ss1', 'kge_ass_s2']].values)
# ax2.plot(x, y**2, 'tab:orange')
# ax3.plot(x + 1, -y, 'tab:green')

# ax1.rc('axes', axisbelow = True)
# fig.grid(True, color='black', axis='y')

delta_err_fcts[['kge_ass_ss1', 'kge_ass_s2']].plot(kind='bar', ylim=[-0.3, 0.3], ax=axes[0])
delta_err_fcts[['nse_ass_ss1', 'nse_ass_s2']].plot(kind='bar', ax=axes[1])
delta_err_fcts[['lognse_ass_ss1', 'lognse_ass_s2']].plot(kind='bar', ax=axes[2])


# calculate NSE components

def nse_man(q_sim, q_obs):
    A = (np.corrcoef(q_sim, q_obs)[0, 1])**2
    B = (np.sqrt(A) - (np.std(q_sim)/np.std(q_obs)))**2
    C = ((np.mean(q_sim) - np.mean(q_obs))/ np.std(q_obs))**2
    print(np.sqrt(C))
    return A - B - C

def nse_man_2(q_sim, q_obs):
    r = np.corrcoef(q_sim, q_obs)[0, 1]
    print('r = ', r)
    alpha = np.std(q_sim)/ np.std(q_obs)
    print('alpha = ', alpha)
    beta = (np.mean(q_sim) - np.mean(q_obs) )/ np.std(q_obs) 
    print('beta = ', beta)
    return 2 * alpha * r - alpha**2 - beta**2

def APFB(df):
    q_sim = df['q_ass_s2']
    q_obs = df['q_obs']
    return np.sqrt(((np.mean(q_sim)/ np.mean(q_obs) - 1))**2)


# calculate high flow percentage bias (Yilmaz, 2008)
def pct_bias_fhv(q_sim, q_obs):
    return np.sum((q_sim - q_obs))/ np.sum(q_obs) *100


data_sorted = q_obs_sim_test.sort_values(by='q_obs')
data_sorted_80 = data_sorted[1168:]
fhv_ol = pct_bias_fhv(data_sorted_80.q_sim_ol, data_sorted_80.q_obs)
fhv_s2 = pct_bias_fhv(data_sorted_80.q_ass_s2, data_sorted_80.q_obs)
fhv_ss1 = pct_bias_fhv(data_sorted_80.q_ass_ss1, data_sorted_80.q_obs)


# calculate slope of regression line of sim against observed runoff
lr_ol = linear_model.LinearRegression().fit(q_obs_sim_test.q_obs.values.reshape(1, -1), q_obs_sim_test.q_sim_ol.values.reshape(1, -1))
r2_ol = r2_score(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_sim_ol.values)
r2_s2 = r2_score(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_ass_s2.values)
r2_ss1 = r2_score(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_ass_ss1.values)


linregress_ol= stats.linregress(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_sim_ol.values)
linregress_s2 = stats.linregress(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_ass_s2.values)
linregress_ss1 = stats.linregress(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_ass_ss1.values)


fig, axs = plt.subplots(1, 3, figsize = (10, 3))

axs[0].plot(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_sim_ol.values, '.')
axs[0].plot(q_obs_sim_test.q_obs.values, linregress_ol.intercept + linregress_ol.slope*q_obs_sim_test.q_obs.values, 'r')
axs[0].plot([0,450], [0,450], linewidth=0.8, color = 'k', linestyle = '--')
axs[0].set_ylim((0, 450))
axs[0].set_xlim((0, 450))
axs[0].text(360, 340, '1:1 line', rotation=45)
axs[0].text(20, 390, 'Open Loop', weight = 'bold', fontsize = 12)
axs[0].annotate('$k_0 = 0.73$', xy=(350, 250), xytext=(320, 160), fontsize = 9,
            arrowprops=dict(facecolor='black', arrowstyle = '->'))


axs[1].plot(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_ass_s2.values, '.')
axs[1].plot(q_obs_sim_test.q_obs.values, linregress_s2.intercept + linregress_s2.slope * q_obs_sim_test.q_obs.values, 'r', linestyle ='solid')
axs[1].plot([0,450], [0,450], linewidth=0.8, color = 'k', linestyle = '--')
axs[1].set_ylim((0, 450))
axs[1].set_xlim((0, 450))
axs[1].text(360, 340, '1:1 line', rotation=45)
axs[1].text(20, 390, 'S2', weight = 'bold', fontsize = 12)
axs[1].annotate('$k_0 = 1.09$', xy=(340, 380), xytext=(150, 410), fontsize = 9,
            arrowprops=dict(facecolor='black', arrowstyle = '->'))

axs[2].plot(q_obs_sim_test.q_obs.values, q_obs_sim_test.q_ass_ss1.values, '.')
axs[2].plot(q_obs_sim_test.q_obs.values, linregress_ss1.intercept + linregress_ss1.slope * q_obs_sim_test.q_obs.values, 'r')
axs[2].plot([0,450], [0,450], linewidth=0.8, color = 'k', linestyle = '--')
axs[2].set_ylim((0, 450))
axs[2].set_xlim((0, 450))
axs[2].text(360, 340, '1:1 line', rotation=45)
axs[2].text(20, 390, 'SS1', weight = 'bold', fontsize = 12)
axs[2].annotate('$k_0 = 0.8$', xy=(360, 283), xytext=(340, 180), fontsize = 9,
            arrowprops=dict(facecolor='black', arrowstyle = '->'))


plt.setp(axs[-3], ylabel = 'simulated runoff [m³ s⁻¹]')
for ax in axs.flat:
    ax.set(xlabel = 'observed runoff [m³ s⁻¹]')


    
# plt.hist(q_obs_sim_test.q_ass_ss1, bins = 50, alpha=0.5, density=True, range=[0,150])
# plt.hist(q_obs_sim_test.q_obs, bins = 50, alpha = 0.5, density=True, range=[0,150])
# plt.show()