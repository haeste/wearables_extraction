# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:07:02 2022

@author: nct76
"""


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import more_itertools as mit
import numpy as np
import scipy as scp
import os
import proj_funcs.FilterEEG as filter_funcs
import utilities.file_reading as file_reading 
import utilities.wearables_hsmm as wbhsmm
import biorhythms


def calc_HR_HRV(hrv, hrv_times):
    
    hrv_calc_len = 5 #minutes
    hr_calc_len = 1 #minutes
    
    hrv_sf= 1/(0.001*(hrv_times[1] - hrv_times[0]).astype('timedelta64[ms]').astype(np.int64))
    
    hrv_chunkedlen = int( hrv_calc_len * (hrv_sf*60))
    hr_chunkedlen = int(hr_calc_len*(60 * hrv_sf))

    rrs_err = [np.array(x)[(np.array(x)<300) | (np.array(x)>1300)] for x in mit.chunked(hrv, hrv_chunkedlen)]

    rrs = [np.array(x)[(np.array(x)>300) & (np.array(x)<1300)] for x in mit.chunked(hrv, hrv_chunkedlen)]
    rrs_hr = [np.array(x)[(np.array(x)>300) & (np.array(x)<1300)] for x in mit.chunked(hrv, hr_chunkedlen)]
    rrs_hr_err = [np.array(x)[(np.array(x)<300) & (np.array(x)>1300)] for x in mit.chunked(hrv, hr_chunkedlen)]

    noise = np.array([sum(r>0) for r in rrs_err])
    noisehr = np.array([sum(r>0) for r in rrs_hr_err])

    RMSSD = np.array([np.sqrt(sum((r[:-1] - r[1:])**2)/(len(r)-1)) if len(r)>100 else np.nan for r in rrs])
    SDNNI = np.array([np.std(r) if len(r)>100 else np.nan for r in rrs])
    SDNNI[noise>20] = np.nan
    hr = np.array([60*1000*(1/np.mean(r)) if len(r)>5 else np.nan for r in rrs_hr])
    hr[noisehr>20] = np.nan
    rdff_t = np.arange(0,len(RMSSD))*(60 *hrv_calc_len)
    rdff_hr_t = np.arange(0,len(hr))*(hr_calc_len*60)

    rdff_t_ms = rdff_t*1000
    rdff_hr_t_ms = rdff_hr_t*1000
    rdff_t_ms =np.datetime64(hrv_times[0]).astype('datetime64[ms]') + rdff_t_ms.astype(int).astype('timedelta64[ms]')
    rdff_hr_t_ms =np.datetime64(hrv_times[0]).astype('datetime64[ms]') + rdff_hr_t_ms.astype(int).astype('timedelta64[ms]')
    
    RMSSD = pd.Series(data=RMSSD, index=rdff_t_ms, name='RMSSD')
    RMSSD = RMSSD.resample('1T').mean()
    RMSSD = RMSSD.interpolate(limit=5)
    
    SDNNI = pd.Series(data=SDNNI, index=rdff_t_ms, name='SDNNI')
    SDNNI = SDNNI.resample('1T').mean()
    SDNNI = SDNNI.interpolate(limit=5)
    
    hr = pd.Series(data=hr, index=rdff_hr_t_ms, name='HR')
    hr = hr.resample('1T').mean()
    hr = hr.interpolate(limit=5)
    
    hr = hr.to_frame()
    hr['HR'] = hr
    hr['SDNNI'] = SDNNI
    hr['t'] = hr.index
    hr = hr.reset_index()
    
    BITData = hr[['t', 'HR', 'SDNNI']]
    
    return BITData

def getCWTParameters(df_meas, meas, show_fig = False):
    df_copy = df_meas.copy()
    RNG = np.random.default_rng(1905)
    fs = 1/(df_meas.t[1] - df_meas.t[0]).seconds
    fig, axs, rhythm, gwps_period, phase  = biorhythms.toolbox.get_rhythm_via_cwt(df_copy[meas], df_copy['t'], 60,60*48, RNG, meas, fs=fs)
    if show_fig: plt.show()
    peaks = scp.signal.argrelextrema(abs(phase-np.pi/2), np.less)[0]
    peaktimes = df_meas.t[peaks]
    peakamplitudes = rhythm[peaks]
    peak_period_var = np.std(np.diff(peaktimes).astype('timedelta64[m]').astype(np.int64))
    peak_amp_var = np.std(np.diff(peakamplitudes))
    return peaktimes, rhythm, phase, peakamplitudes, peak_period_var, peak_amp_var


def addCircadianComponent(df_meas, meas, sf):
    df_temp = df_meas.copy()
    ## cycles periods info
    fluct_name = ["19h-1.3d"]
    #plt.plot(HR.t, HR.pulse_rate_bpm, 'r')
    df_temp['interp'] = df_temp[meas].interpolate(method='linear')
    df_temp.dropna(inplace=True, subset='interp')

    # frequency ranges
    fluct_narrowband = {"19h-1.3d": [0.8*24*60, 1.3*24*60]}

    x = filter_funcs.FilterEEG_Channel(np.array(df_temp.interp), [1/(31*60*60),1/(19*60*60)], sf, 'bandpass', order=2)
    df_temp['circadian'] = x
    analytic_signal = scp.signal.hilbert(x)

    df_temp['amplitude_envelope'] = np.abs(analytic_signal)

    df_temp['instantaneous_phase'] = np.angle(analytic_signal)
    
    return df_temp



#%%

#%%
from importlib import reload
reload(file_reading)
report_folder = '/home/campus.ncl.ac.uk/nct76/Documents/Data/CNNP_Biorhtyhms/S019/report/'
data_dir = '/home/campus.ncl.ac.uk/nct76/Documents/Data/CNNP_Biorhtyhms/'
#data_dir = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23'
subj = 'S019'
#subjects = ['S002']
EMPATICA = True
CGM = False
BITTIUM = False
CORE = False
LYS = False

df_minute_all = pd.DataFrame()
df_agg_measure = pd.DataFrame()

subj_dir = os.path.join(data_dir, subj)
empatica_meta = os.path.join(subj_dir,'Empatica/1-1-' + subj[1:] + '_metadata.csv')
subj_pr = file_reading.read_Empatica_DBM(os.path.join(subj_dir,'Empatica/'), 'pulse-rate')
df_per_minute = subj_pr[['t', 'pulse_rate_bpm']]
del subj_pr
subj_sleep = file_reading.read_Empatica_DBM(os.path.join(subj_dir,'Empatica/'), 'sleep-detection')
df_per_minute = pd.merge(df_per_minute, subj_sleep[['t', 'sleep_detection_stage']], on='t', how='outer')
del subj_sleep
subj_act = file_reading.read_Empatica_DBM(os.path.join(subj_dir,'Empatica/'), 'activity-counts')
df_per_minute = pd.merge(df_per_minute, subj_act[['t', 'activity_counts']], on='t', how='outer')
del subj_act
subj_temp = file_reading.read_Empatica_DBM(os.path.join(subj_dir,'Empatica/'), 'temperature')
df_per_minute = pd.merge(df_per_minute, subj_temp[['t', 'temperature_celsius']], on='t', how='outer')
del subj_temp
subj_eda = file_reading.read_Empatica_DBM(os.path.join(subj_dir,'Empatica/'), 'eda')
df_per_minute = pd.merge(df_per_minute, subj_eda[['t', 'eda_scl_usiemens']], on='t', how='outer')
del subj_eda
if os.path.isdir(os.path.join(subj_dir,'CGM/')):
    subj_cgm = file_reading.read_cgm(os.path.join(subj_dir,'CGM/'))
    subj_cgm = subj_cgm.reset_index()
    subj_cgm = subj_cgm.set_index('t')
    subj_cgm = subj_cgm.resample('1T').mean()
    subj_cgm = subj_cgm.interpolate()
    subj_cgm = subj_cgm.reset_index()
    df_per_minute = pd.merge(df_per_minute, subj_cgm[['t', 'glucose']], on='t', how='outer')
    del subj_cgm
    CGM = True
if os.path.isdir(os.path.join(subj_dir,'Bittium/')):
    ecgs, ecg_times, hrvs, hrv_times = file_reading.read_Bittium_Data(os.path.join(subj_dir,'Bittium/'))
    BITData = calc_HR_HRV(hrvs, hrv_times)
    #GMT = (BITData.t > np.datetime64('2023-10-29T02:00:00')) & (BITData.t < np.datetime64('2024-03-30T02:00:00')) 
    #BITData.loc[GMT,'t'] = BITData.loc[GMT,'t'] - np.timedelta64(60, 'm')
    df_per_minute = pd.merge(df_per_minute, BITData[['t', 'HR', 'SDNNI']], on='t', how='outer')
    del ecgs, ecg_times, hrvs, hrv_times, BITData
    BITTIUM = True
core_path = os.path.join(subj_dir,'CORE/')
if os.path.isdir(core_path):
    c_f = os.path.join(core_path,subj + '_CBT_cloud.csv')
    cbt_df = file_reading.load_CBT_cloud(c_f)
    cbt_df = cbt_df.reset_index()
    df_per_minute = pd.merge(df_per_minute, cbt_df[['t', 'CBT', 'ST']], on='t', how='outer')
    CORE = True
    
LYS_path = os.path.join(subj_dir,'Lys/')
if os.path.isdir(LYS_path):
    l_f = os.path.join(LYS_path,subj + '_LYS.csv')
    lys_data = pd.read_csv(l_f)
    lys_data['t'] = pd.to_datetime(lys_data.Timestamp, yearfirst=True)
    lys_data = lys_data[['t','Kelvin', 'R', 'G', 'B', 'IR', 'Movement', 'Lux',
           'mEDI', 'Covered', 'NotWorn', 'Covered_NotWorn_Nighttime']]
    lys_data['t'] = lys_data.t+np.timedelta64(1, 'h')
    lys_data = lys_data.set_index('t')
    lys_data = lys_data.resample('1T').median()
    df_per_minute = pd.merge(df_per_minute, lys_data, on='t', how='outer')
    LYS = True

df_per_minute = wbhsmm.remove_big_gaps(df_per_minute)
df_per_minute = df_per_minute[df_per_minute.big_gap==0].reset_index()

beginning = df_per_minute.first_valid_index()
ending = df_per_minute.last_valid_index()
df_per_minute = df_per_minute[beginning:ending]

df_per_minute['subj'] = subj

df_minute_all = pd.concat([df_minute_all, df_per_minute])
    

#%%
obs = []
if EMPATICA:
    obs.append('activity_counts')
    obs.append('temperature_celsius')
    obs.append('eda_scl_usiemens')

    if not BITTIUM:
        obs.append('pulse_rate_bpm')
        
if CGM:
    obs.append('glucose')
if BITTIUM:
    obs.append('HR')
    obs.append('SDNNI')
if CORE:
    obs.append('CBT')

to_plot = obs.copy()
if EMPATICA:
    to_plot.append('sleep_detection_stage')

import matplotlib as mpl

fig_labels = {'activity_counts':'Activity', 'temperature_celsius':'Skin Temperature (°C)',
               'eda_scl_usiemens': 'Skin Conductance (nS)', 'glucose':'Blood Glucose (mmol/L)',
'HR':'Heart Rate (bpm)', 'pulse_rate_bpm': 'Pulse Rate (bpm)',
'SDNNI':'Heart Rate Variability (bpm)', 'CBT': 'Core Temperature (°C)',
'sleep_detection_stage': 'Sleep (0=Yes, 100=No)'}

cmap = mpl.colormaps['tab20']
colors = cmap(np.linspace(0, 1, len(to_plot)))
ax1 = plt.subplot(len(to_plot), 1,1)
plt.plot(df_per_minute.t, df_per_minute[to_plot[0]], label=to_plot[0], color=colors[0])
plt.tick_params( labelbottom = False, bottom = False) 
xmin, xmax = ax1.get_xlim()
days = np.arange(0, np.ceil(xmax)+2)
night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_per_minute.t]
ax1.fill_between(df_per_minute.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.5)
plt.ylabel(fig_labels[to_plot[0]], rotation=45, loc='top')

for i, c in enumerate(to_plot[1:]):
    ax = plt.subplot(len(to_plot), 1,i+2, sharex=ax1)
    plt.plot(df_per_minute.t, df_per_minute[c], label=c, color=colors[0])
    xmin, xmax = ax.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_per_minute.t]
    ax.fill_between(df_per_minute.t, *ax.get_ylim(), where=night, facecolor='k', alpha=.5)
    if i<4:
        plt.tick_params( labelbottom = False, bottom = False) 
    plt.ylabel(fig_labels[c], rotation=45, loc='top')


fig = plt.gcf()
fig.set_size_inches(22, 9)
fig.savefig(report_folder+'all_data.pdf',bbox_inches='tight')
plt.close()
#%%

sleeping = {}
waking = {}
sleeping_max = {}
waking_max = {}
sleeping_min = {}
waking_min = {}
for o in obs:
    sleeping[o] = round(df_minute_all.loc[df_minute_all.sleep_detection_stage>1,o].median(),2)
    waking[o] = round(df_minute_all.loc[df_minute_all.sleep_detection_stage==0,o].median(),2)
    sleeping_max[o] = round(df_minute_all.loc[df_minute_all.sleep_detection_stage>1,o].max(),2)
    waking_max[o] = round(df_minute_all.loc[df_minute_all.sleep_detection_stage==0,o].max(),2)
    sleeping_min[o] = round(df_minute_all.loc[df_minute_all.sleep_detection_stage>1,o].min(),2)
    waking_min[o] = round(df_minute_all.loc[df_minute_all.sleep_detection_stage==0,o].min(),2)

index_col = [fig_labels[c] for c in sleeping.keys()]
col_1 = [sleeping[c] for c in sleeping.keys()]
col_2 = [waking[c] for c in sleeping.keys()]
col_3 = [sleeping_max[c] for c in sleeping.keys()]
col_4 = [waking_max[c] for c in sleeping.keys()]
col_5 = [sleeping_min[c] for c in sleeping.keys()]
col_6 = [waking_min[c] for c in sleeping.keys()]


summary_metrics = pd.DataFrame({'Measure':index_col, 'Sleeping Mean':col_1, 'Waking Mean':col_2})
summary_metrics = summary_metrics.set_index('Measure')
summary_metrics = summary_metrics.style.format(precision=2)
summary_metrics.to_latex(report_folder + 'summarytable.tex')
#%%
reload(wbhsmm)
x = wbhsmm.remove_big_gaps(df_per_minute)
#%%
reload(wbhsmm)

all_dfs  = []
for s in df_minute_all.subj.unique():
    
    df = wbhsmm.getNonCircadianComp(df_minute_all[df_minute_all.subj==s], obs)

    all_dfs.append(df)
obsnoncirc = [ob + '_non_circ' for ob in obs]
#%%
obs_with_circ = []
for ob in obs:
    obs_with_circ.append(ob+'_circ')
for i, df_all_subj in enumerate(all_dfs):
    df_all_subj['subj'] = subj
#%%
for i, df_all_subj in enumerate(all_dfs):
    df_all_subj[to_plot + obs_with_circ+ ['subj']].to_csv('/home/campus.ncl.ac.uk/nct76/Documents/Data/CNNP_Biorhtyhms/report_' + str(df_all_subj.subj.values[0]) + '.csv')
 

#%%
#%%
df_all = df_all_subj.reset_index()
colours = colors
circ_labels = {'activity_counts_circ':'Rest Activity Cycle', 'temperature_celsius_circ':'Circadian Temperature (°C)',
               'eda_scl_usiemens_circ': 'Circadian EDA (nS)', 'glucose_circ':'Circadian Glucose (mmol/L)',
'HR_circ':'Circadian Heart Rate (bpm)', 'pulse_rate_bpm_circ': 'Circadian Pulse Rate (bpm)',
'SDNNI_circ':'Circadian Heart Rate Variability (bpm)', 'CBT_circ': 'Circadian Core Temperature (°C)'}
circ_labels_short = {'activity_counts_circ':'RAC', 'temperature_celsius_circ':'Skin Temp',
               'eda_scl_usiemens_circ': 'EDA', 'glucose_circ':'Glucose',
'HR_circ':'HR', 'pulse_rate_bpm_circ': 'HR',
'SDNNI_circ':'HRV', 'CBT_circ': 'CBT'}
ax1 = plt.subplot(len(obs_with_circ)+1, 1,1)
plt.plot(df_all.t, df_all[obs_with_circ[0]], label=circ_labels[obs_with_circ[0]], color=colours[0])
plt.tick_params( labelbottom = False, bottom = False) 
xmin, xmax = ax1.get_xlim()
days = np.arange(0, np.ceil(xmax)+2)
night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_all.t]
ax1.fill_between(df_all.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.5)
plt.ylabel(circ_labels[obs_with_circ[0]], rotation=45, loc='top')
plt.legend()

for i, c in enumerate(obs_with_circ):
    ax = plt.subplot(len(obs_with_circ)+1, 1,i+2, sharex=ax1)
    plt.plot(df_all.t, df_all[c], label=circ_labels[c], color=colours[0])
    xmin, xmax = ax.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_all.t]
    ax.fill_between(df_all.t, *ax.get_ylim(), where=night, facecolor='k', alpha=.5)
    if i<4:
        plt.tick_params( labelbottom = False, bottom = False) 
    plt.ylabel(circ_labels[c], rotation=45, loc='top')
    plt.legend()
fig = plt.gcf()
fig.set_size_inches(22, 9)
fig.savefig(report_folder+'circadian_all.pdf',bbox_inches='tight')
plt.close()

import scipy.signal as sig
import seaborn as sns
def calculatePLV(x,y):
    phi_x = np.angle(sig.hilbert(x))
    phi_y = np.angle(sig.hilbert(y))
    phase_locking=np.abs(np.sum(np.exp(-1j *(phi_x-phi_y))))/len(phi_x)
    return phase_locking
cols = obs_with_circ
cols_short = [circ_labels_short[c] for c in cols]

inst_phases_all = []

for df in all_dfs:
    plvs = []
    for c in cols:
        plv = []
        for ci in cols:
            plv.append(np.mean(calculatePLV(df[c], df[ci])))
        plvs.append(plv)
    inst_phases = np.array(plvs)
    inst_phases_all.append(inst_phases)
inst_phases_all = np.array(inst_phases_all)
plv_per_person = []
for i in range(len([subj])):
    plv_per_person.append(np.mean(inst_phases_all[i,:,:]))
df2 = pd.DataFrame({'subj':[subj], 'plv':plv_per_person})
sns.boxplot(data=df2, y='plv', color='grey', showfliers=False)
sns.stripplot(data=df2, y='plv', color='k')
plt.ylabel('Mean Phase Locking Value')
plt.ylim([0,1])
plt.savefig(report_folder+'circadian_phase_locking.pdf',bbox_inches='tight')


inst_phases_mean = np.mean(inst_phases_all,axis=0)
mask = np.zeros_like(inst_phases_mean)
mask[np.triu_indices_from(mask)] = True

# Want diagonal elements as well
mask[np.diag_indices_from(mask)] = False
cmap = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(inst_phases_mean, xticklabels=cols_short, yticklabels=cols_short, annot=True,mask=mask, square=True, linewidths=.5, cmap=cmap)
plt.savefig(report_folder+'circadian_phase_locking_heatmap.pdf',bbox_inches='tight')
#%%

acrophases = {}
bathyphases = {}
amplitudes = {}
for c, cname in zip(cols, cols_short):
    phi_x = np.angle(sig.hilbert(df_all[c]))
    zero_crossings = np.where(np.diff(np.sign(phi_x))>0)[0]
    zero_crossings_down = np.where(np.diff(np.sign(phi_x))<0)[0]

    if len(zero_crossings)>0:
        acros = np.mean(pd.to_datetime(df_all.loc[zero_crossings].t.dt.time.astype(str))).time().replace(microsecond=0)
        acrophases[cname] = str(acros)
        amplitudes[cname] = [np.mean(df_all.loc[zero_crossings, c])]
    if len(zero_crossings_down)>0:
        bathy = np.mean(pd.to_datetime(df_all.loc[zero_crossings_down].t.dt.time.astype(str))).time().replace(microsecond=0)
        bathyphases[cname] = str(bathy)


table_df = pd.DataFrame({'Measure':acrophases.keys(), 'Acrophase':acrophases.values(), 'Bathyphase':bathyphases.values()})
table_df = table_df.set_index(['Measure'])
table_df = table_df.style.format(precision=2)
table_df.to_latex(report_folder + 'datatable.tex')

#%%
t_of_day = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
            '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
            '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']

df_minute_all['hour'] = df_minute_all.t.dt.hour
if LYS:
    h_of_day = np.arange(0,24)
    times_xticks = [0,4,8,12,16,20]
    t_labs = [t_of_day[i] for i in times_xticks]
    plt.bar(h_of_day, df_minute_all.groupby('hour').Lux.mean(), color='orange')
    plt.xticks(times_xticks, t_labs, rotation=45)
    xmin, xmax = ax.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(h<7)|(h>=20) for h in h_of_day]
    ax = plt.gca()
    ax.fill_between(h_of_day, *ax.get_ylim(), where=night, facecolor='k', alpha=.5)
    plt.ylabel('Brightness (Lux)')
    plt.savefig(report_folder+'average_light_levels.pdf',bbox_inches='tight')
    plt.close()
    blue_light_per_hour = df_minute_all.groupby('hour').Blue_light.sum()/(df_minute_all.groupby('hour').Blue_light.count()/60)
    plt.bar(h_of_day, blue_light_per_hour)
    plt.xticks(times_xticks, t_labs, rotation=45)
    plt.ylabel('Blue Light (minutes)')
    xmin, xmax = ax.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(h<7)|(h>=20) for h in h_of_day]
    ax = plt.gca()
    ax.fill_between(h_of_day, *ax.get_ylim(), where=night, facecolor='k', alpha=.5)
    plt.savefig(report_folder+'blue_light_levels.pdf',bbox_inches='tight')