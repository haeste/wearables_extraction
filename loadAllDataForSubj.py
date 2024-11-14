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
import pyedflib
import scipy as scp
import os
import proj_funcs.generic_funcs as generic_funcs
import proj_funcs.FilterEEG as filter_funcs
import proj_funcs.cycles_funcs as cycles_funcs
import utilities.plotting as plotting
import utilities.file_reading as file_reading 
import utilities.wearables_hsmm as wbhsmm
import biorhythms
import datetime
import datefinder


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
data_dir = '/home/campus.ncl.ac.uk/nct76/Documents/BiorhythmsData/'
#data_dir = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23'
subjects = ['S003', 'S005']
#subjects = ['S002']

obs = ['activity_counts', 'temperature_celsius', 'eda_scl_usiemens', 
                                  'glucose', 'HR', 'SDNNI']
df_minute_all = pd.DataFrame()
df_agg_measure = pd.DataFrame()
for subj in subjects:
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

    subj_cgm = file_reading.read_cgm(os.path.join(subj_dir,'CGM/'))
    subj_cgm = subj_cgm.reset_index()
    subj_cgm = subj_cgm.set_index('t')
    subj_cgm = subj_cgm.resample('1T').mean()
    subj_cgm = subj_cgm.interpolate()
    subj_cgm = subj_cgm.reset_index()
    df_per_minute = pd.merge(df_per_minute, subj_cgm[['t', 'glucose']], on='t', how='outer')
    del subj_cgm
    ecgs, ecg_times, hrvs, hrv_times = file_reading.read_Bittium_Data(os.path.join(subj_dir,'Bittium/'))
    BITData = calc_HR_HRV(hrvs, hrv_times)
    #GMT = (BITData.t > np.datetime64('2023-10-29T02:00:00')) & (BITData.t < np.datetime64('2024-03-30T02:00:00')) 
    #BITData.loc[GMT,'t'] = BITData.loc[GMT,'t'] - np.timedelta64(60, 'm')
    df_per_minute = pd.merge(df_per_minute, BITData[['t', 'HR', 'SDNNI']], on='t', how='outer')
    df_per_minute = wbhsmm.remove_big_gaps(df_per_minute)
    df_per_minute = df_per_minute[df_per_minute.big_gap==0].reset_index()
    beginning = df_per_minute.first_valid_index()
    ending = df_per_minute.last_valid_index()
    df_per_minute = df_per_minute[beginning:ending]
    
    df_per_minute['subj'] = subj
    
    df_minute_all = pd.concat([df_minute_all, df_per_minute])
    del ecgs, ecg_times, hrvs, hrv_times, BITData

#%%
colours = ['lightcoral', 'goldenrod', 'lightseagreen', 'royalblue', 'darkgreen', 'peru']
ax1 = plt.subplot(6, 1,1)
plt.plot(df_per_minute.t, df_per_minute['activity_counts'], label='activity_counts', color=colours[0])
plt.tick_params( labelbottom = False, bottom = False) 
xmin, xmax = ax1.get_xlim()
days = np.arange(0, np.ceil(xmax)+2)
night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_per_minute.t]
ax1.fill_between(df_per_minute.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.5)
#plt.ylabel('activity_counts')
plt.legend()
for i, c in enumerate(['temperature_celsius', 'eda_scl_usiemens', 'glucose',
'HR', 'SDNNI']):
    ax = plt.subplot(6, 1,i+2, sharex=ax1)
    plt.plot(df_per_minute.t, df_per_minute[c], label=c, color=colours[i+1])
    xmin, xmax = ax.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_per_minute.t]
    ax.fill_between(df_per_minute.t, *ax.get_ylim(), where=night, facecolor='k', alpha=.5)
    if i<4:
        plt.tick_params( labelbottom = False, bottom = False) 
    #plt.ylabel(c)
    plt.legend()
plt.show()
#%%
reload(wbhsmm)
df6 = df_minute_all[df_minute_all.subj=='S006']
x = wbhsmm.remove_big_gaps(df_per_minute)
#%%
ethica_diary = {}
ethica_sleep = {}
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    diary_file = os.path.join(subj_dir,'Ethica/', subj+'_Diary.csv')
    sleep_file = os.path.join(subj_dir,'Ethica/', subj+'_Sleep.csv')
    if os.path.exists(diary_file):
        diary = file_reading.get_ethica_diary(diary_file)
        ethica_diary[subj] = diary
    if os.path.exists(sleep_file):
        sleep = file_reading.get_ethica_sleep(sleep_file)
        ethica_sleep[subj] = sleep


#%%
reload(wbhsmm)

all_dfs  = []
for s in df_minute_all.subj.unique():
    
    df = wbhsmm.getNonCircadianComp(df_minute_all[df_minute_all.subj==s], obs)

    all_dfs.append(df)
obsnoncirc = [ob + '_non_circ' for ob in obs]
#%%

for i, df_all_subj in enumerate(all_dfs):
    df_all_subj['subj'] = subjects[i]
#%%
for i, df_all_subj in enumerate(all_dfs):
    df_all_subj[['sleep_detection_stage', 'activity_counts', 'temperature_celsius',
           'eda_scl_usiemens', 'glucose', 'HR', 'SDNNI',
           'activity_counts_circ',
           'temperature_celsius_circ',
           'eda_scl_usiemens_circ',
           'glucose_circ', 'HR_circ',
           'SDNNI_circ', 'subj']].to_csv('/home/campus.ncl.ac.uk/nct76/Documents/Data/CNNP_Biorhtyhms/' + str(df_all_subj.subj.values[0]) + '.csv')
    

#%%
from importlib import reload
reload(wbhsmm)
np.random.seed(seed=0)
df, model, models = wbhsmm.getStates(all_dfs, obsnoncirc)
wbhsmm.plotStates(df, model, ['Acc', 'Temp', 'EDA', 'Glucose', 'HR', 'HRV'])
#wbhsmm.visualise_training(models, '/home/campus.ncl.ac.uk/nct76/Pictures/animations/')
#%%
import pickle
file = open("/home/campus.ncl.ac.uk/nct76/Documents/Code/BiologicalRhythms/hsmm_model1.pickle",'rb')
model = pickle.load(file)
file = open("/home/campus.ncl.ac.uk/nct76/Documents/Code/BiologicalRhythms/df_1.pickle",'rb')
df = pickle.load(file)
#%%
subjects = ['S016']

all_subjs = []
for i, df_subj in enumerate(df):
    df_subj['subj'] = subjects[i]
    all_subjs.append(df_subj)
all_subjs = pd.concat(all_subjs)
nans = all_subjs[['activity_counts', 'temperature_celsius',
      'eda_scl_usiemens', 'glucose', 'HR', 'SDNNI',]].isna().any(axis=1)
all_subjs.loc[nans,'state'] = np.nan
total_samples = len(all_subjs.dropna())
states_proportions = all_subjs.groupby(['state']).subj.count()/total_samples
num_per_subj = all_subjs.groupby(['state', 'subj']).subj.count()
print(num_per_subj.groupby('state').count())
as_cumanta = states_proportions.index[np.argsort(list(states_proportions))]
states_used = [1,13,4,6,15,21,11,16]

as_cumanta_dhachaidh = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                        'tab:gray', 'tab:olive', 'tan']
wbhsmm.plotStates(df, model, ['Acc', 'Temp', 'EDA', 'Glucose', 'HR', 'HRV'], colors=as_cumanta_dhachaidh)
#%%
s = all_subjs[all_subjs.subj=='S002'].reset_index()
ax = plt.subplot(1,1,1)
plt.plot(s.t, s.HR)
plt.plot(s.t, s.HR_circ+s.HR.median(), 'k')
xmin, xmax = ax.get_xlim()
days = np.arange(0, np.ceil(xmax)+2)
night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in s.t]
ax.fill_between(s.t, *ax.get_ylim(), where=night, facecolor='k', alpha=.1)
ax.set_xlim(xmin, xmax) # set limits back to default values
plt.ylabel('HR (BPM)')
plt.show()
#%%
s = all_subjs[all_subjs.subj=='S002'].reset_index()
ax = plt.subplot(1,1,1)
plt.plot(s.t, s.HR_non_circ)
xmin, xmax = ax.get_xlim()
days = np.arange(0, np.ceil(xmax)+2)
night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in s.t]
ax.fill_between(s.t, *ax.get_ylim(), where=night, facecolor='k', alpha=.1)
ax.set_xlim(xmin, xmax) # set limits back to default values
plt.ylabel('HR (BPM)')
plt.show()
#%%
states_used = np.squeeze(np.where(model.state_usages>0.02))
states_used = list(as_cumanta.astype(int))
states_used[9] = 11
states_used[8] = 22
states_used = [1,13,4,6,15,21,11,16]
num_states = len(states_used)
plt.show()
phis = {}
for ii, i in enumerate(states_used):
    phis[i] = []
for subj_df in df:
    df_temp = subj_df.reset_index()
    for s in states_used:
        state_times = df_temp[df_temp.state==s].t
        for st in state_times:
            minutes = st.hour*60 + st.minute
            phis[s].append(minutes/1440.0 * (2*np.pi))

#%%
#phis = np.array(phis)
fig = plt.figure()

for ii, i in enumerate(states_used):
    ax = fig.add_subplot(2,int(np.ceil(len(states_used)/2)),ii+1,projection='polar')
    
    ax.set_ylim([0,np.max(np.histogram(phis[i])[0])])
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_xticklabels(['00:00', '','06:00','', '12:00','', '18:00',''])
    plt.title('state ' + str(i))
    plt.hist(phis[i], color=as_cumanta_dhachaidh[ii])
plt.show()
#%%
import seaborn as sns
all_subjs_phi = all_subjs.reset_index()
states_used = as_cumanta
all_subjs_phi = all_subjs_phi[all_subjs_phi.state.isin(states_used)]
all_subjs_phi['phi'] = (all_subjs_phi.t.dt.hour*60 + all_subjs_phi.t.dt.minute)/1440.0 * (2*np.pi)
g = sns.FacetGrid(all_subjs_phi, col="state",col_wrap=6,
              subplot_kws=dict(projection='polar'),
              sharex=False, sharey=False, despine=False)
g.map_dataframe(sns.histplot, x="phi",hue='subj', multiple="stack")
g.set(theta_direction=-1)
g.set(theta_zero_location="N")
g.set(ylim=[0,np.max(np.histogram(phis[i])[0])])
g.set_xticklabels(['00:00', '','06:00','', '12:00','', '18:00',''])
g.set_ylabels(label='')

g.add_legend()
plt.show()
#%%
curr_subj = all_subjs_phi[all_subjs_phi.subj=='S010']
curr_subj.set_index('t', inplace=True)
curr_subj[['activity_counts_non_circ', 'temperature_celsius_non_circ',
       'eda_scl_usiemens_non_circ', 'glucose_non_circ', 'HR_non_circ',
       'SDNNI_non_circ', 'sleep', 'state']].plot()
plt.show()
#%%
s = all_subjs[all_subjs.subj=='S004'].reset_index()
ax = plt.subplot(1,1,1)
plt.plot(s.t, s.activity_counts_non_circ)
plt.plot(s.t, s.temperature_celsius_non_circ + 5*1)
plt.plot(s.t, s.eda_scl_usiemens_non_circ + 5*2)
plt.plot(s.t, s.glucose_non_circ  + 5*3)
plt.plot(s.t, s.HR_non_circ  + 5*4)
plt.plot(s.t, s.SDNNI_non_circ  + 5*5)
plt.tick_params(labelleft=False, left=False)
xmin, xmax = ax.get_xlim()
days = np.arange(0, np.ceil(xmax)+2)
night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in s.t]
ax.fill_between(s.t, *ax.get_ylim(), where=night, facecolor='k', alpha=.1)
ax.set_xlim(xmin, xmax) # set limits back to default values
colorthings = ['tan', 'teal']
for i, cs in enumerate(as_cumanta):
    night = [(sshtate==cs) for sshtate in s.state]
    ax.fill_between(s.t, *ax.get_ylim(), where=night, facecolor=as_cumanta_dhachaidh, alpha=.9)
    ax.set_xlim(xmin, xmax) # set limits back to default values
plt.ylabel('z-score')
plt.show()
#%%
states_used = np.squeeze(np.where(model.state_usages>0.02))

s = all_subjs[all_subjs.subj=='S004'].reset_index()
ax = plt.subplot(1,1,1)
plt.plot(s.t, s.activity_counts_non_circ, 'k')
plt.plot(s.t, s.temperature_celsius_non_circ + 5*1,  'k')
plt.plot(s.t, s.eda_scl_usiemens_non_circ + 5*2, 'k')
plt.plot(s.t, s.glucose_non_circ  + 5*3,  'k')
plt.plot(s.t, s.HR_non_circ  + 5*4,  'k')
plt.plot(s.t, s.SDNNI_non_circ  + 5*5,  'k')
plt.tick_params(labelleft=False, left=False)
xmin, xmax = ax.get_xlim()
days = np.arange(0, np.ceil(xmax)+2)
night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in s.t]
ax.fill_between(s.t, *ax.get_ylim(), where=night, facecolor='k', alpha=.1)
ax.set_xlim(xmin, xmax) # set limits back to default values
colorthings = ['tan', 'teal']
cmap = plt.get_cmap('tab20')
coloursplt = cmap.colors
for i, cs in enumerate(states_used[:20]):
    night = [(sshtate==cs) for sshtate in s.state]
    ax.fill_between(s.t, *ax.get_ylim(), where=night, facecolor=cmap.colors[i], alpha=.7)
    ax.set_xlim(xmin, xmax) # set limits back to default values
plt.ylabel('z-score')
plt.show()
#%%
s = all_subjs[all_subjs.subj=='S004'].reset_index()


#as_cumanta_dhachaidh = ['r', 'b', 'g', 'y', 'purple', 'cyan']
for i, cs in enumerate([as_cumanta[4]]):
    ax = plt.subplot(5,2,i+1)
    plt.plot(s.t, s.activity_counts_non_circ)
    plt.plot(s.t, s.temperature_celsius_non_circ + 5*1)
    plt.plot(s.t, s.eda_scl_usiemens_non_circ + 5*2)
    plt.plot(s.t, s.glucose_non_circ  + 5*3)
    plt.plot(s.t, s.HR_non_circ  + 5*4)
    plt.plot(s.t, s.SDNNI_non_circ  + 5*5)
    night = [(sshtate==cs) for sshtate in s.state]
    ax.fill_between(s.t, *ax.get_ylim(), where=night, facecolor=as_cumanta_dhachaidh[i], alpha=.6)
    ax.set_xlim(xmin, xmax) # set limits back to default values
    plt.tick_params(labelleft=False, left=False)
    plt.ylabel('z-score')
    #plt.xlim([x_min, x_max,ax])
plt.show()
#%%
df_all_ethica = pd.DataFrame()
for i, sub in enumerate(subjects):  
    if sub in ethica_diary.keys():
        curr_df = df[i].copy()
        curr_df['subj'] = sub
        ethica_d = ethica_diary[sub]
        ethica_d['t5min'] = ethica_d['t'].dt.round('5min')
        ethica_d = ethica_d.set_index('t5min')
        df_merged = ethica_d.merge(curr_df, left_index=True, right_index=True, how='outer')
        df_merged['Activation'] = df_merged.Activation - df_merged.Activation.mean()
        df_merged['Activation2'] = df_merged.Activation2 - df_merged.Activation2.mean()
        
        df_all_ethica = pd.concat([df_all_ethica, df_merged])
df_all_ethica.loc[df_all_ethica.subj=='S003', 'Activation'] = 2*(df_all_ethica.loc[df_all_ethica.subj=='S003', 'Activation']/100)-1
df_all_ethica.groupby(['subj', 'state']).Activation.mean()
#for pa in 
#%%
for s in subjects:
    df_subj_ethica = df_all_ethica[df_all_ethica.subj==s]
    
subjStateAct1Means = df_all_ethica.groupby(['subj', 'state']).Activation.mean()
subjStateAct2Means = df_all_ethica.groupby(['subj', 'state']).Activation2.mean()
sns.boxplot(subjStateAct1Means.reset_index(), x='state', y='Activation')
#sns.stripplot(subjStateAct1Means.reset_index(), x='state', y='Activation')
plt.ylim([-2,2])
plt.show()
sns.boxplot(subjStateAct2Means.reset_index(), x='state', y='Activation2')

#sns.stripplot(subjStateAct2Means.reset_index(), x='state', y='Activation2')
plt.ylim([-2,2])
plt.show()

#%%
excertimes = df_all_ethica[['subj', '[6_CAL] Exercise Time']].dropna()
df_all_ethica['Exercise'] = False
excertimes['exTime'] = excertimes['[6_CAL] Exercise Time']
for i,e in zip(excertimes.index,excertimes.exTime):
    x = list(datefinder.find_dates(e))
    if len(x)<2:
        continue
    if x[0].date() == datetime.datetime.today().date():
        start= i
        stop=i
        start = np.datetime64(start.replace(hour=x[0].hour, minute=x[0].minute))
        stop = np.datetime64(stop.replace(hour=x[1].hour, minute=x[1].minute))
    else:
        start = np.datetime64(x[0])
        stop = np.datetime64(x[1])
    df_all_ethica.loc[(df_all_ethica.index>=start) & (df_all_ethica.index<=stop), 'Exercise'] = True
    
df_all_ethica[df_all_ethica.Exercise].state.hist(bins=26)

plt.xlabel('state')

plt.ylabel('Epochs')
plt.show()
#%%
mealtimes = df_all_ethica[['subj', '[4_CAL] When did you eat?']].dropna()
df_all_ethica['Mealtime'] = False
mealtimes['mTime'] = mealtimes['[4_CAL] When did you eat?']

for i,e in zip(mealtimes.index,mealtimes.mTime):
    x = list(datefinder.find_dates(e))

    if x[0].date() == datetime.datetime.today().date():
        start= i
        stop=i
        start = np.datetime64(start.replace(hour=x[0].hour, minute=x[0].minute))
        stop = start + np.timedelta64(1,'h')
    else:
        start = np.datetime64(x[0])
        stop = start + np.timedelta64(1,'h')
    df_all_ethica.loc[(df_all_ethica.index>=start) & (df_all_ethica.index<=stop), 'Mealtime'] = True
    
df_all_ethica[df_all_ethica.Mealtime].state.hist(bins=26)

plt.xlabel('state')
plt.ylabel('Epochs')
#plt.show()

#%%
cafetimes = df_all_ethica[['subj', 'CaffeineTime']].dropna()
df_all_ethica['cafetimes'] = False

for i,e in zip(cafetimes.index,cafetimes.CaffeineTime):
    x = list(datefinder.find_dates(e))

    if x[0].date() == datetime.datetime.today().date():
        start= i
        stop=i
        start = np.datetime64(start.replace(hour=x[0].hour, minute=x[0].minute))
        stop = start + np.timedelta64(1,'h')
    else:
        start = np.datetime64(x[0])
        stop = start + np.timedelta64(1,'h')
    df_all_ethica.loc[(df_all_ethica.index>=start) & (df_all_ethica.index<=stop), 'cafetimes'] = True
    
df_all_ethica[df_all_ethica.cafetimes].state.hist(bins=26)

plt.xlabel('state')
plt.ylabel('Epochs')
#plt.show()

#%%
alctimes = df_all_ethica[['subj', 'AlcoholTime']].dropna()
df_all_ethica['alctimes'] = False

for i,e in zip(alctimes.index,alctimes.AlcoholTime):
    x = list(datefinder.find_dates(e))

    if x[0].date() == datetime.datetime.today().date():
        start= i
        stop=i
        start = np.datetime64(start.replace(hour=x[0].hour, minute=x[0].minute))
        stop = start + np.timedelta64(1,'h')
    else:
        start = np.datetime64(x[0])
        stop = start + np.timedelta64(1,'h')
    df_all_ethica.loc[(df_all_ethica.index>=start) & (df_all_ethica.index<=stop), 'alctimes'] = True
    
df_all_ethica[df_all_ethica.alctimes].state.hist(bins=26)

plt.xlabel('state')
plt.ylabel('Epochs')
plt.show()
#%%
sns.stripplot(subjStateAct2Means.reset_index(), x='state', y='Activation2', hue='subj')
plt.show()
#%%
x = ethica_data['S007']
x.loc[x.Activation>1.1,'Activation'] =  x.loc[x.Activation>1.1,'Activation']/100.0
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S007'], measure='Activation', phase='PR_phase')
plt.show()
#%%
x = ethica_data['S007']
x.loc[x.Activation>1.1,'Activation2'] =  x.loc[x.Activation>1.1,'Activation2']/100.0
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S007'], measure='Activation2', phase='PR_phase')
plt.show()
#%%
x = ethica_data['S004']
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S004'], measure='Activation', phase='PR_phase')
plt.show()
#%%
x = ethica_data['S004']
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S004'], measure='Activation2', phase='PR_phase')
plt.show()

#%%
x = ethica_data['S004']
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S004'], measure='Activation', phase='GLUC_phase')
plt.show()
#%%
x = ethica_data['S004']
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S004'], measure='Activation2', phase='GLUC_phase')
plt.show()

#%%
x = ethica_data['S005']
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S005'], measure='Activation', phase='GLUC_phase')
plt.show()
#%%
x = ethica_data['S005']
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S005'], measure='Activation2', phase='RMSSD_phase')
plt.show()
#%%
cgm004 = file_reading.read_cgm('/home/campus.ncl.ac.uk/nct76/Documents/BiorhythmsData/S004/CGM/')
cgm004 = cgm004.reset_index()
cgm004 = addCircadianComponent(cgm004, 'glucose', (1/15)/60)
plotCGM(cgm004)
plt.show()
#%%

diary004 = file_reading.get_ethica_diary('/home/campus.ncl.ac.uk/nct76/Documents/BiorhythmsData/S004/Ethica/S004_Diary.csv')
plot_ethica_polar(diary004, cgm004, 'Activation')
plt.show()
#%%
plot_ethica(diary004, cgm004, 'Activation')
plt.show()