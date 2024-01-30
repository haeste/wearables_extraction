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

    noise = np.array([sum(r>0) for r in rrs_err])
    RMSSD = np.array([np.sqrt(sum((r[:-1] - r[1:])**2)/(len(r)-1)) if len(r)>100 else np.nan for r in rrs])
    hr = np.array([60*1000*(1/np.mean(r)) if len(r)>5 else np.nan for r in rrs_hr])
    rdff_t = np.arange(0,len(RMSSD))*(60 *hrv_calc_len)
    rdff_hr_t = np.arange(0,len(hr))*(hr_calc_len*60)

    rdff_t_ms = rdff_t*1000
    rdff_hr_t_ms = rdff_hr_t*1000
    rdff_t_ms =np.datetime64(hrv_times[0]).astype('datetime64[ms]') + rdff_t_ms.astype(int).astype('timedelta64[ms]')
    rdff_hr_t_ms =np.datetime64(hrv_times[0]).astype('datetime64[ms]') + rdff_hr_t_ms.astype(int).astype('timedelta64[ms]')
    
    RMSSD = pd.Series(data=RMSSD, index=rdff_t_ms, name='RMSSD')
    RMSSD = RMSSD.resample('1T').mean()
    RMSSD = RMSSD.interpolate(limit=5)
    
    hr = pd.Series(data=hr, index=rdff_hr_t_ms, name='HR')
    hr = hr.resample('1T').mean()
    hr = hr.interpolate(limit=5)
    
    hr = hr.to_frame()
    hr['HR'] = hr
    hr['RMSSD'] = RMSSD
    hr['t'] = hr.index
    hr = hr.reset_index()
    
    BITData = hr[['t', 'HR', 'RMSSD']]
    
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
data_dir = '/home/campus.ncl.ac.uk/nct76/Documents/BiorhythmsData/'
subjects = ['S002', 'S003', 'S004', 'S005']
df_minute_all = pd.DataFrame()
df_agg_measure = pd.DataFrame()
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    
    subj_pr = file_reading.read_Empatica_DBM(os.path.join(subj_dir,'Empatica/'), 'pulse-rate')
    df_per_minute = subj_pr[['t', 'pulse_rate_bpm']]
    
    subj_temp = file_reading.read_Empatica_DBM(os.path.join(subj_dir,'Empatica/'), 'temperature')
    df_per_minute = pd.merge(df_per_minute, subj_temp[['t', 'temperature_celsius']], on='t', how='outer')
    
    subj_eda = file_reading.read_Empatica_DBM(os.path.join(subj_dir,'Empatica/'), 'eda')
    df_per_minute = pd.merge(df_per_minute, subj_eda[['t', 'eda_scl_usiemens']], on='t', how='outer')
    
    subj_cgm = file_reading.read_cgm(os.path.join(subj_dir,'CGM/'))
    subj_cgm = subj_cgm.reset_index()
    subj_cgm = subj_cgm.set_index('t')
    subj_cgm = subj_cgm.resample('1T').mean()
    subj_cgm = subj_cgm.interpolate()
    subj_cgm = subj_cgm.reset_index()
    df_per_minute = pd.merge(df_per_minute, subj_cgm[['t', 'glucose']], on='t', how='outer')
    
    ecgs, ecg_times, hrvs, hrv_times = file_reading.read_Bittium_Data(os.path.join(subj_dir,'Bittium/'))
    BITData = calc_HR_HRV(hrvs, hrv_times)
    df_per_minute = pd.merge(df_per_minute, BITData[['t', 'HR', 'RMSSD']], on='t', how='outer')
    df_per_minute['subj'] = subj
    
    agg_measures = {}
    peaktimes, rhythm, phase, peakamplitudes, peak_period_var, peak_amp_var = getCWTParameters(df_per_minute, 'pulse_rate_bpm')
    df_per_minute['PR_circadian'] = rhythm
    df_per_minute['PR_phase'] = phase

    agg_measures['PR_per_var'] = [peak_period_var]
    agg_measures['PR_amp_var'] = [peak_amp_var]
    agg_measures['PR_peaktime'] = [np.median((peaktimes.dt.hour*60)+peaktimes.dt.minute)/60]

    peaktimes, rhythm, phase, peakamplitudes, peak_period_var, peak_amp_var = getCWTParameters(df_per_minute, 'temperature_celsius')
    df_per_minute['TEMP_circadian'] = rhythm
    df_per_minute['TEMP_phase'] = phase

    agg_measures['TEMP_per_var'] = [peak_period_var]
    agg_measures['TEMP_amp_var'] = [peak_amp_var]
    agg_measures['TEMP_peaktime'] = [np.median((peaktimes.dt.hour*60)+peaktimes.dt.minute)/60]

    peaktimes, rhythm, phase, peakamplitudes, peak_period_var, peak_amp_var = getCWTParameters(df_per_minute, 'eda_scl_usiemens')
    df_per_minute['EDA_circadian'] = rhythm
    df_per_minute['EDA_phase'] = phase

    agg_measures['EDA_per_var'] = [peak_period_var]
    agg_measures['EDA_amp_var'] = [peak_amp_var]
    agg_measures['EDA_peaktime'] = [np.median((peaktimes.dt.hour*60)+peaktimes.dt.minute)/60]

    peaktimes, rhythm, phase, peakamplitudes, peak_period_var, peak_amp_var = getCWTParameters(df_per_minute, 'glucose')
    df_per_minute['GLUC_circadian'] = rhythm
    df_per_minute['GLUC_phase'] = phase

    agg_measures['GLUC_per_var'] = [peak_period_var]
    agg_measures['GLUC_amp_var'] = [peak_amp_var]
    agg_measures['GLUC_peaktime'] = [np.median((peaktimes.dt.hour*60)+peaktimes.dt.minute)/60]

    peaktimes, rhythm, phase, peakamplitudes, peak_period_var, peak_amp_var = getCWTParameters(df_per_minute, 'HR')
    df_per_minute['HR_circadian'] = rhythm
    df_per_minute['HR_phase'] = phase

    agg_measures['HR_per_var'] = [peak_period_var]
    agg_measures['HR_amp_var'] = [peak_amp_var]
    agg_measures['HR_peaktime'] = [np.median((peaktimes.dt.hour*60)+peaktimes.dt.minute)/60]

    peaktimes, rhythm, phase, peakamplitudes, peak_period_var, peak_amp_var = getCWTParameters(df_per_minute, 'RMSSD')
    df_per_minute['RMSSD_circadian'] = rhythm
    df_per_minute['RMSSD_phase'] = phase

    agg_measures['RMSSD_per_var'] = [peak_period_var]
    agg_measures['RMSSD_amp_var'] = [peak_amp_var]
    agg_measures['RMSSD_peaktime'] = [np.median((peaktimes.dt.hour*60)+peaktimes.dt.minute)/60]
    agg_measures['subj'] = [subj]
    
    
    df_agg_measure = pd.concat([df_agg_measure, pd.DataFrame(agg_measures)])
    df_minute_all = pd.concat([df_minute_all, df_per_minute])
#%%

ethica_data = {}
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    diary_file = os.path.join(subj_dir,'Ethica/', subj+'_Diary.csv')
    if os.path.exists(diary_file):
        diary = file_reading.get_ethica_diary(diary_file)
        ethica_data[subj] = diary

#%%
x = ethica_data['S003']
x.loc[x.Activation>1.1,'Activation'] =  x.loc[x.Activation>1.1,'Activation']/100.0
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S003'], measure='Activation', phase='PR_phase')
plt.show()
#%%
x = ethica_data['S003']
x.loc[x.Activation>1.1,'Activation2'] =  x.loc[x.Activation>1.1,'Activation2']/100.0
plotting.plot_ethica_polar(x, df_minute_all[df_minute_all.subj=='S003'], measure='Activation2', phase='PR_phase')
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