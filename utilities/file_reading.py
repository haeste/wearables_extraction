#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:49:19 2023

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
import biorhythms
import more_itertools as mit
from avro.datafile import DataFileReader
from avro.io import DatumReader

def read_Empatica_DBM(filepath, var):
    
    dbm_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(filepath)) for f in fn if f.endswith(var+'.csv')]
    df_dbm = pd.DataFrame()
    if len(dbm_files) == 0:
        raise FileNotFoundError('No Empatica files found at ' + filepath + ' with values ' + var)
        
    for f in dbm_files:
        df_i = pd.read_csv(f)
        df_dbm = pd.concat([df_dbm, df_i])
    
    df_dbm['t'] = pd.to_datetime(df_dbm.timestamp_unix, unit='ms')
    
    df_dbm.sort_values(by='t', ascending=True, inplace=True)
    df_dbm.reset_index(inplace=True,drop=True)
    return df_dbm

def correct_Empatica_Offset(df, meta_path):
    meta = pd.read_csv(meta_path)
    meta['t'] = pd.to_datetime(meta.timestamp_unix, unit='ms')
    meta = meta[meta.time_offset.notna()]
    seconds_offset = meta.time_offset
    for i in range(len(meta)):
        start = np.datetime64(pd.to_datetime(meta.iloc[i,0], unit='ms')).astype('datetime64[s]')
        if i+1<len(meta):
            end = np.datetime64(pd.to_datetime(meta.iloc[i+1,0], unit='ms')).astype('datetime64[s]')
        else:
            end = max(df.t)
        shift = np.timedelta64(int(meta.iloc[i,2]), 's')
        shift_min = str(shift.astype('timedelta64[m]'))
        print('Timezone change: ' + str(start) + ' to ' + str(end) + ' [' +shift_min + ']')
        if shift >0:
            df.loc[(df.t>=start) & (df.t<=end), 't'] = df.loc[(df.t>=start) & (df.t<=end), 't'] +  shift
        elif shift <0:
            df.loc[(df.t>=start) & (df.t<=end), 't'] = df.loc[(df.t>=start) & (df.t<=end), 't'] +  shift
            #df.loc[(df.t>start) & (df.t<=end), 't'] = df.loc[(df.t>start) & (df.t<=end), 't'] -  shift
    df.sort_values(by='t', ascending=True, inplace=True)
    return df
    

def read_cgm(filepath):
    cgm_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(filepath)) for f in fn if f.endswith('.csv')]
    df_cgm = pd.DataFrame()
    for f in cgm_files:
        df_i = pd.read_csv(f, header=0)
        if 'Historic Glucose mmol/L' not in df_i.columns:
            df_i = pd.read_csv(f, header=1)
        df_i['glucose'] = df_i['Historic Glucose mmol/L']
        df_i['glucose'].fillna(df_i['Scan Glucose mmol/L'], inplace=True)
        df_i = df_i[['Device Timestamp', 'glucose']]
        df_i.dropna(subset=['glucose'], inplace=True)
        df_cgm = pd.concat([df_cgm, df_i])
    df_cgm['t'] = pd.to_datetime(df_cgm['Device Timestamp'], dayfirst=True)
    df_cgm = df_cgm[['t', 'glucose']]
    df_cgm.loc[(df_cgm.t>np.datetime64('2023-11-01T15:30:00')) & (df_cgm.t<np.datetime64('2023-11-04T18:10:00')), 't'] = df_cgm.loc[(df_cgm.t>np.datetime64('2023-11-01T15:30:00')) & (df_cgm.t<np.datetime64('2023-11-04T18:10:00')), 't'] - np.timedelta64(1, 'h')
    df_cgm.loc[(df_cgm.t>np.datetime64('2023-11-29T07:10:00')) & (df_cgm.t<np.datetime64('2023-11-29T22:10:00')), 't'] = df_cgm.loc[(df_cgm.t>np.datetime64('2023-11-29T07:10:00')) & (df_cgm.t<np.datetime64('2023-11-29T22:10:00')), 't'] - np.timedelta64(1, 'h')
    df_cgm.loc[(df_cgm.t>np.datetime64('2023-11-29T22:10:00')) & (df_cgm.t<np.datetime64('2023-12-07T09:00:00')), 't'] = df_cgm.loc[(df_cgm.t>np.datetime64('2023-11-29T22:10:00')) & (df_cgm.t<np.datetime64('2023-12-07T09:00:00')), 't'] + np.timedelta64(5, 'h')
    df_cgm.sort_values(by='t', ascending=True, inplace=True)
    df_cgm = df_cgm.set_index('t')
    df_cgm = df_cgm.resample('15T').mean()
    return df_cgm

def get_ethica_diary(filepath):
    ethica_sub = pd.read_csv(filepath)
    ethica_sub['t'] = pd.to_datetime(ethica_sub['Record Time'],utc=True, format='mixed')
    ethica_sub['t'] = ethica_sub.t.dt.tz_localize(None)
    ethica_sub['t'] = ethica_sub['t'].dt.round('min')
    return ethica_sub

def get_ethica_sleep(filepath):
    ethica_sub = pd.read_csv(filepath)
    ethica_sub['t'] = pd.to_datetime(ethica_sub['Record Time'],utc=True, format='mixed')
    ethica_sub['t'] = ethica_sub.t.dt.tz_localize(None)
    ethica_sub['t'] = ethica_sub['t'].dt.round('min')
    return ethica_sub

def computeENMO(x):
    r = np.sqrt(np.sum(np.power(np.array(x),2),axis=1))-1000
    return r
def computeMAD(acc, duration_in_steps):
    r = np.sqrt(np.sum(np.power(np.array(acc,2),axis=1)))
    mad = np.array([np.mean(np.abs(x)-np.mean(np.abs(x))) for x in mit.chunked(r, duration_in_steps)])
    return mad

def read_Bittium_Data(filepath):
    from random import shuffle
    bittium_folder = filepath
    bittium_files = []
    [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(bittium_folder)) for f in fn]
    bittium_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(bittium_folder)) for f in fn if f.endswith('.EDF')]
    shuffle(bittium_files)
    ecgs = []
    ecg_times = []
    acc = []
    acc_times =[]
    hrvs = []
    hrv_times = []
    
    for f in bittium_files:
        try:
            bittium_data = loadBFEDF(f)
        except IndexError as e:
            print('Error opening file: ' + f)
            print('EDF file does not contain the expected channels.')
            continue
        (ecg, ecg_t_ms) = bittium_data[0] 
        (hrv, hrv_t_ms) = bittium_data[1]
        (accX, accX_t_ms) = bittium_data[2]
        (accY, accY_t_ms) = bittium_data[3]
        (accZ, accZ_t_ms) = bittium_data[4]
        
        ecg_si = ecg_t_ms[1] - ecg_t_ms[0]
        hrv_si = hrv_t_ms[1] - hrv_t_ms[0]
        ecgs.append(ecg)
        ecg_times.append(ecg_t_ms)

        hrvs.append(hrv)
        hrv_times.append(hrv_t_ms)
    
    new_inds = [x for x,y in sorted(enumerate(hrv_times), key = lambda x: x[1][0])]
    ecg_times = [ecg_times[i] for i in new_inds]
    hrv_times = [hrv_times[i] for i in new_inds]
    ecgs = [ecgs[i] for i in new_inds]
    hrvs = [hrvs[i] for i in new_inds]

    count = 0
    for i in range(0,len(hrv_times)-1):
        filler = np.arange(start=hrv_times[i+count][-1] + hrv_si, stop=hrv_times[i+count+1][0], step=hrv_si, dtype='datetime64[ms]')
        count = count+1
        hrv_times.insert(i+count, filler)
        hrvs_filler = np.empty(filler.shape)
        hrvs.insert(i+count,hrvs_filler)
        # add to the start of the list if it occurs before all previous

    ecg_times = np.concatenate(ecg_times, axis=0)
    hrv_times = np.concatenate(hrv_times, axis=0)
    ecgs = np.concatenate(ecgs, axis=0)
    hrvs = np.concatenate(hrvs, axis=0)
        
    return ecgs, ecg_times, hrvs, hrv_times

def loadBFEDF(filename):
    
    f = pyedflib.EdfReader(filename)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    ECG_chan = signal_labels.index('ECG')
    HRV_chan = signal_labels.index('HRV')
    ecg = f.readSignal(ECG_chan)
    hrv = f.readSignal(HRV_chan)
    ecg_sf = f.getSampleFrequencies()[ECG_chan]

    hrv_sf = f.getSampleFrequencies()[HRV_chan]

    startdate = f.getStartdatetime()

    ecg_t = np.arange(1,len(ecg)+1)/ecg_sf

    hrv_t = np.arange(1,len(hrv)+1)/hrv_sf


    hrv_t_ms = hrv_t*1000
    hrv_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + hrv_t_ms.astype(int).astype('timedelta64[ms]')


    ecg_t_ms = ecg_t*1000
    ecg_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + ecg_t_ms.astype(int).astype('timedelta64[ms]')
    
    if n>3:
        accx_chan = signal_labels.index('Accelerometer_X')
        accy_chan = signal_labels.index('Accelerometer_Y')
        accz_chan = signal_labels.index('Accelerometer_Z')

        accX = f.readSignal(accx_chan)
        accY = f.readSignal(accy_chan)
        accZ = f.readSignal(accz_chan)
        accX_sf = f.getSampleFrequencies()[accx_chan]
        accY_sf = f.getSampleFrequencies()[accy_chan]
        accZ_sf = f.getSampleFrequencies()[accz_chan]
        accX_t = np.arange(1,len(accX)+1)/accX_sf
        accY_t = np.arange(1,len(accY)+1)/accY_sf
        accZ_t = np.arange(1,len(accZ)+1)/accZ_sf
        accX_t_ms = accX_t*1000
        accX_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accX_t_ms.astype(int).astype('timedelta64[ms]')

        accY_t_ms = accY_t*1000
        accY_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accY_t_ms.astype(int).astype('timedelta64[ms]')

        accZ_t_ms = accZ_t*1000
        accZ_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accZ_t_ms.astype(int).astype('timedelta64[ms]')

        return [(ecg, ecg_t_ms),(hrv, hrv_t_ms), (accX, accX_t_ms), (accY, accY_t_ms), (accZ, accZ_t_ms)]
    return [(ecg, ecg_t_ms), (hrv, hrv_t_ms), ([], []), ([], []), ([], [])]

def load_CBT_cloud(filename):
    
    cbt_data_all = pd.read_csv(filename, sep=';', skiprows=1)
    cbt_data_all['t'] = pd.to_datetime(cbt_data_all.date_time_local, dayfirst=True)
    cbt_data_all.set_index('t')
    
    cbt_data_all['CBT'] = cbt_data_all['core_temperature [C]']
    cbt_data_all['ST'] = cbt_data_all['skin_temperature [C]']

    cbt_data_all = cbt_data_all.set_index('t')
    cbt_data = cbt_data_all[['CBT', 'ST']]
    cbt_data = cbt_data.resample('1T').median()
    
    return cbt_data