#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:50:15 2023

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

def plot_all_features(df,feats):
    for i in range(len(feats)):
        plt.subplot(len(feats)+1, 1,i+1)
        plt.plot(df.t,df[feats[i]])
        plt.ylabel(feats[i])
    plt.xlabel('t')
def compare_subjects():
    
    pr002['t0'] = pr002.t
    pr003['t0'] = pr003.t-(min(pr003.t)-min(pr002.t))
    pr004['t0'] = pr004.t-(min(pr004.t)-min(pr002.t))
    plt.figure()
    plt.title('Circadian HR')
    plt.plot(pr002.t0, pr002['circadian'], 'k')
    plt.plot(pr003.t0, pr003['circadian'], 'b')
    plt.plot(pr004.t0, pr004['circadian'], 'g')
    ax1 = plt.gca()
    xmin, xmax = ax1.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in pr002.t]
    ax1.fill_between(pr002.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.1)
    ax1.set_xlim(xmin, xmax) # set limits back to default values
    ax1.set_ylabel('HR (BPM)')
    plt.show()
def plot_ethica(ethica, signal, measure, circ_measure):
    plt.title('Circadian Component')
    plt.plot(signal.t, signal[circ_measure], 'k')

    ethica = ethica.dropna(subset='Activation')
    ethica['t'] = ethica['t'].dt.round('15min')
    ax1 = plt.gca()
    xmin, xmax = ax1.get_xlim()
    
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in signal.t]
    ax1.fill_between(signal.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.1)
    ax1.set_xlim(xmin, xmax) # set limits back to default values
    plt.scatter(signal[signal.t.isin(ethica.t)].t, signal.loc[signal.t.isin(ethica.t),circ_measure], 75, marker='D', c=ethica.loc[ethica.t.isin(signal.t),measure],cmap='viridis')

def plot_ethica_polar(ethica, signal, measure, phase):
    ethica = ethica.dropna(subset=measure)
    ethica['t'] = ethica['t'].dt.round('15min')
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.set_ylim([-4,2])
    print(ethica.loc[ethica.t.isin(signal.t), measure])
    c = ax.scatter(signal.loc[signal.t.isin(ethica.t), phase], ethica.loc[ethica.t.isin(signal.t), measure], c=ethica.loc[ethica.t.isin(signal.t), measure], s=70)
    #p = ax.plot(signal.instantaneous_phase,signal.circadian)
    ax.set_theta_zero_location("N")
    
def plotDBM(df_meas, meas, meas2, ylab=''):
    ax1 = plt.subplot(2, 1,1)
    plt.title(meas)
    plt.plot(df_meas.t, df_meas[meas], 'k')
    xmin, xmax = ax1.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_meas.t]
    ax1.fill_between(df_meas.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.1)
    ax1.set_xlim(xmin, xmax) # set limits back to default values
    ax1.set_ylabel(ylab)
    
    ax1 = plt.subplot(2,1,2)
    plt.title('Circadian Component')
    plt.plot(df_meas.t, df_meas[meas2], 'k')
    xmin, xmax = ax1.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_meas.t]
    ax1.fill_between(df_meas.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.1)
    ax1.set_xlim(xmin, xmax) # set limits back to default values
    plt.show()
    
def plotCGM(df_cgm):
    ax1 = plt.subplot(2, 1,1)
    plt.title('Blood Glucose (mmol/L)')
    plt.plot(df_cgm.t, df_cgm.glucose, 'k')
    xmin, xmax = ax1.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_cgm.t]
    ax1.fill_between(df_cgm.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.1)
    ax1.set_xlim(xmin, xmax) # set limits back to default values
    ax1.set_ylabel('(mmol/L)')
    
    ax1 = plt.subplot(2,1,2)
    plt.title('Circadian Component')
    plt.plot(df_cgm.t, df_cgm.circadian, 'k')
    xmin, xmax = ax1.get_xlim()
    days = np.arange(0, np.ceil(xmax)+2)
    night = [(dt.time().hour<7)|(dt.time().hour>=22) for dt in df_cgm.t]
    ax1.fill_between(df_cgm.t, *ax1.get_ylim(), where=night, facecolor='k', alpha=.1)
    ax1.set_xlim(xmin, xmax) # set limits back to default values
    return ax1
