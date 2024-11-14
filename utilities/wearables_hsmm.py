#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:20:48 2024

@author: nct76
"""

import pyhsmm
import pyhsmm.basic.distributions as distributions
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from scipy.stats import zscore
import pandas as pd
from pyts import decomposition # SSA
import copy 
def applyHSMM2D(data):
    obs_dim = 2
    Nmax = 10

    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.eye(obs_dim),
                    'kappa_0':0.3,
                    'nu_0':obs_dim+5}
    dur_hypparams = {'alpha_0':2*30,
                     'beta_0':2}

    obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
            alpha=6.,gamma=6., # better to sample over these; see concentration-resampling.py
            init_state_concentration=6., # pretty inconsequential
            obs_distns=obs_distns,
            dur_distns=dur_distns)

    posteriormodel.add_data(data,trunc=10)

    models = []
    for idx in range(0,1500):
        posteriormodel.resample_model()


    return posteriormodel
def BIC(model):
    return -2*sum(model.log_likelihood(s.data).sum() for s in model.states_list) + \
                num_parameters(model) * np.log(
                        sum(s.data.shape[0] for s in model.states_list))
def BIC2(model):
    return -2*model.log_likelihood() + \
                num_parameters(model) * np.log(
                        sum(s.data.shape[0] for s in model.states_list))
def num_parameters(model):
    return sum(o.num_parameters for o in model.obs_distns) \
            + len(model.dur_distns) \
            + model.num_states**2 - model.num_states

def applyHSMM(data, D):
    np.random.seed(seed=0)
    obs_dim = D
    Nmax = 30

    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.eye(obs_dim),
                    'kappa_0':0.3,
                    'nu_0':obs_dim+5}
    dur_hypparams = {'alpha_0':2*30,
                     'beta_0':2}

    obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
            #alpha_a_0=1./4,alpha_b_0=0.5,
            #gamma_a_0=1./4,gamma_b_0=0.5,#1.,gamma_b_0=1./4, # better to sample over these; see concentration-resampling.py
            alpha=0.25,gamma=0.25,
            init_state_concentration=6., # pretty inconsequential
            obs_distns=obs_distns,
            dur_distns=dur_distns)
    for d in data:
        posteriormodel.add_data(d,trunc=10)
    
    models = []
    bic=0
    for idx in range(0,100):
        posteriormodel.resample_model()
        prevbic = bic
        bic = BIC2(posteriormodel)
        bic_change = round(100*(prevbic-bic)/prevbic,1)
        print(str(idx) + ' ' + str(len(posteriormodel.used_states)) + ' states used. BIC=' + str(bic_change))
        if (idx+1) % 10 == 0:
            models.append(copy.deepcopy(posteriormodel))

    return posteriormodel, models

def visualise_training(models,animations_loc):
    
    iterations = []
    likeli = []
    fig = plt.figure()
    for idx, model in enumerate(models):
        iterations.append(idx)
        likeli.append(model.log_likelihood())
        plt.clf()
        model.plot()
        plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % (10*(idx+1)))
        plt.savefig(animations_loc + '/model/' + 'iter_%.3d.png' % (10*(idx+1)))
    fig = plt.figure()
    plt.plot(iterations, likeli)
    plt.xlabel('iterations')
    plt.ylabel('log likelihood')
    plt.savefig(animations_loc + '/ll/' + 'loglikelihood.png')
        
def visualise_HSMM(posteriormodel, obs_names, time_asleep=None, colors=None):
    state_means = []
    state_mean_durations = []
    state_numbers = []
    states_used = np.squeeze(np.where(posteriormodel.state_usages>0.01))
    #states_used = [1,13,4,6,15,21,11,16]

    if colors==None:
        colors = ['k']*len(states_used)
    print(states_used)
    n = len(states_used)
    while n%3>0: n= n+1
    ind = 0
    nwide = int(n/3)
    nhigh = int(n/int(n/3))
    fig, axes = plt.subplots(nrows=nwide, ncols=nhigh, sharex=True, sharey=True)
    axes_r = axes.ravel()
    usage = posteriormodel.state_usages[posteriormodel.state_usages>0.01]
    statevals = np.array([ob.mu for ob in posteriormodel.obs_distns])

    statesmax = np.max(statevals[states_used])
    statesmin = np.min(statevals[states_used])

    for ii, i in enumerate(states_used):
        state_means.append(posteriormodel.obs_distns[i].mu)
        state_mean_durations.append(posteriormodel.dur_distns[i].mean)
        
        xvals = np.arange(0,len(posteriormodel.obs_distns[i].mu))
        lab = str(round(usage[ii]*100,1)) + '% signal'
        if time_asleep:
            lab = lab + '\n' + str(round(time_asleep[i-1]*100,1)) + '% asleep'
            
        axes_r[ind].bar(xvals,posteriormodel.obs_distns[i].mu, label=lab, color=colors[ii])
        axes_r[ind].set_ylim([statesmin, statesmax])
        axes_r[ind].set_xticks(xvals)
        axes_r[ind].set_xticklabels(obs_names)
        #axes_r[ind].legend()
        axes_r[ind].set_ylabel('state ' + str(i))
        ind = ind+1
    state_means = np.array(state_means)
    state_mean_durations = np.array(state_mean_durations)
    
    plt.show()
    return state_means, state_mean_durations, usage

def filtercircadianout(X,fs):
    comp = getCircFromSSA(X, fs)
    circadian_range = [18/24,31/24]
    periods = []
    non_circadian_signals = []
    circadian_signals = []
    for component in comp[0]:
        yf = np.abs(fft.rfft(component))
        xf = 1/(fft.rfftfreq(len(component))*((1/fs)))
        #plt.plot(xf,yf)
        peak_idx = np.where(yf == np.max(yf))[0][0]
        peak_period = xf[peak_idx]
        if peak_period < circadian_range[0]:
            non_circadian_signals.append(component)
        if peak_period >= circadian_range[0] and peak_period <=circadian_range[1]:
            circadian_signals.append(component)

    non_circadian_signals = np.sum(np.array(non_circadian_signals),axis=0)
    circadian_signals = np.sum(np.array(circadian_signals),axis=0)

    return non_circadian_signals, circadian_signals

def getUltradianSignal(X,fs):
    comp = getCircFromSSA(X, fs)
    ultradian_range = [0.5/24,18/24]
    periods = []
    non_ultradian_signals = []
    ultradian_signals = []
    for component in comp[0]:
        yf = np.abs(fft.rfft(component))
        xf = 1/(fft.rfftfreq(len(component))*((1/fs)))
        #plt.plot(xf*24,yf)
        #plt.plot([18,18],[0,max(yf)], 'k--')
        #plt.plot([0.5,0.5],[0,max(yf)], 'k--')

        #plt.xlabel('Cycle Period (Hours)')
        peak_idx = np.where(yf == np.max(yf))[0][0]
        peak_period = xf[peak_idx]
        if peak_period < ultradian_range[0]:
            non_ultradian_signals.append(component)
        if peak_period >= ultradian_range[0] and peak_period <=ultradian_range[1]:
            ultradian_signals.append(component)
    #plt.show()
    non_ultradian_signals = np.sum(np.array(non_ultradian_signals),axis=0)
    ultradian_signals = np.sum(np.array(ultradian_signals),axis=0)

    return ultradian_signals

def getCircFromSSA(y, fs):
    groups=None
    window_size = int((24*60)/5)
    components = decomposition.SingularSpectrumAnalysis(window_size=window_size, groups=groups, chunksize=1).fit_transform(y)
    return components

def remove_big_gaps(df):
    missing = df.isna().any(axis=1)
    #missing_chunks = np.split(missing, np.where(missing==False)[0])
    missing_t = np.split(df.t, np.where(missing==False)[0])
    #chunk_lengths = np.array([len(m) for m in missing_chunks])
    #missing_t = pd.Series([m.iloc[0] for m in missing_t if len(m)>0])
    missing_days = pd.Series([m for m in missing_t if len(m)>1440])
    shifted = np.zeros(np.shape(df.t))
    for md in missing_days:
        print(md)
        num_days = (missing_days[0].iloc[-1] - missing_days[0].iloc[0]).days
        
        days_shift = np.timedelta64(num_days,'D')
        shifted[(df.t>=md.iloc[0]) & (df.t<=days_shift + md.iloc[0])] = 1
    df['big_gap'] = shifted
    return df
    
    
    
    
def getNonCircadianComp(df_minute_all, obs):
    df = df_minute_all.copy()
    start = df_minute_all.t[0]
    sf = df_minute_all.t[1] -df_minute_all.t[0]
    time_art = np.arange(start, start + len(df_minute_all.t)*sf, sf)
    df['t'] = time_art
    df = df[['t','sleep_detection_stage'] + obs].set_index('t').resample('5min').median()
    df_intp = df.interpolate()

   # df_withna = df_minute_all[['t', 'sleep_detection_stage'] + obs].set_index('t').resample('5min').median()
    df_intp = df_intp.ffill().bfill()
    
    t = df_intp.index - df_intp.index[0]
    fs = 1/t[1].total_seconds()
    #t = (t.total_seconds()).astype(int)
    t = np.arange(0,len(df_intp.index)*t[1].total_seconds(),t[1].total_seconds())
    for ob in obs:
        #print(ob)
        X = np.array([df_intp[ob],t])
        non_circ = getUltradianSignal(X,fs)
        _, circ = filtercircadianout(X, fs)
        df[ob + '_non_circ'] = non_circ
        df[ob + '_circ'] = circ
    #df.loc[df_withna.pulse_rate_bpm.isna(), 'PR_non_circ'] = np.nan
    
    return df
def plotStates(df_five_minute, model, obs, colors=None):
    df_all = pd.concat(df_five_minute)
    #sleepstates = df_s.groupby('state').count()['HR_non_circ'] - df_s[df_s.sleep_detection_stage>0].groupby('state').count()['HR_non_circ']
    time_asleep = list(df_all.groupby('state').sleep.mean())
    state_mu, state_delta, usage = visualise_HSMM(model,obs, time_asleep=None,colors=colors )
    model.plot_observations()
    plt.show()
    
def getStates(df_five_minute, obs):
    data = []
    for i, df_s in enumerate(df_five_minute):
        df_five_minute[i][obs] = df_s[obs].apply(zscore)
        df_five_minute[i].loc[df_s.sleep_detection_stage==101, 'sleep'] = 1
        df_five_minute[i].loc[df_s.sleep_detection_stage==0, 'sleep'] = 0
        data.append(np.array(df_five_minute[i][obs]))
    number_obs = len(obs)
    model, models = applyHSMM(data, number_obs)
    
    for i, df_s in enumerate(df_five_minute):
        df_five_minute[i]['state'] = model.stateseqs[i]
    
    
    return df_five_minute, model, models

