#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:31:38 2024

@author: nct76
"""
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filter_for_circadian import filtercircadianout
from scipy.signal import find_peaks
import os
import sys
# This code describes the equations of the limit oscillator model used by 
# https://www.nature.com/articles/s41598-019-47290-6 and first described by 
# https://doi.org/10.1016/j.jtbi.2007.04.001
# Task 0: See if you can map some of the variables here to those in the equations - don't worry if you can't,
# we will go through it once you are back. 
def vanDerPol(t, y, a,b,c,d):
    dydt = [np.nan,np.nan,np.nan]
    x = y[0]
    xc = y[1]
    n = y[2]
    I = np.interp([t],a, b)[0]
    I0 = 9500
    p = 0.5
    alpha0 = 0.1
    alpha = alpha0*((I/I0)**p)*(I/(I+100))
    G = 37
    Bh = G*alpha*(1-n)
    B = Bh*(1-0.4*x)*(1-0.4*xc)
    beta = 0.007
    dydt[2] = 60*(alpha*(1-n)-beta*n)
    sigma = np.interp([t],a, c)[0]
    # if sigma>1:
    #     sigma = 1
    # else:
    #     sigma=0
    tx = d
    k=0.55
    mu=0.1300
    Lq=1/3
    rho = 0.032
    As = 10.0
    
    C = np.mod(t,24)
    C = np.arctan(xc/x)*24/(2*np.pi);
    phi_xcx = -2.98
    phi_ref = 0.97
    CBTmin = phi_xcx + phi_ref
    CBTmin = CBTmin*24/(28*np.pi)
    psi_cx = C - CBTmin
    psi_cx = np.mod(psi_cx, 24)
    if (psi_cx > 16.5) and (psi_cx < 21):
        Nsh = rho*(1/3);
    
    Nsh = rho*(1/3 - sigma)
    Ns = Nsh*(1-np.tanh(As*x))
    dydt[0] = (np.pi/12) * (xc + mu*((1/3)*x + (4/3)*x**3 - (256/105)*x**7 ) + B + Ns)
    dydt[1] = (np.pi/12) * (Lq*B*xc-x*((24/(0.99729*tx))**2 + k*B))
    return dydt

# To help you understand what is going on here, I recommend going through
# this tutorial: https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html
# This whole book is good so, you might want to go back to the Numerical Integration section
# for an introduction to what that is (it really just means solving differential equations with a computer).
# Or have a look at the Python programming sections if you need to any reminders of those. 
# This function will solve the model above, using numerical integration.  
def run_model(tau, steps,sf,init_cond, I, wake):
    return solve_ivp(vanDerPol, [0, steps*sf], init_cond, t_eval=np.arange(0, steps*sf, sf), args=(np.arange(0,steps*sf, sf), I, wake, tau), method='Radau')

#%%
# Empatica code
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
#%%
# Specify location of the Lys and CBT data
LYS_DATA = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23/S017/Lys/thorntonchristopher19gmail.com_2024-09-27_2024-10-21.csv'
CBT_DATA = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23/S017/CORE/S017_CBT_cloud.csv'
EMPATICA_DATA = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23/S017/Empatica/'
LYS_DATA = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23/S013/Lys/CN001_LysDataTwoWeeks.csv'
EMPATICA_DATA = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23/S013/Empatica'
CBT_DATA = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23/S013/CORE/CN001_CBT.csv'

LYS_DATA = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23/S013/Lys/CN001_LysDataTwoWeeks.csv'
CBT_DATA = '/Users/nct76/Library/CloudStorage/OneDrive-NewcastleUniversity/CNNP_pilot_winter23/CN02/CN002_CBT.csv'

# Specify the start and end times of the control week
# Must start and end at exactly 00:00
WEEK_ONE_START = '2024-09-28T00:00'
WEEK_ONE_END = '2024-10-05T00:00'

WEEK_ONE_START = '2024-06-07T00:00'
WEEK_ONE_END = '2024-06-14T00:00'
# Set to True if you are running this after only week one, to get initial recommended PA time.
FIRST_WEEK_ONLY = False

# Otherwise specify the start and end of scheduled physical activity week
# Must start and end at exactly 00:00
WEEK_TWO_START = '2024-10-06T00:00'
WEEK_TWO_END = '2024-10-13T00:00'

WEEK_TWO_START = '2024-06-14T00:00'
WEEK_TWO_END = '2024-06-21T00:00'


#%% Load light exposure and movement data 
# This forms the input to the model
# Visualise this light and movement data, and try to understand what is being measured (lux, Kelvin, etc).

data = pd.read_csv(LYS_DATA)
data['t'] = pd.to_datetime(data.Timestamp, yearfirst=True)
data = data[['t','Lux', 'Movement']]
data['t'] = data.t+np.timedelta64(1, 'h')
data = data.set_index('t')




subj_sleep = read_Empatica_DBM(EMPATICA_DATA, 'sleep-detection')
subj_act = read_Empatica_DBM(EMPATICA_DATA, 'accelerometers-std')

subj_sleep = subj_sleep[['t','sleep_detection_stage']]
subj_sleep.loc[subj_sleep.sleep_detection_stage.isna(), 'sleep_detection_stage'] = 0
subj_sleep = subj_sleep.set_index('t')
subj_sleep = subj_sleep.resample('5T').median()
subj_sleep = subj_sleep.reset_index()

subj_act = subj_act[['t','accelerometers_std_g']]
subj_act.loc[subj_act.accelerometers_std_g.isna(), 'accelerometers_std_g'] = 0
subj_act = subj_act.set_index('t')
subj_act = subj_act.resample('5T').median()
subj_act = subj_act.reset_index()

subj_sleep['Wake'] = 1
subj_sleep.loc[subj_sleep.sleep_detection_stage>0, 'Wake'] = 0

data = data.resample('5T').median()
data = data.reset_index()


data = pd.merge(data, subj_sleep[['t', 'Wake']])
data = pd.merge(data,subj_act[['t', 'accelerometers_std_g']] )

data.loc[data.Lux.isna(), 'Lux'] = 0.0
data.loc[data.Movement.isna(), 'Movement'] = 0.0

data['Movement'] = data.accelerometers_std_g*100


data['I'] = data.Lux
data.loc[data.I<data.Movement*100, 'I'] = data.Movement*100
data['Wake'] = data.Movement>0

data = data[(data.t<=pd.Timestamp(WEEK_ONE_END)) & (data.t>=pd.Timestamp(WEEK_ONE_START))]
data = data.set_index('t')
data = data.reset_index()


plt.plot(data.t, data.I)
plt.show()

#%%
# How does changing the intrinsic period (x0) affect the model output?
# Can you quantify this in a graph?
x0 = 24.2 # intrinsic period of the circadian rhythm - a person specific parameter.
sf = 5/60 # Sample frequency in samples per hour
# Task 3: What are the initial conditions? What should they be set to? How does changing them affect the output?
# Can you quantify this in a graph?
initial_conditions = [-1,0,0]

# Running the simulation (solving the equation)
steps = len(data)
sol = run_model(x0, steps, sf, initial_conditions, data.I, data.Wake)

# Optimise initial conditions
final_init_cond = sol.y[:, data[(data.t.dt.hour==data.t[0].hour) & (data.t.dt.minute==data.t[0].minute)].index[-1]]

sol = run_model(x0, steps, sf, final_init_cond, data.I, data.Wake)
# plotting the solution
ax = plt.subplot(2,1,1)
plt.plot(data.t, data.I, 'b', label='Input')
plt.ylabel('Lux + Acceleration')
ax = plt.subplot(2,1,2)
plt.plot(data.t, sol.y[0], 'r', label='Model')
plt.ylabel('Core Temperature (°C)')
plt.plot(data.t, np.zeros_like(sol.y[0]), "--", color="gray")
plt.legend()
plt.show()
#%% Core body temperature data
# The model predicts the circadian fluctuation in body temperature, so we can compare this to the real data.
# However, the real data contains all changes in body temperature, not just the circadian fluctuations. 
# To isolate the circadian fluctuations we must filter it. There are many ways to do this but we can use a 
# method called singular spectrum analysis (https://pyts.readthedocs.io/en/stable/auto_examples/decomposition/plot_ssa.html)
# I've provided some code to filter the circadian component using SSA - filter_for_circadian.py

# Task 4: Load the core body temperature data, filter it, and check how accurate is the model prediction?
# Can you quantify this?
# Can you improve the model prediction?

cbt_data_all = pd.read_csv(CBT_DATA, sep=';', skiprows=1)
cbt_data_all['t'] = pd.to_datetime(cbt_data_all.date_time_local, dayfirst=True)
cbt_data_all.set_index('t')
cbt_data_all = cbt_data_all[(cbt_data_all.t<=pd.Timestamp(WEEK_ONE_END)) & (cbt_data_all.t>=pd.Timestamp(WEEK_ONE_START))]

cbt_data_all['CBT'] = cbt_data_all['core_temperature [C]']
cbt_data_all['ST'] = cbt_data_all['skin_temperature [C]']

cbt_data = cbt_data_all[['t', 'CBT']]
cbt_data = cbt_data.set_index('t')
cbt_data = cbt_data.resample('5T').median()
cbt_data = cbt_data.interpolate()

skt_data = cbt_data_all[['t', 'ST']]
skt_data = skt_data.set_index('t')
skt_data = skt_data.resample('5T').median()
skt_data = skt_data.interpolate()

beginning = cbt_data.first_valid_index()
ending = cbt_data.last_valid_index()
cbt_data = cbt_data[beginning:ending]
cbt_data = cbt_data.reset_index()

beginning = skt_data.first_valid_index()
ending = skt_data.last_valid_index()
skt_data = skt_data[beginning:ending]
skt_data = skt_data.reset_index()

fs = 1/(cbt_data.t[1]-cbt_data.t[0]).seconds # sample frequency in Hz
filterinput = np.array([cbt_data.CBT, cbt_data.t])
non_circ, circ = filtercircadianout(filterinput, fs)
cbt_data['circ_cbt'] = circ

fs = 1/(skt_data.t[1]-skt_data.t[0]).seconds # sample frequency in Hz
filterinput = np.array([skt_data.ST, skt_data.t])
non_circ, circsk = filtercircadianout(filterinput, fs)
skt_data['circ_skt'] = circsk




plt.plot(cbt_data.t,cbt_data['circ_cbt'])
#plt.plot(skt_data.t,1/circsk)

plt.show()
#%% Optimising the intrinsic period
# The intrinsic 
from scipy.optimize import minimize
from scipy.signal import find_peaks
def compare_model(sol,data,cbt_data):
    x = sol.y[0]*0.3
    
    peaks, _ = find_peaks(-x, prominence=[0.1,5])
    peaks_cbt, _ = find_peaks(-cbt_data.circ_cbt, prominence=[0.01,5])
    peaks_cbt = peaks_cbt[1:]
    
    cbt_min = cbt_data.t[peaks_cbt].reset_index().t[1:]
    model_min = data.t[peaks].reset_index().t[1:]
    return np.abs(cbt_min-model_min)

def evaluate_model(x, data,sf,init_cond,cbt_data):
    
    sol = run_model2(x[0], data,sf,init_cond)
    model_error = compare_model(sol, data, cbt_data)
    return model_error.sum().total_seconds()
    
def run_model2(tau, data,sf,init_cond):
    return solve_ivp(vanDerPol, [0, len(data)*sf], init_cond, t_eval=np.arange(0, len(data)*sf, sf), args=(np.arange(0,len(data)*sf, sf),data.I,data.Wake, tau), method='Radau')

#%%
sf = ((data.t[1]-data.t[0]).seconds/60)/60# Samples per hour
initial_conditions = final_init_cond
args = (data, sf,initial_conditions,cbt_data)
x0 = [24.2]
res = minimize(evaluate_model, x0, method='nelder-mead', args=args,
               options={'xatol': 1e-8, 'disp': True})
#%%
sol_ext = run_model2(res.x[0], data, sf, initial_conditions)
tau_opt = res.x[0]
scaled = sol_ext.y[0]*0.3
week1model = sol_ext
#%%
plt.figure(figsize=(12, 6))
ax = plt.subplot(1,1,1)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Core Body Temp', fontsize=12)
plt.plot(data.t, scaled, 'r', label='Model')
plt.plot(data.t, np.zeros_like(scaled), "--", color="gray")
plt.plot(cbt_data.t, cbt_data.circ_cbt, 'g', label='CBT')
plt.title('Optimised Intrinsic Period Model Compared to CBT data', fontsize=14)
plt.legend()
plt.tight_layout()
#plt.savefig(r'C:\Users\willm\OneDrive - Newcastle University\Bioinformatics\Research project\Week 9\Results\CN001_optimised_intrin_merged.png')
plt.show()
#%%
from scipy.stats import norm
peaks, _ = find_peaks(-scaled, prominence=[0.1,5])

# Function to get user input for time ranges
def get_time_ranges():
    morning_start = input("Enter the start time for the morning range (e.g., 7 for 7am): ")
    morning_end = input("Enter the end time for the morning range (e.g., 9 for 9am): ")
    evening_start = input("Enter the start time for the evening range (e.g., 17 for 5pm): ")
    evening_end = input("Enter the end time for the evening range (e.g., 22 for 10pm): ")

    morning_range = range(int(morning_start), int(morning_end) + 1)
    evening_range = range(int(evening_start), int(evening_end) + 1)
    return morning_range, evening_range

morning_range, evening_range = get_time_ranges()

def get_gaussian_fill(n, amp):
    x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), n)
    return np.abs(norm.pdf(x) * amp + np.random.normal(amp / 10, amp / 10, n))

phase_responses = []
datas = []
models = []
ax = plt.subplot(1, 1, 1)
for hourofday in range(0, 24):
    d_cp = data.copy()
    d_cp['I'] = d_cp.Lux
    for d in np.unique(d_cp.t.dt.dayofyear):
        time_index = (d_cp.t.dt.hour == hourofday) & (d_cp.t.dt.dayofyear == d)
        number_samples = len(d_cp.loc[time_index, 'Movement'])
        d_cp.loc[time_index, 'Movement'] = get_gaussian_fill(number_samples, 100)
    
    d_cp.loc[d_cp.I < d_cp.Movement * 100, 'I'] = d_cp.Movement * 100
    d_cp['Wake'] = d_cp.Movement > 0
    datas.append(d_cp)
    sol_mod = run_model2(tau_opt, d_cp, sf, initial_conditions)
    models.append(sol_mod)

# Plot all models and the reference model on the same graph
plt.figure(figsize=(12, 6))
for i, sol_mod in enumerate(models):
    scaled_mod = sol_mod.y[0] * 0.3
    plt.plot(data.t, scaled_mod, label=f'Model with exercise at hour {i+1}')

plt.plot(data.t, scaled, 'r', label='Reference Model')

date_ticks = pd.date_range(start=data['t'].iloc[0].date(), end=data['t'].iloc[-1].date(), freq='D')
for date in date_ticks:
    plt.axvline(x=date, color='gray', linestyle='--', alpha=0.5)

plt.xticks(rotation=45, ha='right', fontsize=8)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Core Body Temp', fontsize=12)
plt.title('Model Prediction of Core Body Temperature Over Time', fontsize=14)
plt.legend()
plt.tight_layout()
#plt.savefig(rootfolder + '/CN002_actvitity_time_effect.png')
plt.show()

#%%
import scipy.signal as sig
def get_acrophases(x):
    phi_x = np.angle(sig.hilbert(x))
    
    zero_crossings_x = np.where(np.diff(np.sign(phi_x))>0)[0]

    return zero_crossings_x
# Calculate Phase Shifts
peaks_ref = get_acrophases(scaled)
final_peak_ref = (data.t[peaks_ref].dt.hour*60 + data.t[peaks_ref].dt.minute).iloc[-1]
median_peak_ref = (data.t[peaks_ref].dt.hour*60 + data.t[peaks_ref].dt.minute).median()

max_num_peaks = 0
for i, sol_mod in enumerate(models):
    scaled_mod = sol_mod.y[0] * 0.3
    peaks_mod = get_acrophases(scaled_mod)
    max_num_peaks = len(peaks_mod)-1
shifts = []
for i, sol_mod in enumerate(models):
    scaled_mod = sol_mod.y[0] * 0.3
    peaks_mod = get_acrophases(scaled_mod)
    final_peak_mod = (data.t[peaks_mod[max_num_peaks]].hour*60 + data.t[peaks_mod[max_num_peaks]].minute)
    shift = (final_peak_mod - median_peak_ref)
    shifts.append(shift)

# Function to get user input for shift preference
def get_shift_preference():
    preference = input("Do you want a positive or negative shift? (Enter 'positive' or 'negative'): ").strip().lower()
    return preference

preference = get_shift_preference()

if preference == 'positive':
    max_shift_index = None
    max_shift_value = -np.inf
    for i in morning_range:
        if shifts[i - 1] > max_shift_value:
            max_shift_value = shifts[i - 1]
            max_shift_index = i
    for i in evening_range:
        if shifts[i - 1] > max_shift_value:
            max_shift_value = shifts[i - 1]
            max_shift_index = i
elif preference == 'negative':
    max_shift_index = None
    max_shift_value = np.inf
    for i in morning_range:
        if shifts[i - 1] < max_shift_value:
            max_shift_value = shifts[i - 1]
            max_shift_index = i
    for i in evening_range:
        if shifts[i - 1] < max_shift_value:
            max_shift_value = shifts[i - 1]
            max_shift_index = i

# Plot phase shifts
plt.figure(figsize=(14, 7))

optimal_model = models[max_shift_index]
optimal_exercise = datas[max_shift_index]
# Highlight the specified time ranges
for i in morning_range:
    plt.axvspan(i, i + 1, color='blue', alpha=0.1)
for i in evening_range:
    plt.axvspan(i, i + 1, color='blue', alpha=0.1)

# Highlight the optimal time in bright red
if max_shift_index is not None:
    plt.axvspan(max_shift_index, max_shift_index + 1, color='red', alpha=0.3)

# Plot points and connect with lines
plt.plot(range(1, 25), shifts, marker='o', linestyle='-', color='black')

# Annotate points
for i, shift in enumerate(shifts):
    plt.annotate(f'{shift:.2f}', (i + 1, shift), textcoords="offset points", xytext=(0, 10), fontsize=10, ha='center')

plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Hour of Exercise')
plt.ylabel('Phase Shift (minutes)')
plt.title('Phase Shift Relative to Reference Model')
plt.grid(True)
plt.tight_layout()
#plt.savefig(rootfolder + '/CN002_actvitity_time_effect_quant_range.png')
plt.show()
#%%
# Find peaks in the recorded data (CBT)
peaks_cbt, _ = find_peaks(cbt_data.circ_cbt, prominence=[0.01, 5])

# Find peaks in the model data
peaks_model, _ = find_peaks(scaled, prominence=[0.1, 5])

# Create the plot
plt.figure(figsize=(12, 6))
ax = plt.subplot(1,1,1)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Core Body Temp', fontsize=12)
plt.plot(data.t, scaled, 'r', label='Model')
plt.plot(data.t, np.zeros_like(scaled), "--", color="gray")
plt.plot(cbt_data.t, cbt_data.circ_cbt, 'g', label='CBT')

# Add vertical lines at peaks in the recorded data (CBT)
for peak in peaks_cbt:
    plt.vlines(x=cbt_data.t[peak], ymin=cbt_data.circ_cbt[peak] - 0.05, ymax=cbt_data.circ_cbt[peak] + 0.05, color='blue', linestyle='--', alpha=0.6)

# Add vertical lines at peaks in the model data
for peak in peaks_model:
    plt.vlines(x=data.t[peak], ymin=scaled[peak] - 0.05, ymax=scaled[peak] + 0.05, color='orange', linestyle='--', alpha=0.6)

plt.title('Optimised Intrinsic Period Model Compared to CBT data', fontsize=14)
plt.legend()
plt.tight_layout()
#plt.savefig(rootfolder + '/CN002_opt_peak_plot.png')
plt.show()

#%%
# Find peaks in the recorded data (CBT)
peaks_cbt, _ = find_peaks(cbt_data.circ_cbt, prominence=[0.01, 5])

# Find peaks in the model data
peaks_model, _ = find_peaks(sol.y[0], prominence=[0.1, 5])

# Create the plot
plt.figure(figsize=(12, 6))
ax = plt.subplot(1,1,1)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Core Body Temp', fontsize=12)
plt.plot(data.t, sol.y[0]*0.3, 'r', label='Model')
plt.plot(data.t, np.zeros_like(scaled), "--", color="gray")
plt.plot(cbt_data.t, cbt_data.circ_cbt, 'g', label='CBT')

# Add vertical lines at peaks in the recorded data (CBT)
for peak in peaks_cbt:
    plt.vlines(x=cbt_data.t[peak], ymin=cbt_data.circ_cbt[peak] - 0.05, ymax=cbt_data.circ_cbt[peak] + 0.05, color='blue', linestyle='--', alpha=0.6)

# Add vertical lines at peaks in the model data
for peak in peaks_model:
    plt.vlines(x=data.t[peak], ymin=sol.y[0][peak] - 0.05, ymax=sol.y[0][peak] + 0.05, color='orange', linestyle='--', alpha=0.6)

plt.title('Unoptimised Intrinsic Period Model Compared to CBT data', fontsize=14)
plt.legend()
plt.tight_layout()
#plt.savefig(rootfolder + '/CN002_unopt_peak_plot.png')
plt.show()

#%%
def calculate_phase_shifts(cbt_peaks, model_peaks, cbt_times, model_times):
    """ Calculate the phase shift by finding the closest daily CBT peak for each model peak. """
    cbt_peaks_df = pd.DataFrame({'time': cbt_times, 'peak': cbt_peaks})
    model_peaks_df = pd.DataFrame({'time': model_times, 'peak': model_peaks})

    # Group peaks by date
    cbt_peaks_df['date'] = cbt_peaks_df['time'].dt.date
    model_peaks_df['date'] = model_peaks_df['time'].dt.date

    # Only keep dates where both datasets have peaks
    common_dates = cbt_peaks_df['date'].unique()
    model_peaks_df = model_peaks_df[model_peaks_df['date'].isin(common_dates)]
    
    total_phase_shift = 0
    
    # For each date, find the closest CBT peak for each model peak
    for date in common_dates:
        daily_cbt_peaks = cbt_peaks_df[cbt_peaks_df['date'] == date]['time']
        daily_model_peaks = model_peaks_df[model_peaks_df['date'] == date]['time']
        
        for model_peak in daily_model_peaks:
            # Find the closest CBT peak
            closest_cbt_peak = daily_cbt_peaks.iloc[(daily_cbt_peaks - model_peak).abs().argmin()]
            # Calculate phase shift in hours
            phase_shift = (closest_cbt_peak - model_peak).total_seconds() / 3600
            total_phase_shift += abs(phase_shift)
    
    return total_phase_shift

# Find peaks
cbt_peaks, _ = find_peaks(cbt_data['circ_cbt'], prominence=[0.01, 5])
unopt_peaks, _ = find_peaks(sol.y[0], prominence=[0.1, 5])
opt_peaks, _ = find_peaks(scaled, prominence=[0.1, 5])

# Get peak times
cbt_peak_times = cbt_data['t'].iloc[cbt_peaks]
unopt_peak_times = data['t'].iloc[unopt_peaks]
opt_peak_times = data['t'].iloc[opt_peaks]

# Calculate phase shifts
unopt_shift = calculate_phase_shifts(cbt_peaks, unopt_peaks, cbt_peak_times, unopt_peak_times)
opt_shift = calculate_phase_shifts(cbt_peaks, opt_peaks, cbt_peak_times, opt_peak_times)

# Bar plot of phase shifts
fig, ax = plt.subplots(figsize=(8, 5))
opt_not = ['Unoptimised Model', 'Optimised Model']
shifts = [unopt_shift, opt_shift]

# Check if shifts contain zero or very small values which may not display well
print("Unoptimized Model Error:", unopt_shift)
print("Optimized Model Error:", opt_shift)

bars = ax.bar(opt_not, shifts, color=['blue', 'green'])

ax.set_title('Total Error for Unoptimised vs. Optimised Models')
ax.set_ylabel('Total Error (Hours)')
ax.set_xlabel('Model')
ax.set_ylim(0, max(shifts) + 10)  # Ensure y-axis is properly scaled to show smaller bars

# Adding legend directly with bars created for clarity
ax.legend(bars, ['Unoptimised Model', 'Optimised Model'], loc='upper right')

ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
#plt.savefig(rootfolder + '/CN002_opt_vs_unpot_bar.png')
plt.show()

if FIRST_WEEK_ONLY:
    sys.exit()
#%% Exercise Week
cbt_data_2nd = pd.read_csv(CBT_DATA, sep=';', skiprows=1)
cbt_data_2nd['t'] = pd.to_datetime(cbt_data_2nd.date_time_local, dayfirst=True)
cbt_data_2nd.set_index('t')
cbt_data_2nd = cbt_data_2nd[(cbt_data_2nd.t<=pd.Timestamp(WEEK_TWO_END)) & (cbt_data_2nd.t>=pd.Timestamp(WEEK_TWO_START))]

cbt_data_2nd['CBT'] = cbt_data_2nd['core_temperature [C]']

cbt_data_2nd = cbt_data_2nd[['t', 'CBT']]
cbt_data_2nd = cbt_data_2nd.set_index('t')
cbt_data_2nd = cbt_data_2nd.resample('5T').median()
cbt_data_2nd = cbt_data_2nd.interpolate()

beginning = cbt_data_2nd.first_valid_index()
ending = cbt_data_2nd.last_valid_index()
cbt_data_2nd = cbt_data_2nd[beginning:ending]
cbt_data_2nd = cbt_data_2nd.reset_index()

fs = 1/(cbt_data_2nd.t[1]-cbt_data_2nd.t[0]).seconds # sample frequency in Hz
filterinput = np.array([cbt_data_2nd.CBT, cbt_data_2nd.t])
non_circ, circ = filtercircadianout(filterinput, fs)
cbt_data_2nd['circ_cbt'] = circ
#%%


y = cbt_data_2nd['circ_cbt']
x = cbt_data['circ_cbt'][:-1]
t = cbt_data_2nd.t
def plot_two_signals_with_acro(t, x, y, x_label, y_label):
    
    phi_x = np.angle(sig.hilbert(x))
    phi_y = np.angle(sig.hilbert(y))
    
    zero_crossings_x = np.where(np.diff(np.sign(phi_x))>0)[0]
    zero_crossings_y = np.where(np.diff(np.sign(phi_y))>0)[0]
    
    
    acrophases_x = t[zero_crossings_x]
    amplitudes_x = x[zero_crossings_x]
    acrophases_y = t[zero_crossings_y]
    amplitudes_y = y[zero_crossings_y]
    fig, ax = plt.subplots()
    ax.plot(t, x, label=x_label, color='k')
    ax.plot(t, y, label=y_label, color='r')
    ax.plot(acrophases_x,amplitudes_x , '|', markersize=30, color='k')
    ax.plot(acrophases_y,amplitudes_y , '|', markersize=30, color='r')

    for acro_x, acro_y in zip(acrophases_x, acrophases_y):
        print(str(acro_x) + ' - ' + str(acro_y))
        phase_shift = str((np.datetime64(acro_x)-np.datetime64(acro_y)).astype('timedelta64[m]'))
        print(phase_shift)
        ax.annotate(phase_shift, xy=(acro_x, np.max(amplitudes_x)))
    ax.legend()
    plt.ylabel('Core Temperature (°C)')
    plt.show()
plot_two_signals_with_acro(t, x, y, 'Control', 'PA')
optimal_mod_CBT = optimal_model.y[0]*0.2
#plt.plot(cbt_data_2nd.t[2:],optimal_mod_CBT, label='Model')
#%%
plot_two_signals_with_acro(cbt_data.t, cbt_data.circ_cbt, optimal_mod_CBT, 'Baseline CBT Rhythm', 'Estimated CBT Rhythm with PA')
#%%

data2nd = pd.read_csv(LYS_DATA)
data2nd['t'] = pd.to_datetime(data2nd.Timestamp, yearfirst=True)
data2nd = data2nd[['t','Lux', 'Movement']]
data2nd['t'] = data2nd.t+np.timedelta64(1, 'h')
data2nd = data2nd.set_index('t')


subj_sleep = read_Empatica_DBM(EMPATICA_DATA, 'sleep-detection')
subj_act = read_Empatica_DBM(EMPATICA_DATA, 'accelerometers-std')

subj_sleep = subj_sleep[['t','sleep_detection_stage']]
subj_sleep.loc[subj_sleep.sleep_detection_stage.isna(), 'sleep_detection_stage'] = 0
subj_sleep = subj_sleep.set_index('t')
subj_sleep = subj_sleep.resample('5T').median()
subj_sleep = subj_sleep.reset_index()

subj_act = subj_act[['t','accelerometers_std_g']]
subj_act.loc[subj_act.accelerometers_std_g.isna(), 'accelerometers_std_g'] = 0
subj_act = subj_act.set_index('t')
subj_act = subj_act.resample('5T').median()
subj_act = subj_act.reset_index()

subj_sleep['Wake'] = 1
subj_sleep.loc[subj_sleep.sleep_detection_stage>0, 'Wake'] = 0

data2nd = data2nd.resample('5T').median()
data2nd = data2nd.reset_index()

data2nd = pd.merge(data2nd, subj_sleep[['t', 'Wake']])
data2nd = pd.merge(data2nd,subj_act[['t', 'accelerometers_std_g']] )
data2nd['Movement'] = data2nd.accelerometers_std_g*100


data2nd['I'] = data2nd.Lux
data2nd.loc[data2nd.I<data2nd.Movement*100, 'I'] = data2nd.Movement*100
data2nd['Wake'] = data2nd.Movement>0

data2nd = data2nd[(data2nd.t<=pd.Timestamp(WEEK_TWO_END)) & (data2nd.t>=pd.Timestamp(WEEK_TWO_START))]
data2nd = data2nd.set_index('t')
data2nd = data2nd.reset_index()

plt.plot(optimal_exercise.t+np.timedelta64(7,'D') , optimal_exercise.I, label='Estimated')
plt.plot(data2nd.t, data2nd.I, label='Actual')
plt.ylabel('Lux + Acceleration')
plt.legend()
plt.show()
#%%
x0 = 24.2 # intrinsic period of the circadian rhythm - a person specific parameter.
sf = 5/60 # Sample frequency in samples per hour

initial_conditions = [-1,0,0]

# Running the simulation (solving the equation)
steps = len(data2nd)
sol2 = run_model(x0, steps, sf, initial_conditions, data2nd.I, data2nd.Wake)

# Optimise initial conditions
final_init_cond = sol.y[:, data2nd[(data2nd.t.dt.hour==data2nd.t[0].hour) & (data2nd.t.dt.minute==data2nd.t[0].minute)].index[-1]]

sol2 = run_model(x0, steps, sf, final_init_cond, data2nd.I, data2nd.Wake)
# plotting the solution
plot_two_signals_with_acro(data2nd.t, optimal_model.y[0][:-1], sol2.y[0], 'Using estimated PA + Light', 'Using actual PA + Light')



