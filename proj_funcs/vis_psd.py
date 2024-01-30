# external modules
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
import pandas as pd
from scipy.signal import find_peaks, peak_widths
import seaborn as sns
import scipy.stats as stats


def format_global(value):

    if (value >= 0.8):
        return "{}d".format ( round ( value, 1 ) )
    elif (value < 0.8) and (value >= 1 / 24):
        return "{}h".format ( round ( value * 24, 1 ) )
    elif (value < 1 / 24):
        return "{}m".format ( round ( value * 24 * 60, 1 ) )


def format_func(value, tick_number):

    if (value >= 0.8):
        return "{}d".format ( round ( value, 1 ) )
    elif (value < 0.8) and (value >= 1 / 24):
        return "{}h".format ( round ( value * 24, 1 ) )
    elif (value < 1 / 24):
        return "{}m".format ( round ( value * 24 * 60, 1 ) )



def vis_imfs_char_pdf(imfs, instF, instA, instPwrap, t_days, path_out, fig_name, outcome_clrs=None):

    [modes, channels, time] = imfs.shape

    if outcome_clrs is None :
        # Choose your desired colormap
        cmap = cm.get_cmap ( 'viridis' )
        # Segmenting the whole range (from 0 to 1) of the color map into multiple segments
        outcome_clrs = cmap ( np.linspace ( 0, 1, modes ) )

    with PdfPages ( os.path.join ( path_out, fig_name ) )as pdf:

        for m in range ( modes ):
            for c in range ( channels ):

                fig, (ax1, ax2, ax3, ax4) = plt.subplots ( 4 , figsize=(8, 5) )
                fig.suptitle ( 'Raw signal \n IMF{} - DIM{}'.format ( m + 1, c + 1 ) )
                fig.subplots_adjust ( left=0.125, bottom=0.1, right=0.95, top=0.9, wspace=0.03, hspace=0.4 )
                ax1.plot ( t_days, imfs[m, c, :] , c = outcome_clrs[m], linewidth=0.5)
                ax1.set ( ylabel="raw IMF{} DIM{}".format ( m, c ) )
                ax2.plot ( t_days[:-1], instF[m, c, :] , c = outcome_clrs[m], linewidth=0.5)
                ax2.set ( ylabel="inst F" )
                ax3.plot ( t_days[:-1], instA[m, c, :] , c = outcome_clrs[m], linewidth=0.5)
                ax3.set ( ylabel="inst A" )
                ax4.plot ( t_days, instPwrap[m, c, :] , c = outcome_clrs[m], linewidth=0.5)
                ax4.set ( ylabel="inst Pwrap" )
                ax4.set(xlabel = "time (days)")
                pdf.savefig ()  # saves the current figure into a pdf page
                plt.close ( "all" )

    print("Plots were generated successfully!")

def smooth (s, win):
    return pd.Series ( s ).rolling ( window=win, center=True ).mean ().ffill ().bfill ()


def plot_psd_acrossP(avg_norm_psd_all, frequency_est,  label_name, smooth_win = 20, ax=None, clr = None, is_freq_vis = False):

    if clr is None:
        cmap = cm.get_cmap ( 'Dark2', 4)
        clr = np.array(cmap ( 0 )[:3])

    if is_freq_vis == True:
        x = frequency_est
    else:
        cycle_est = 1 / frequency_est
        x = cycle_est

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 5) )

    # double smoothing
    ddiff = smooth ( avg_norm_psd_all, smooth_win ).to_numpy ()
    # find peaks
    peak_indx, _ = find_peaks ( ddiff, height=0.0001 )
    widths, h_eval, left_ips, right_ips = peak_widths ( ddiff, peak_indx, rel_height=0.5 )

    ax.plot ( x, avg_norm_psd_all, linewidth=1, color=clr, label=label_name )
    # ax.plot ( x[peak_indx], avg_norm_psd_all_var[var, peak_indx], marker = "x" )
    ax.legend ()
    ax.set_xscale ( "log" )
    if is_freq_vis == True:
        ax.set ( xlabel="frequency (cycles/day)" )
    else:
        ax.set ( xlabel="cycle period (days/cycle)" )

    ax.set(ylabel = "amplitude")

    return ax

def plot_psd_acrossP_vars(avg_norm_psd_all_var, var_names, frequency_est, smooth_win = 20, ax=None, outcome_clrs = None, is_freq_vis = False):

    [n_vars, _] = avg_norm_psd_all_var.shape

    if outcome_clrs is None:
        cmap = cm.get_cmap ( 'Dark2',  n_vars)
        outcome_clrs = np.zeros ( (n_vars, 3) )
        for ii in range(n_vars):
            outcome_clrs[ii, :] = cmap ( ii )[:3]

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 5) )

    for var in range(n_vars):
        plot_psd_acrossP ( avg_norm_psd_all_var[var,:], frequency_est, label_name = var_names[var],
                           smooth_win=smooth_win, ax=ax, clr=outcome_clrs[var],
                           is_freq_vis=is_freq_vis )

    return ax


def plotShade_psd_acrossP_vars (avg_norm_psd_all_var, var_names, fluct_name, frequency_est, shade_freqs ,smooth_win=20, ax=None, outcome_clrs=None,
                           is_freq_vis=False):
    [n_vars, _] = avg_norm_psd_all_var.shape

    if outcome_clrs is None:
        cmap = cm.get_cmap ( 'Dark2', n_vars )
        outcome_clrs = np.zeros ( (n_vars, 3) )
        for ii in range ( n_vars ):
            outcome_clrs[ii, :] = cmap ( ii )[:3]

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 5) )

    for var in range ( n_vars ):
        plot_psd_acrossP ( avg_norm_psd_all_var[var, :], frequency_est, label_name=var_names[var],
                           smooth_win=smooth_win, ax=ax, clr=outcome_clrs[var],
                           is_freq_vis=is_freq_vis )

    # shaded area on the frequency ranges
    # shade_colour = (189 / 255, 189 / 255, 189 / 255)
    cmap = cm.get_cmap ( 'Set2', len(fluct_name) )
    shade_colours = np.zeros ( (len(fluct_name) , 3) )
    for ii in range ( len(fluct_name)  ):
        shade_colours[ii, :] = cmap ( ii )[:3]

    jj = 0
    for x_pair in shade_freqs:
        if is_freq_vis == True:
            ax.axvspan(x_pair[0], x_pair[1], color = shade_colours[jj], alpha = 0.5)
            x_pos = np.mean ( [x_pair[1], x_pair[0]] )
        else:
            ax.axvspan(1/x_pair[1], 1/x_pair[0], color = shade_colours[jj], alpha = 0.5)
            x_pos = np.mean ( [1 / x_pair[1], 1 / x_pair[0]] )

        if jj == 0:
            y_pos = np.max ( avg_norm_psd_all_var ) - 3 * np.std ( avg_norm_psd_all_var )
            x_text = x_pos - 0.2
        elif jj == 1:
            y_pos = np.max ( avg_norm_psd_all_var ) - 2 * np.std ( avg_norm_psd_all_var )
            x_text = x_pos - 0.05
        elif jj == 2:
            y_pos = np.max ( avg_norm_psd_all_var ) - 2 * np.std ( avg_norm_psd_all_var )
            x_text = x_pos + 0.2

        y_text = y_pos + np.std(avg_norm_psd_all_var)
        ax.annotate (fluct_name[jj], xy=(x_pos, y_pos), xytext=(x_text, y_text),
                  arrowprops={"arrowstyle": "->", "color": "gray"} )
        jj = jj + 1
    return ax

def plot_psd_eachP (data_display, frequency_est, clr_avg = None, is_freq_vis = False, outcome_clrs = None, ax = None):

    [n_patients, _] = data_display.shape

    if clr_avg is None:
        clr_avg = "k"
    if outcome_clrs is None:
        outcome_clrs = "gray"
        # pat_colors = ["#a9a9a9", "#2f4f4f", "#556b2f", "#a0522d", "#191970", "#006400", "#8b0000",
        #           "#808000", "#3cb371", "#bdb76b", "#008b8b", "#4682b4", "#00008b", "#32cd32", "#daa520",
        #           "#800080", "#b03060", "#ff4500", "#ff8c00", "#ffff00", "#00ff00", "#00fa9a", "#dc143c",
        #           "#00ffff", "#00bfff", "#f4a460", "#0000ff", "#a020f0", "#adff2f", "#da70d6", "#ff00ff",
        #           "#1e90ff", "#fa8072", "#dda0dd", "#ff1493", "#7b68ee", "#afeeee", "#ffdab9", "#ffb6c1"]

    if is_freq_vis == True:
        x = frequency_est
    else:
        cycle_est = 1 / frequency_est
        x = cycle_est

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 5) )

    for pat in range(n_patients):
        ax.plot ( x, data_display[pat, :], linewidth=1, color=outcome_clrs, alpha = 0.5, label=pat )

    # average psd
    data_avg = np.nanmean(data_display, axis=0)
    ax.plot ( x, data_avg, linewidth=3, color=clr_avg, label="average across \n subjects" )

    # display legend
    # ax.legend ( bbox_to_anchor=(1.04, 1), borderaxespad=0, ncol=2 )
    # ax.get_legend().remove()
    ax.set_xscale ( "log" )

    if is_freq_vis == True:
        ax.set ( xlabel="frequency (cycles/day)" )
    else:
        ax.set ( xlabel="cycle period (days/cycle)" )

    ax.set ( ylabel="amplitude" )

    return ax






def plot_heatmap_psd_peaks(data_display, frequency_est, all_patients, peaks_id, left_ips, right_ips,
                           is_freq_vis = False,
                           cmap = None, ax = None, vmin = 0, vmax = 1):

    if cmap is None:
        green_palette = sns.light_palette ( "seagreen", as_cmap=True )

    if is_freq_vis == True:
        x = frequency_est
    else:
        cycle_est = 1 / frequency_est
        x = cycle_est

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 5) )

    # Compute x and y grids, passed to `ax.pcolormesh()`.
    xgrid = x
    ygrid = np.arange ( 0, len ( all_patients ) )
    cp = ax.pcolormesh ( xgrid, ygrid, data_display, vmin = vmin, vmax = vmax, cmap=green_palette )
    ax.set_frame_on ( False )  # remove all spines
    ax.set_yticks ( ygrid )  # set yticks
    ax.set_yticklabels ( all_patients )
    ax.set_xscale ( "log" )

    for ii in range ( len ( peaks_id ) ):
        # plot dotted line in each peak
        peak_ii = peaks_id[ii]
        ax.axvline ( x[peak_ii], color="k", linestyle="--", linewidth=1 )
        # plot lines on the edges of peaks
        left_ip = left_ips[ii]
        right_ip = right_ips[ii]
        ax.axvline ( x[left_ip], color="red", linestyle="--", linewidth=1 )
        ax.axvline ( x[right_ip], color="red", linestyle="--", linewidth=1 )

    if len ( peaks_id ) != 1:
        x_ticks = [1.e-03, 1.e-02, 1, 10]
        for jj in range ( 1, len ( peaks_id ) ):
            x_tick_peak_l = x[left_ips[jj]]
            x_tick_peak_r = x[right_ips[jj]]
            x_ticks = x_ticks + [x_tick_peak_l, x_tick_peak_r]
            ax.set_xticks ( x_ticks )  # set x ticks
            ax.set_xticklabels ( x_ticks, rotation=0 )  # x tick labels

    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    if is_freq_vis == True:
        ax.set ( xlabel="frequency (cycles/day)" )
    else:
        ax.set ( xlabel="cycle period (days/cycle)" )

    ax.set ( ylabel="subjects" )

    return ax, cp








