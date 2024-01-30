import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt



def plot_cycle_timeOfday (time_x, time_x_names, daily_var_dayAvg, roi_names, ylabel, fig_title, is_avg_plotted = True,tick_Hfreq = 2,  ax=None, ylim = None, outcome_clrs=None, clr_avg = None):

    [n_roi, n_win] = daily_var_dayAvg.shape # n_win should be 24 hours

    if outcome_clrs is None:
        # plot colours
        cmap = cm.get_cmap ( 'viridis' )
        # Segmenting the whole range (from 0 to 1) of the color map into multiple segments
        outcome_clrs = cmap ( np.linspace ( 0, 1, n_roi + 1) )

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots ()

    for ch in range(n_roi):
        # plot
        ax.plot ( time_x, daily_var_dayAvg[ch, :], linewidth=0.5, color=outcome_clrs[ch] , label = roi_names[ch])

    # Is the average plotted?
    if is_avg_plotted is True and clr_avg is not None:
        global_avg = np.nanmean(daily_var_dayAvg, axis=0)
        ax.plot ( time_x, global_avg, linewidth=2, color=clr_avg, label = "Avg")
    elif is_avg_plotted is True and clr_avg is None:
        global_avg = np.nanmean ( daily_var_dayAvg, axis=0 )
        ax.plot ( time_x, global_avg, linewidth=2, color="k", label="Avg" )

    # limits
    ax.set_xlim ( min ( time_x ), max ( time_x ) )

    if ylim is not None:
        ax.set_ylim ( ylim )

    # labels
    tick_x = np.arange ( 0, n_win, tick_Hfreq )  # make tick mark every tick_freq hours
    ax.set_xticks ( tick_x )
    ax.set_xticklabels ( time_x_names[tick_x], rotation=45 )

    ax.set_ylabel ( ylabel )
    ax.set_title (fig_title)

    ax.set_xlabel ( 'time of day' )  # x axis label

    return ax


def plot_cycle_shade_timeOfday(time_x, time_x_names, daily_var_dayAvg, ylabel, legend_label, fig_title, tick_Hfreq = 2,  ax=None, ylim = None, clr=(92 / 255, 52 / 255, 127 / 255)):

    [n_roi, n_win] = daily_var_dayAvg.shape # n_win should be 24 hours

    # Compute the average across channels
    global_avg = np.nanmean ( daily_var_dayAvg, axis=0 )
    global_std = np.nanstd ( daily_var_dayAvg, axis=0 )

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots ()

    # plot
    ax.plot ( time_x, global_avg, linewidth=0.7, color=clr, label=legend_label)
    ax.fill_between( time_x, global_avg - global_std, global_avg + global_std, alpha = 0.2, color=clr)
    # limits
    ax.set_xlim ( min ( time_x ), max ( time_x ) )

    if ylim is not None:
        ax.set_ylim ( ylim )

    # labels
    tick_x = np.arange ( 0, n_win, tick_Hfreq )  # make tick mark every tick_freq hours
    ax.set_xticks ( tick_x )
    ax.set_xticklabels ( time_x_names[tick_x], rotation=45 )

    ax.set_ylabel ( ylabel )
    ax.set_title ( fig_title )

    ax.set_xlabel ( 'time of day' )  # x axis label

    return ax

def plot_cycle_shade_Compare_timeOfday(time_x, time_x_names, resected_channels, spared_channels, ylabel, fig_title, tick_Hfreq = 2,  ax=None, ylim = None, outcome_clrs = None):

    [_, n_win] = resected_channels.shape # n_win should be 24 hours

    # Compute the average/std across resected and spared channels
    # average or median?
    resected_global_avg = np.nanmedian ( resected_channels, axis=0 )
    spared_global_avg = np.nanmedian ( spared_channels, axis=0 )

    resected_global_std = np.nanstd ( resected_channels, axis=0 )
    spared_global_std = np.nanstd ( spared_channels, axis=0 )

    if outcome_clrs is None:
        cmap = cm.get_cmap ('Set1', 8 )
        outcome_clrs = np.zeros ( (2, 3) )
        outcome_clrs[0, :] = cmap ( 1 )[:3]
        outcome_clrs[1, :] = cmap ( 0 )[:3]

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots ()

    # plot
    # resected
    ax.plot ( time_x, resected_global_avg, linewidth=0.7, color=outcome_clrs[1,:], label="resected" )
    ax.fill_between ( time_x, resected_global_avg - resected_global_std, resected_global_avg + resected_global_std, alpha=0.2, color=outcome_clrs[1,:] )

    # spared
    ax.plot ( time_x, spared_global_avg, linewidth=0.7, color=outcome_clrs[0, :], label="spared" )
    ax.fill_between ( time_x, spared_global_avg - spared_global_std, spared_global_avg + spared_global_std,
                      alpha=0.2, color=outcome_clrs[0, :] )


    if ylim is not None:
        ax.set_ylim ( ylim )

    # labels
    tick_x = np.arange ( 0, n_win, tick_Hfreq )  # make tick mark every tick_freq hours
    ax.set_xticks ( tick_x )
    ax.set_xticklabels ( time_x_names[tick_x], rotation=45 )

    ax.set_ylabel ( ylabel )
    ax.set_title ( fig_title )

    ax.set_xlabel ( 'time of day' )  # x axis label

    return ax

def plot_cycle_line_Compare_timeOfday(time_x, time_x_names, resected_channels, spared_channels, ylabel, fig_title, tick_Hfreq = 2,  ax=None, ylim = None, outcome_clrs = None):

    [n_roi_resected, n_win] = resected_channels.shape # n_win should be 24 hours
    [n_roi_spared, n_win] = spared_channels.shape # n_win should be 24 hours

    # Compute the average/std across resected and spared channels
    # average or median?
    resected_global_avg = np.nanmedian ( resected_channels, axis=0 )
    spared_global_avg = np.nanmedian ( spared_channels, axis=0 )


    if outcome_clrs is None:
        cmap = cm.get_cmap ('Set1', 8 )
        outcome_clrs = np.zeros ( (2, 3) )
        outcome_clrs[0, :] = cmap ( 1 )[:3]
        outcome_clrs[1, :] = cmap ( 0 )[:3]

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots ()

    for ch in range(n_roi_resected):
        # plot
        # resected
        ax.plot ( time_x, resected_channels[ch,:], linewidth=0.5, alpha = 0.5, color=outcome_clrs[1,:], label="resected" )
    for ch in range ( n_roi_spared ):
        # spared
        ax.plot ( time_x, spared_channels[ch,:], linewidth=0.5, alpha = 0.5, color=outcome_clrs[0, :], label="spared" )

    # the median
    ax.plot ( time_x, resected_global_avg, "--", linewidth=2, color=outcome_clrs[1, :], label="resected" )
    ax.plot ( time_x, spared_global_avg, "--", linewidth=2, color=outcome_clrs[0, :], label="spared" )

    if ylim is not None:
        ax.set_ylim ( ylim )

    # labels
    tick_x = np.arange ( 0, n_win, tick_Hfreq )  # make tick mark every tick_freq hours
    ax.set_xticks ( tick_x )
    ax.set_xticklabels ( time_x_names[tick_x], rotation=45 )

    ax.set_ylabel ( ylabel )
    ax.set_title ( fig_title )

    ax.set_xlabel ( 'time of day' )  # x axis label

    return ax
