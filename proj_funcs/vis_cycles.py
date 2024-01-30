import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import sklearn.metrics as sk_metrics
import scipy.stats
import pandas as pd
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def plot_cycles_t_days (t_days, cycle_channel, ylabel, fig_title,  linewidth, label_coords = None, ax=None, ylim = None, clr=(92 / 255, 52 / 255, 127 / 255)):
    # number of time points (windows)
    n_win = len ( t_days )

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots ()

    if label_coords is None:
        x = -0.15
        y = 0.26
    else: # label_coords is a tuple
        x = label_coords[0]
        y = label_coords[1]

    # plot
    ax.plot ( t_days, cycle_channel, linewidth=linewidth, color=clr )

    # limits
    ax.set_xlim ( min ( t_days ), max ( t_days ) )

    if ylim is not None:
        ax.set_ylim ( ylim )

    # labels
    win_per_hr = float(round(1 / ((t_days[1] - t_days[0]) * 24)))  # number of windows per hr
    if win_per_hr.is_integer ():
        win_per_hr = int ( win_per_hr )
    else:
        raise ValueError ( 'There must be an integer number of windows per hour' )

    ax.set_ylabel ( ylabel, rotation=0 )
    ax.set_title (fig_title)
    ax.yaxis.set_label_coords ( x, y )

    tick_x = np.arange ( 0, n_win, win_per_hr * 12 )  # make tick mark every 12 hrs
    ax.set_xticks ( t_days[tick_x] )  # set x ticks
    ax.set_xlabel ( 'time (days)' )  # x axis label

    return ax



def plot_cycles_all_t_days (t_days, cycle_channels, roi_names, roi_is_resect, fig_title, display_yticks = False,  label_coords = None, ax=None, ylim = None, outcome_clrs = None,showspines=False):
    # number of time points (windows)
    [n_roi, n_win] = cycle_channels.shape

    # sort ROIs
    # resected ROIs will be at top of plot
    idx = np.argsort ( np.invert ( roi_is_resect ) )

    if outcome_clrs is None:
        # # plot colours
        # cmap_cycle = cm.get_cmap ( 'hsv' )
        # # Segmenting the whole range (from 0 to 1) of the color map into multiple segments
        # outcome_clrs = cmap_cycle ( np.linspace ( 0, 1, n_roi ) )
        # colours for resected against spared
        cmap = cm.get_cmap ('Set1', 8 )
        label_clrs = np.zeros ( (2, 3) )
        label_clrs[0, :] = cmap ( 1 )[:3] # blue colour
        label_clrs[1, :] = cmap ( 0 )[:3] # red colour

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots (n_roi, 1)

    ii = 0
    for ch, axx in zip ( idx[:-1], ax[:-1] ):
        if roi_is_resect[ch] == 1:
            # resected
            linewidth = 0.5
            color_l = label_clrs[1]
            axx.yaxis.label.set_color ( color_l )
        else:
            # spared
            linewidth = 0.5
            color_l = label_clrs[0]
            axx.yaxis.label.set_color ( color_l )

        if ii == 0:
            # plot the title once in the first plot
            plot_cycles_t_days ( t_days, cycle_channels[ch], linewidth = linewidth, ylabel=roi_names[ch],
                                 fig_title=fig_title, label_coords = label_coords, ax=axx, ylim=ylim, clr=color_l )
        else:
            # do not plot the title in the remaining plots
            plot_cycles_t_days ( t_days, cycle_channels[ch], linewidth = linewidth, ylabel=roi_names[ch],
                                 fig_title="", label_coords = label_coords, ax=axx, ylim=ylim, clr=color_l )
        if display_yticks == False:
            axx.set_yticks ( [] )

        # do not display xtick marks and xlabel in all plot except the last one
        axx.set_xticks ( [] )
        #axx.xaxis.set_visible(False) # same for y axis.
        
        axx.set_xlabel ( "" )
        ii = ii + 1

    if roi_is_resect[idx[-1]] == 1:
        # resected
        linewidth = 0.5
        color_l = label_clrs[1]
        axx.yaxis.label.set_color ( color_l )
    else:
        # spared
        linewidth = 0.5
        color_l = label_clrs[0]
        axx.yaxis.label.set_color ( color_l )

    plot_cycles_t_days ( t_days, cycle_channels[idx[-1]], linewidth = linewidth, ylabel= roi_names[idx[-1]],
                                    fig_title="", ax=ax[-1], label_coords = label_coords, ylim=ylim, clr=color_l )
    ax[-1].set_xlabel ( "time (days)" )
    ax[-1].yaxis.label.set_color ( color_l )
    ax[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    # Do yticks will be displayed?
    if display_yticks == False:
        ax[-1].set_yticks ( [] )

    if ylim is not None:
        ax[-1].set_ylim ( ylim )
    
    if showspines==False:
        for axx in ax:
            for spine in ['top', 'right', 'left', 'bottom']:
                axx.spines[spine].set_visible(False)
    
    return ax

def plot_barplot_measure(cycle_1measure, roi_is_resect, xlim = None, fig_title = None, ax = None, outcome_clrs = None, display_values=False):

    n_roi = cycle_1measure.shape[0]

    if outcome_clrs is None:
        # # plot colours
        # cmap_cycle = cm.get_cmap ( 'hsv' )
        # # Segmenting the whole range (from 0 to 1) of the color map into multiple segments
        # outcome_clrs = cmap_cycle ( np.linspace ( 0, 1, n_roi ) )
        cmap = cm.get_cmap ( 'Set1', 8 )
        label_clrs = np.zeros ( (2, 3) )
        label_clrs[0, :] = cmap ( 1 )[:3]  # blue colour
        label_clrs[1, :] = cmap ( 0 )[:3]  # red colour
    # sort ROIs
    # resected ROIs will be at top of plot
    idx = np.argsort ( np.invert ( roi_is_resect ) )

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots ( n_roi, 1 )

    ii = 0
    for ch, axx in zip(idx[:-1], ax[:-1]):
        if roi_is_resect[ch] == 1:
            color_l = label_clrs[1]
        else:
            # spared
            color_l = label_clrs[0]

        df_disp = pd.DataFrame ( {"x": cycle_1measure[ch], "y": ["measure"]})
        sns.barplot ( x="x", y="y", data=df_disp, width=0.5, color=color_l, ax=axx )
        axx.set_yticks ( [] )
        axx.set_xticks ( [] )
        axx.set_ylabel("")
        axx.set_xlabel ("")
        # display values
        if display_values:
            axx.bar_label ( axx.containers[0])

        if xlim is not None:
            axx.set_xlim ( xlim )

        if ii == 0:
            axx.set_title(fig_title)
        else:
            axx.set_title("")

        ii = ii + 1

    df_disp = pd.DataFrame ( {"x": cycle_1measure[idx[-1]], "y": ["measure"]} )
    if roi_is_resect[idx[-1]] == 1:
        color_l = label_clrs[1]
    else:
        # spared
        color_l = label_clrs[0]
    sns.barplot ( x="x", y="y", data=df_disp, width=0.5, color=color_l, ax=ax[-1] )
    # display values
    if display_values:
        ax[-1].bar_label ( ax[-1].containers[0])

    # Do yticks will be displayed?
    ax[-1].set_yticks ( [] )
    ax[-1].set_ylabel ("")
    ax[-1].set_xticks ( [] )
    ax[-1].set_xlabel ("")

    if xlim is not None:
        ax[-1].set_xlim ( xlim )
    
    for axx in ax:
        for spine in ['top', 'right', 'bottom']:
            axx.spines[spine].set_visible(False)

    return ax

def plot_cycles_RS_t_days(t_days, cycle_channel, ylabel, fig_title,  linewidth,  ax=None, ylim = None, outcome_clrs = None):

    # number of time points (windows)
    n_win = len ( t_days )

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots ()

    if outcome_clrs is None:
        cmap = cm.get_cmap ('Set1', 8 )
        outcome_clrs = np.zeros ( (2, 3) )
        outcome_clrs[0, :] = cmap ( 1 )[:3] # spared
        outcome_clrs[1, :] = cmap ( 0 )[:3] # resected

    # plot
    ax.plot ( t_days, cycle_channel[0,:], label = "resected", linewidth=linewidth, color=outcome_clrs[1, :] )
    ax.plot ( t_days, cycle_channel[1,:], label = "spared", linewidth=linewidth, color=outcome_clrs[0, :] )

    # limits
    ax.set_xlim ( min ( t_days ), max ( t_days ) )

    if ylim is not None:
        ax.set_ylim ( ylim )

    # labels
    win_per_hr = 1 / ((t_days[1] - t_days[0]) * 24)  # number of windows per hr
    if win_per_hr.is_integer ():
        win_per_hr = int ( win_per_hr )
    else:
        raise ValueError ( 'There must be an integer number of windows per hour' )

    ax.set_ylabel ( ylabel)
    ax.set_title ( fig_title )
    ax.legend()
    tick_x = np.arange ( 0, n_win, win_per_hr * 12 )  # make tick mark every 12 hrs
    ax.set_xticks ( t_days[tick_x] )  # set x ticks
    ax.set_xlabel ( 'time (days)' )  # x axis label

    return ax


def plot_cycles_all_t_days_comp (t_days, cycle_channels, roi_is_resect, fig_title, display_yticks = False,  label_coords = None, ax=None, ylim = None, outcome_clrs = None):

    # sort ROIs
    # resected ROIs will be at beginning of array
    idx = np.argsort ( np.invert ( roi_is_resect ) )
    # flip the array
    idx = np.flip(idx)

    if outcome_clrs is None:
        # # plot colours
        # cmap_cycle = cm.get_cmap ( 'hsv' )
        # # Segmenting the whole range (from 0 to 1) of the color map into multiple segments
        # outcome_clrs = cmap_cycle ( np.linspace ( 0, 1, n_roi ) )
        # colours for resected against spared
        cmap = cm.get_cmap ('Set1', 8 )
        label_clrs = np.zeros ( (2, 3) )
        label_clrs[0, :] = cmap ( 1 )[:3] # blue colour
        label_clrs[1, :] = cmap ( 0 )[:3] # red colour

    # make new figure if axes not supplied
    if ax is None:
        fig, ax = plt.subplots (1, 1)

    for ch in idx:
        if roi_is_resect[ch] == 1:
            # resected
            linewidth = 0.5
            color_l = label_clrs[1]
        else:
            # spared
            linewidth = 0.5
            color_l = label_clrs[0]

        # plot the title once in the first plot
        plot_cycles_t_days ( t_days, cycle_channels[ch], linewidth = linewidth, ylabel="",
                             fig_title=fig_title, label_coords = label_coords, ax=ax, ylim=ylim, clr=color_l )

        if display_yticks == False:
            ax.set_yticks ( [] )

        # do not display xtick marks and xlabel in all plot except the last one
        ax.set_xticks ( [] )
        ax.set_xlabel ( "time (days)" )

    if ylim is not None:
        ax.set_ylim ( ylim )

    return ax
