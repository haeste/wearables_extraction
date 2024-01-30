"""
This python file includes:

    format_global
    format_func

    visualise_imfs_psd
    plot_psd_peaks
    scatter_ampl_freq_peaks

    dotplot_peaks_allP
    dot_plot_count

    plot_heatmap_psd

    scatter_ampl_freq_shade_avg

    barplot_count_subj

    plot_cycle_allP
    plot_cycle_allP_colP
    plot_outcome_auc

    dot_plot_subjects

    plot_outcome_auc_metadata

    dotplot_Drs_fb_size
    violinplot_circ_pathology

    scatterplot_corr
    dotplot_Drs_metadata
"""


# external modules
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scipy.stats
import matplotlib.colors as plt_colors
import sklearn.metrics as sk_metrics



def map_timescale(x):
    if x < 0.8 and x >= 0:
        return "ultradian"
    elif x >= 0.8 and x <= 1.3:
        return "circadian"
    else:
        return "multidien"

def map_color(timescale):
    if timescale == "circadian":
        color = "#05712f"
    elif timescale == "ultradian":
        color = "#572c92"
    elif timescale == "multidien":
        color = "#ad3803"
    return color


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


def visualise_imfs_psd(power_est_allimfs, frequency_est, imfs_range = None, outcome_clrs=None, ax=None, log_scale_yaxis = False, is_freq_vis = True, vis_line_sum = True):
    """
    This function works for the exact range of IMFs needed to display because it modifies the names
    based on that.
    Args:
        power_est_allimfs:
        frequency_est:
        imfs_range:
        outcome_clrs:
        ax:
        log_scale_yaxis:
        is_freq_vis:
        vis_line_sum:

    Returns:

    """
    [modes, nfreqs] = power_est_allimfs.shape

    if outcome_clrs is None:
        if is_freq_vis == True:
            # Choose your desired colormap
            cmap = cm.get_cmap ('viridis' )
        else:
            cmap = cm.get_cmap ('viridis_r' )

        # Segmenting the whole range (from 0 to 1) of the color map into multiple segments
        outcome_clrs = cmap ( np.linspace ( 0, 1, modes ) )

    # choose the imf range to plot
    # if not plot all imfs
    if imfs_range == None:
        lowest_imf_id = 0
        highest_imf_id = modes
    else:
        lowest_imf_id = imfs_range[0]
        highest_imf_id = imfs_range[1]

    if is_freq_vis == True:
        x = frequency_est
    else:
        cycle_est = 1 / frequency_est
        x = cycle_est

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 5) )

    for imf in range ( lowest_imf_id, highest_imf_id  ):

        if ax is None:
            fig.suptitle ( 'IMFs {}-{}'.format ( lowest_imf_id, highest_imf_id ) )

        ax.plot ( x, power_est_allimfs[imf, :], label="IMF{}".format ( imf + 2 ) , c = outcome_clrs[imf],  linewidth=0.5)

    ax.set ( ylabel="Power index" )
    if is_freq_vis == True:
        ax.set(xlabel = "Frequency (cycles/day)")
    else:
        ax.set(xlabel = "Cycle period (days/cycle)")

    if vis_line_sum == True:
        power_sum = np.sum ( power_est_allimfs[lowest_imf_id:highest_imf_id, :], axis=0 )
        ax.plot ( x, power_sum, label="SUM", c="k", linewidth=0.7 )

    # display legend in the graph
    ax.legend(bbox_to_anchor=(1, 0.98), loc="upper left")

    if log_scale_yaxis == True:
        ax.set_yscale ( "log" )

    # limits
    ax.set_xlim ( min ( x ), max ( x ) )
    ax.set_xscale ( "log" )
    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    return ax


def plot_psd_peaks(data_display_avg, frequency_est, label, peaks_id, left_ips, right_ips,
                   display_peaks_labels = False, is_freq_vis = False, outcome_clrs = None, shade_clrs = None, ylim = None, ax = None):

    if outcome_clrs is None:
        outcome_clrs = "k"

    if shade_clrs is None:
        shade_clrs = "gray"

    if is_freq_vis == True:
        x = frequency_est
    else:
        cycle_est = 1 / frequency_est
        x = cycle_est

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 5) )

    ax.plot ( x, data_display_avg, linewidth=1, color=outcome_clrs, alpha = 1, label=label )

    ax.scatter ( x[peaks_id], data_display_avg[peaks_id], marker="o" )

    for ii in range ( len ( peaks_id ) ):
        # plot dotted line in each peak
        peak_ii = peaks_id[ii]
        ax.axvline ( x[peak_ii], color="k", linestyle="--", linewidth=0.7 )
        # plot shaded area around each peak
        left_ip = left_ips[ii]
        right_ip = right_ips[ii]
        ax.axvspan ( x[left_ip], x[right_ip], color = shade_clrs, alpha=0.4 )

    if display_peaks_labels == True:
        ax.set_xscale ( "log" )
        if len(peaks_id) != 1:
            x_ticks = [min(x), 1.e-02, max(x)]
            for jj in range(0, len(peaks_id)):
                #x_tick_peak_l = x[left_ips[jj]]
                #x_tick_peak_r = x[right_ips[jj]]
                #x_ticks = x_ticks + [x_tick_peak_l, x_tick_peak_r]
                x_ticks.append(x[peaks_id[jj]])
        else:
            x_ticks = [min(x), 1.e-02, 1, max(x)]

        ax.set_xticks ( x_ticks )  # set x ticks
        ax.set_xticklabels ( x_ticks, rotation=0 )  # x tick labels
        ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )
    else:
        ax.set_xlim ( min ( x ), max ( x ) )
        ax.set_xscale ( "log" )
        ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    # y-axis limits
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if is_freq_vis == True:
        ax.set ( xlabel="frequency (cycles/day)" )
    else:
        ax.set ( xlabel="cycle period (days/cycle)" )

    ax.set ( ylabel="power" )

    return ax


def scatter_ampl_freq_peaks(data_df, ax = None, xlim = None):

    # colours for selected peaks vs not selected
    #colors = {1: "#380283", 0: "#C5C9C7"}
    #colors = {"ultradian": "#D2691E", "circadian": "#DBB40C", "multidien": "#6E750E", 0: "#C5C9C7"}
    colors = {"ultradian": "#572c92", "circadian": "#05712f", "multidien": "#ad3803", 0: "#C5C9C7"}

    x = data_df["cycle_len_peak"].to_numpy ().flatten ()
    y = data_df["power_peak"].to_numpy ().flatten ()

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 4) )

    # make scatterplot
    ax.plot(x, y, '-', color = "silver", linewidth = 2, zorder=0)
    ax.scatter ( x, y, c = data_df["final_selected_peaks"].apply(lambda x: colors[x]), s = 90, zorder=1)

    for i, txt in enumerate ( range(len(x)) ):
        for key in colors.keys():
            if data_df["final_selected_peaks"][i] == key and data_df["final_selected_peaks"][i] != 0:
                ax.annotate ( format_global(data_df["cycle_len_peak"][i]), (x[i], y[i]) , color = colors[key])

    ax.set_xscale ( "log" )

    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    ax.set ( ylabel="power" )
    ax.set ( xlabel="cycle period" )

    return ax

def dotplot_peaks_allP(data_df, ax = None):

    # colors = {"ultradian": "#D2691E", "circadian": "#DBB40C", "multidien": "#6E750E"}
    colors = {"ultradian": "#572c92", "circadian": "#05712f", "multidien": "#ad3803"}

    x = data_df["cycle_len_peak"].to_numpy ().flatten ()
    y = data_df["subject_id"].to_numpy ().flatten ()
    y_labels = data_df["subject"].to_numpy ().flatten ()
    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 9) )

    # make scatterplot
    ax.scatter ( x, y, c = data_df["what_peak"].apply(lambda x: colors[x]))

    ax.set_xscale ( "log" )

    # set xticks
    x_ticks = [0.0015, 1.e-02, 0.01, 0.1, 0.5, 1]
    # x_ticks = ax.get_xticks()
    ax.set_xticks ( x_ticks )  # set x ticks
    ax.set_xticklabels ( x_ticks, rotation=0 )  # x tick labels
    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    # set yticks
    ax.set_yticks(y)
    ax.set_yticklabels ( y_labels, rotation=0 )  # y tick labels

    ax.set ( ylabel="subjects" )
    ax.set ( xlabel="cycle period" )

    return ax

def dot_plot_count(data_df, ax = None, apply_format = False):

    # colours for selected peaks vs not selected
    # colors = {"ultradian": "#D2691E", "circadian": "#DBB40C", "multidien": "#6E750E"}
    colors = {"ultradian": "#572c92", "circadian": "#05712f", "multidien": "#ad3803"}

    cp_subj = data_df["cycle_len_peak"]

    n_bins_cp = 50
    bins_cp = np.logspace ( np.log10 ( 0.0015 ), np.log10 ( 4.7 ), n_bins_cp, endpoint=True )
    bin_counts, bin_edges, binnumber = stats.binned_statistic ( x=cp_subj,
                                                                values=cp_subj, statistic='count',
                                                                bins=bins_cp )
    cycle_period_est = (bin_edges[:-1] + bin_edges[1:]) / 2

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 4) )

    unique_values = cycle_period_est
    counts = bin_counts
    # get the individual counts for each unique cycle period
    scatter_x = []  # x values
    scatter_y = []  # corresponding y values
    idx = 0
    # array for colours
    what_peak = []
    for value in unique_values:
        if value < 0.8 and value >= 0.0:
            peak_label = "ultradian"
        elif value >= 0.8 and value <= 1.3:
            peak_label = "circadian"
        else:
            peak_label = "multidien"

        for counter in range ( 1, int ( counts[idx] ) + 1 ):
            scatter_x.append ( value )
            scatter_y.append ( counter )
            what_peak.append(peak_label)
        idx = idx + 1

    # draw dot plot using scatter()
    ax.scatter ( scatter_x, scatter_y, c = (pd.Series(what_peak)).map(colors), marker="s", s=20 )
    ax.set_xscale ( "log" )

    # set y labels
    # display every 2 data points on y-axis
    y_ticks = np.arange(np.min(counts), np.max(counts), 2)
    ax.set_yticks(y_ticks)

    # display only the cycle periods that appear to have counts
    x_ticks = [scatter_x[i] for i in range(0, len(scatter_y)) if scatter_y[i] != 0 ]
    ax.set_xticks ( x_ticks )

    if apply_format == False:
        ax.set_xticklabels ( np.round ( x_ticks, 3 ), rotation=90 )  # x tick labels
    else:
        ax.set_xticklabels ( x_ticks, rotation=90 )
        ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    ax.set ( ylabel="number of subjects" )
    ax.set ( xlabel="cycle period" )

    return ax

def plot_heatmap_psd(data_display, cycle_period, all_patients,
                           cmap = None, ax = None, vmin = 0, vmax = 1):

    if cmap is None:
        cmap = sns.light_palette ( "seagreen", as_cmap=True )

    x = cycle_period

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 5) )

    # Compute x and y grids, passed to `ax.pcolormesh()`.
    xgrid = x
    ygrid = np.arange ( 0, len ( all_patients ) )
    cp = ax.pcolormesh ( xgrid, ygrid, data_display, vmin = vmin, vmax = vmax, cmap=cmap, snap = True )
    ax.set_frame_on ( False )  # remove all spines
    ax.set_yticks ( ygrid )  # set yticks
    ax.set_yticklabels ( all_patients )
    ax.set_xscale ( "log" )

    # display only the cycle periods that appear to have power
    data_sum = np.sum(data_display, axis=0)
    x_ticks = [x[i] for i in range ( 0, len ( x ) ) if data_sum[i] != 0]
    ax.set_xticks ( x_ticks )  # set x ticks
    ax.set_xticklabels ( x_ticks, rotation=90 )  # x tick labels

    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    ax.set ( xlabel="cycle period (days/cycle)" )
    ax.set ( ylabel="subjects" )

    return ax, cp

def scatter_ampl_freq_shade_avg(data_mat, cycle_period_est, ax = None):

    # colours for selected peaks vs not selected
    #colors = {1: "#380283", 0: "#C5C9C7"}
    # colors = {"ultradian": "#D2691E", "circadian": "#DBB40C", "multidien": "#6E750E", 0: "#C5C9C7"}
    colors = {"ultradian": "#572c92", "circadian": "#05712f", "multidien": "#ad3803", 0: "#C5C9C7"}

    # compute average across subjects
    data_avg = np.mean(data_mat, axis=0)

    ### combine cycle period and average power in one dataframe
    data_df = pd.DataFrame({"avg_power": data_avg, "cycle_period": cycle_period_est})

    # exclude power data points with 0 avg power
    data_df_subset = data_df[data_df["avg_power"] != 0]
    # form x and y arrays for plot
    x = data_df_subset["cycle_period"].to_numpy().flatten()
    y = data_df_subset["avg_power"].to_numpy().flatten()

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 4) )

    # make scatterplot
    ax.plot(x, y, '-', color = "silver", linewidth = 2, zorder=0)
    ax.scatter(x, y, c = "silver", s = 70, zorder = 1)

    ax.set_xscale ( "log" )

    # display only the cycle periods that appear to have counts
    x_ticks = [x[i] for i in range ( 0, len ( x ) ) if y[i] != 0]
    ax.set_xticks ( x_ticks )
    ax.set_xticklabels ( x_ticks, rotation=90 )
    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    ax.set_xlim(np.min(cycle_period_est), np.max(cycle_period_est))

    ax.set ( ylabel="power" )
    ax.set ( xlabel="cycle period" )

    return ax

def barplot_count_subj(data_df, which_timescale = "all", ax = None):


    colors = {"ultradian": "#572c92", "circadian": "#05712f", "multidien": "#ad3803"}

    n_bins_cp = 50
    bins_cp = np.logspace ( np.log10 ( 0.0015 ), np.log10 ( 4.7 ), n_bins_cp, endpoint=True )

    cp_subj = data_df["cyclep_est"]
    bin_counts, bin_edges, binnumber = stats.binned_statistic ( x=cp_subj,
                                                                values=cp_subj, statistic='count',
                                                                bins=bins_cp )

    cycle_period_est = (bin_edges[:-1] + bin_edges[1:]) / 2

    # array for colours
    what_peak = []
    for value in cycle_period_est:
        if value < 0.8 and value >= 0.0:
            what_peak.append("ultradian")
        elif value >= 0.8 and value <= 1.3:
            what_peak.append("circadian")
        else:
            what_peak.append("multidien")

    count_df = pd.DataFrame({"cycle_period": cycle_period_est, "count": bin_counts, "what_peak": what_peak})

    if which_timescale == "ultradian":
        count_df_sub = count_df[count_df["what_peak"] == "ultradian"]

    elif which_timescale == "circadian":
        data_counts = count_df[count_df["what_peak"] == "circadian"]["count"].sum()
        count_df_sub = pd.DataFrame({"cycle_period": [1], "count": [data_counts], "what_peak": ["circadian"]})

    else:
        count_df_sub = count_df.copy()

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 4) )

    if which_timescale == "all":
        ax.bar ( x=count_df_sub["cycle_period"].apply ( lambda x: format_global ( x ) ), height=count_df_sub["count"],
                 color=count_df_sub["what_peak"].apply(lambda x: colors[x]), width=0.7, align='center' )
    else:
        ax.bar ( x=count_df_sub["cycle_period"].apply(lambda x: format_global(x)), height=count_df_sub["count"], color=colors[which_timescale], width=0.7, align='center' )

    plt.xticks ( rotation=90 )
    ax.set ( ylabel="number of subjects" )
    ax.set ( xlabel="timescale of cycles" )

    return ax, count_df_sub


def plot_cycle_allP (data_df, ylim=[0, 1], x_var = None,
                      outcome_clrs=None, meas_ref=0.5, ax=None, size=7,
                      plot_type='beeswarm', p_side='less', overlay_violin=True,
                      violin_tint=0.9):
    if outcome_clrs is None:
        # colors = {"ultradian": "#572c92", "circadian": "#05712f", "multidien": "#ad3803"}
        # #colors = {"ultradian": "#D2691E", "circadian": "#DBB40C", "multidien": "#6E750E"}
        # clr = data_df["what_peak"].apply(lambda x: colors[x]).unique()[0]
        outcome_clrs = "#C5C9C7"

    clr_rgb = plt_colors.hex2color(outcome_clrs)

    # get variable for measure
    meas_var = data_df["Drs"].to_numpy ().flatten ()

    # compute AUC
    if meas_var.shape[0] > 0:
        diff = 0.5 - meas_var
        _, p = scipy.stats.wilcoxon ( diff, alternative=p_side,
                                      mode='exact' )  # less = 0.5<patient D_RSs
        # H0: Drs=0.5 with alternative that
        # H1: Drs>0.5
    else:
        p = np.nan

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    ax.set_ylim ( ylim )
    ax.axhline ( meas_ref, color=(0.5, 0.5, 0.5) , linestyle ="--")

    if plot_type == 'beeswarm':

        # violin
        if overlay_violin:
            # lighten clrs
            v_clrs = np.array([cc * 255 for cc in clr_rgb])
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=data_df, y='Drs', x = x_var,
                             color=v_clrs, ax=ax, saturation=1,
                             inner='quartiles', cut=0 ) # CAN USE cut = 0.5 AS WELL

        # beeswarm
        sns.swarmplot ( data=data_df, y='Drs', x = x_var,
                        color = outcome_clrs,
                        size=size, ax=ax )

    ax.set_title (
        f'H0:Drs = 0.5, H1:Drs > 0.5: {round ( p, 5 )} ({p_side})' )
    ax.set_ylabel ( '$D_{RS}$')
    ax.set_xlabel ( "timescale" )

    if x_var == "cyclep_est":
        x_ticks_l = list(data_df["cyclep_est"].drop_duplicates().sort_values(ascending=True).to_numpy())
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)
        x_tick_labels = [format_global(xt) for xt in x_ticks_l]
        ax.set_xticklabels(x_tick_labels)
    else:
        ax.set_xticklabels ( data_df["what_peak"].drop_duplicates().to_numpy().tolist())
    return ax, p

def plot_cycle_allP_colP (data_df, ylim=[0, 1], x_var = None,
                      outcome_clrs=None, meas_ref=0.5, ax=None, size=7,
                      plot_type='beeswarm', p_side='less', overlay_violin=True,
                      violin_tint=0.9):

    pat_colors = ["#a9a9a9", "#2f4f4f", "#556b2f", "#a0522d", "#191970", "#006400", "#8b0000",
                  "#808000", "#3cb371", "#bdb76b", "#008b8b", "#4682b4", "#00008b", "#32cd32", "#daa520",
                  "#800080", "#b03060", "#ff4500", "#ff8c00", "#ffff00", "#00ff00", "#00fa9a", "#dc143c",
                  "#00ffff", "#00bfff", "#f4a460", "#0000ff", "#a020f0", "#adff2f", "#da70d6", "#ff00ff",
                  "#1e90ff", "#fa8072", "#dda0dd", "#ff1493", "#7b68ee", "#afeeee", "#ffdab9", "#ffb6c1"]

    n_pat = data_df["subject"].drop_duplicates().count()

    pat_colors = pat_colors[0:n_pat]

    # make a color palette using those custom colors
    patients_clrs = sns.color_palette ( pat_colors )

    outcome_clrs = "#C5C9C7"
    clr_rgb = plt_colors.hex2color(outcome_clrs)

    # get variable for measure
    meas_var = data_df["Drs"].to_numpy ().flatten ()

    # compute AUC
    if meas_var.shape[0] > 0:
        diff = 0.5 - meas_var
        _, p = scipy.stats.wilcoxon ( diff, alternative=p_side,
                                      mode='exact' )  # less = 0.5<patient D_RSs
        # H0: Drs=0.5 with alternative that
        # H1: Drs>0.5
    else:
        p = np.nan

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    ax.set_ylim ( ylim )
    ax.axhline ( meas_ref, color=(0.5, 0.5, 0.5) , linestyle ="--")

    if plot_type == 'beeswarm':

        # violin
        if overlay_violin:
            # lighten clrs
            v_clrs = np.array([cc * 255 for cc in clr_rgb])
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=data_df, y='Drs', x = x_var,
                             color=v_clrs, ax=ax, saturation=1,
                             inner='quartiles', cut=0 ) # CAN USE cut = 0.5 AS WELL

        # beeswarm
        sns.swarmplot ( data=data_df, y='Drs', hue = "subject", x = x_var,
                        palette = patients_clrs,
                        size=size, ax=ax )

    ax.legend ( bbox_to_anchor=(1.04, 1), borderaxespad=0, ncol=2 )

    ax.set_title (
        f'H0:Drs = 0.5, H1:Drs > 0.5: {round ( p, 5 )} ({p_side})' )
    ax.set_ylabel ( '$D_{RS}$')
    ax.set_xlabel ( "timescale" )

    if x_var == "cyclep_est":
        x_ticks_l = list(data_df["cyclep_est"].drop_duplicates().sort_values(ascending=True).to_numpy())
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)
        x_tick_labels = [format_global(xt) for xt in x_ticks_l]
        ax.set_xticklabels(x_tick_labels)
    else:
        ax.set_xticklabels ( data_df["what_peak"].drop_duplicates().to_numpy().tolist())
    return ax, p

def plot_AUC_per_cycle(data_df, ylim=[0, 1], x_var = None,
                      outcome_clrs=None, meas_ref=0.5, ax=None, size=7,
                      plot_type='beeswarm', p_side='less', overlay_violin=True,
                      violin_tint=0.9):

    pat_colors = ["#a9a9a9", "#2f4f4f", "#556b2f", "#a0522d", "#191970", "#006400", "#8b0000",
                  "#808000", "#3cb371", "#bdb76b", "#008b8b", "#4682b4", "#00008b", "#32cd32", "#daa520",
                  "#800080", "#b03060", "#ff4500", "#ff8c00", "#ffff00", "#00ff00", "#00fa9a", "#dc143c",
                  "#00ffff", "#00bfff", "#f4a460", "#0000ff", "#a020f0", "#adff2f", "#da70d6", "#ff00ff",
                  "#1e90ff", "#fa8072", "#dda0dd", "#ff1493", "#7b68ee", "#afeeee", "#ffdab9", "#ffb6c1"]

    n_pat = data_df[x_var].drop_duplicates().count()
    if n_pat==1:
        pat_colors = pat_colors[0:n_pat]
    else:
        pat_colors = pat_colors[1:n_pat+1]
    pat_colors = pat_colors[0:n_pat]
    print("Hellow")
    # make a color palette using those custom colors
    patients_clrs = sns.color_palette ( pat_colors )

    outcome_clrs = "#C5C9C7"
    clr_rgb = plt_colors.hex2color(outcome_clrs)

    # get variable for measure
    meas_var = data_df["drs"].to_numpy ().flatten ()

    # compute AUC
    if meas_var.shape[0] > 0:
        diff = 0.5 - meas_var
        _, p = scipy.stats.wilcoxon ( diff, alternative=p_side,
                                      mode='exact' )  # less = 0.5<patient D_RSs
        # H0: Drs=0.5 with alternative that
        # H1: Drs>0.5
    else:
        p = np.nan

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    #ax.set_ylim ( ylim )
    ax.axhline ( meas_ref, color=(0.5, 0.5, 0.5) , linestyle ="--")

    if plot_type == 'beeswarm':

        # violin
        if overlay_violin:
            # lighten clrs
            v_clrs = np.array([cc * 255 for cc in clr_rgb])
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=data_df, y='drs', x = x_var,
                             color=v_clrs, ax=ax, saturation=1,
                             inner='quartiles', cut=0, legend=False ) # CAN USE cut = 0.5 AS WELL

        # beeswarm
        sns.swarmplot ( data=data_df, y='drs', hue =x_var, x = x_var,
                        palette = patients_clrs,
                        size=size, ax=ax, legend=False )
    ax.set_ylabel ( 'AUC')
    #ax.set_yticklabels([0. , 0.2, 0.4, 0.6, 0.8, 1. ], size = 11)
    if n_pat ==1:
        ax.set_xlabel('')
        ax.set_xticklabels([''])
    else:
        ax.set_xlabel ( "Ultradian Rhythm" )
        #ax.set_xticklabels(ax.get_xticks(), size = 11)


def plot_outcome_auc (pnt_ilae, pnt_meas, ilae_thresh=3, ylim=[0, 1], meas_name='$D_{RS}$',
                      outcome_clrs=None, meas_ref=0.5, ax=None, size=7,
                      plot_type='beeswarm', p_side='less', overlay_violin=True,
                      violin_tint=0.9):
    # ilae thresh = min ilae that is bad outcome

    # first remove any nans
    is_nan = np.isnan ( pnt_meas )
    pnt_meas = pnt_meas[np.invert ( is_nan )]
    pnt_ilae = pnt_ilae[np.invert ( is_nan )]
    print ( f'removed {np.sum ( is_nan )} patients due to NaN measure' )
    # define bad outcome patients
    bad_outcome = pnt_ilae >= ilae_thresh

    if outcome_clrs is None:
        cmap = plt.get_cmap( 'Set1', 8 )
        outcome_clrs = np.zeros ( (2, 3) )
        outcome_clrs[0, :] = cmap ( 1 )[:3]
        outcome_clrs[1, :] = cmap ( 0 )[:3]

    # compute AUC
    if  pnt_meas.shape[0] > 0:
        auc = sk_metrics.roc_auc_score ( bad_outcome, pnt_meas )
        _, p = scipy.stats.ranksums ( pnt_meas[bad_outcome == 0],
                                      pnt_meas[bad_outcome == 1],
                                      alternative=p_side )
    else:
        auc = np.nan
        p = np.nan

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    ax.set_ylim ( ylim )
    ax.axhline ( meas_ref, color=(0.5, 0.5, 0.5) , linestyle ="--")
    if len ( pnt_meas ) > 0:

        # make dataframe
        dct = {'bad_outcome': bad_outcome, 'pnt_meas': pnt_meas}
        df = pd.DataFrame ( dct )
        if plot_type == 'beeswarm':

            # violin
            if overlay_violin:
                # lighten clrs
                v_clrs = outcome_clrs * 255
                v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
                v_clrs = v_clrs / 255

                sns.violinplot ( data=df, x='bad_outcome', y='pnt_meas',
                                 palette=v_clrs, ax=ax, saturation=1,
                                 inner='quartiles', cut=0 ) # CAN USE cut = 0.5 AS WELL

                # beeswarm
            sns.swarmplot ( data=df, x='bad_outcome', y='pnt_meas', hue='bad_outcome',
                            size=size,
                            palette=outcome_clrs, ax=ax )

        elif plot_type == 'violin':
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, x='bad_outcome', y='pnt_meas',
                             palette=v_clrs, ax=ax, saturation=1,
                             inner='stick', cut=0 )
        ax.get_legend ().remove ()
        ax.set_xticklabels ( ['good', f'bad (ILAE {ilae_thresh}+)'] )

    ax.set_title (
        f'{meas_name} \n AUC = {round ( auc, 3 )}, p = {round ( p, 5 )} ({p_side}) \n n = {len ( pnt_meas )}' )
    ax.set_ylabel ( "$D_{RS}$" )
    ax.set_xlabel ( 'surgical outcome' )

    return ax, auc, p

def dot_plot_subjects(data_df, cyclep_vec, patients_vec, circ_size_factor = 200, ax = None):
    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    ax.grid ( which='minor' )
    ax.set_aspect ( "equal", "box" )

    ax.scatter ( np.nan, np.nan, marker="o", s=100, facecolors='none', edgecolors='k' )
    ax.scatter ( np.nan, np.nan, marker="o", s=circ_size_factor * (0.5) ** 2 + 100, facecolors='none', edgecolors='k' )
    ax.scatter ( np.nan, np.nan, marker="o", s=circ_size_factor * 1 + 100, facecolors='none', edgecolors='k' )
    text = ["0", "0.5", "1"]
    ax.legend ( text )

    for i in range ( data_df.shape[1] ):
        for j in range ( data_df.shape[0] ):
            if data_df[j, i] != 0:
                timescale = map_timescale ( cyclep_vec[i] )
                color = map_color ( timescale )
                ax.scatter ( i, j, marker="o", s=circ_size_factor * data_df[j, i] ** 2 + 100, c=color )

    ultr_map = [i for i in range(len(cyclep_vec)) if cyclep_vec[i] < 0.8]
    id_max_ultr = np.argmax(cyclep_vec[ultr_map])

    id_max_circ = np.where(cyclep_vec == 1)[0]
    # ultradian
    ax.axvspan ( 0 - 0.4, id_max_ultr + 0.5, color=map_color("ultradian"), alpha=0.2, lw=0 )
    # circadian
    ax.axvspan( id_max_circ - 0.4, id_max_circ + 0.4, color=map_color("circadian"), alpha=0.2, lw=0 )
    # multidien
    ax.axvspan (id_max_circ + 0.5, np.argmax(cyclep_vec), color=map_color("multidien"), alpha=0.2, lw=0 )

    ax.set_yticks ( range ( data_df.shape[0] ) )
    ax.set_yticklabels ( patients_vec )
    x_ticks = [i for i in range ( 0, len ( cyclep_vec ) ) if data_df[:, i].any () == True]
    ax.set_xticks(x_ticks)
    x_labels = [format_global(cycle) for cycle in cyclep_vec[x_ticks]]
    ax.set_xticklabels ( x_labels , rotation=90 )

    return ax

def plot_outcome_patient_SOZ (roi_is_soz, measure_patient, roi_names, d_rs, ylim = None,
                      outcome_clrs=None, ax=None, size=7,
                      plot_type='beeswarm', overlay_violin=True,
                      violin_tint=0.9, ylabel = None):

    if outcome_clrs is None:
        cmap = plt.get_cmap ('Set1', 8 )
        outcome_clrs = np.zeros ( (2, 3) )
        outcome_clrs[0, :] = cmap ( 1 )[:3]
        outcome_clrs[1, :] = cmap ( 0 )[:3]

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    if ylim is not None:
        ax.set_ylim ( ylim )

    # make dataframe
    dct = {'is_soz': roi_is_soz, 'measure_patient': measure_patient, 'roi_names': roi_names}
    df = pd.DataFrame ( dct )

    if plot_type == 'beeswarm':
        # violin
        if overlay_violin:
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, x='is_soz', y='measure_patient',
                             palette=v_clrs, ax=ax, saturation=1,
                             inner='quartiles', cut = 0)

        # beeswarm
        sns.swarmplot ( data=df, x='is_soz', y='measure_patient', hue='is_soz',
                        size=size,
                        palette=outcome_clrs, ax=ax )

    elif plot_type == 'violin':
        # lighten clrs
        v_clrs = outcome_clrs * 255
        v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
        v_clrs = v_clrs / 255

        sns.violinplot ( data=df, x='is_soz', y='measure_patient',
                         palette=v_clrs, ax=ax, saturation=1,
                         inner='stick', cut = 0)
    ax.get_legend ().remove ()
    ax.set_xticklabels ( ['non-SOZ', 'SOZ'] )

    ax.set_title (
        f' AUC = {round ( d_rs, 2 )} \n ROIs = {len(roi_names)}')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel ( "$D_{RS}$")

    ax.set_xlabel ( '' )

    return ax

def dotplot_Drs_fb(data_df, is_FDR = False, size = 7, ax = None):

    # colors = {"ultradian": "#D2691E", "circadian": "#DBB40C", "multidien": "#6E750E"}
    colors = {"ultradian": "#572c92", "circadian": "#05712f", "multidien": "#ad3803", 0: "#C5C9C7"}

    x = data_df["cyclep_est"].to_numpy ().flatten ()
    y = data_df["id_fb"].to_numpy ().flatten ()
    y_labels = data_df["freq_band"].to_numpy ().flatten ()
    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 9) )

    if is_FDR == True:
        # make scatterplot
        ax.scatter ( x, y, c = data_df["color_map_qval"].apply(lambda x: colors[x]), s = size)
    else:
        # make scatterplot
        ax.scatter ( x, y, c=data_df["color_map_pval"].apply ( lambda x: colors[x] ), s = size )

    ax.set_xscale ( "log" )

    # set xticks
    x_ticks = np.unique(x).tolist()
    ax.set_xticks ( x_ticks )  # set x ticks
    ax.set_xticklabels ( ax.get_xticks (), rotation=90 )
    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    # set yticks
    ax.set_yticks(y)
    ax.set_yticklabels ( y_labels, rotation=0 )  # y tick labels
    ax.invert_yaxis()

    ax.set ( ylabel="subjects" )
    ax.set ( xlabel="cycle period" )

    return ax

def plot_outcome_auc_metadata (pnt_meas, pnt_metavar, ylim=[0, 1], meas_name='$D_{RS}$',
                      outcome_clrs=None, meas_ref=0.5, ax=None, size=7,
                      plot_type='beeswarm', p_side='less', overlay_violin=True,
                      violin_tint=0.9):

    if outcome_clrs is None:
        cmap = plt.get_cmap( 'Set1', 8 )
        outcome_clrs = np.zeros ( (2, 3) )
        outcome_clrs[0, :] = cmap ( 1 )[:3]
        outcome_clrs[1, :] = cmap ( 0 )[:3]

    # compute AUC
    if pnt_meas.shape[0] > 0:
        auc = sk_metrics.roc_auc_score ( pnt_metavar, pnt_meas )
        _, p = scipy.stats.ranksums ( pnt_meas[pnt_metavar == 0],
                                      pnt_meas[pnt_metavar == 1],
                                      alternative=p_side )
    else:
        auc = np.nan
        p = np.nan

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    ax.set_ylim ( ylim )
    ax.axhline ( meas_ref, color=(0.5, 0.5, 0.5) , linestyle ="--")
    if len ( pnt_meas ) > 0:

        # make dataframe
        dct = {'pnt_var': pnt_metavar, 'pnt_meas': pnt_meas}
        df = pd.DataFrame ( dct )
        if plot_type == 'beeswarm':

            # violin
            if overlay_violin:
                # lighten clrs
                v_clrs = outcome_clrs * 255
                v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
                v_clrs = v_clrs / 255

                sns.violinplot ( data=df, x='pnt_var', y='pnt_meas',
                                 palette=v_clrs, ax=ax, saturation=1,
                                 inner='quartiles', cut=0 ) # CAN USE cut = 0.5 AS WELL

                # beeswarm
            sns.swarmplot ( data=df, x='pnt_var', y='pnt_meas', hue='pnt_var',
                            size=size,
                            palette=outcome_clrs, ax=ax )

        elif plot_type == 'violin':
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, x='pnt_var', y='pnt_meas',
                             palette=v_clrs, ax=ax, saturation=1,
                             inner='stick', cut=0 )
        ax.get_legend ().remove ()

    ax.set_title (
        f'{meas_name} \n AUC = {round ( auc, 3 )}, p = {round ( p, 5 )} ({p_side}) \n n = {len ( pnt_meas )}' )
    ax.set_ylabel ( "$D_{RS}$" )

    return ax, auc, p

def dotplot_Drs_fb_size(data_df, is_FDR = False, size_factor = 100, ax = None):

    # colors = {"ultradian": "#D2691E", "circadian": "#DBB40C", "multidien": "#6E750E"}
    colors = {"ultradian": "#572c92", "circadian": "#05712f", "multidien": "#ad3803", 0: "#C5C9C7"}

    x = data_df["cyclep_est"].to_numpy ().flatten ()
    y = data_df["id_fb"].to_numpy ().flatten ()
    s = data_df["drs_median"].to_numpy().flatten()
    y_labels = data_df["freq_band"].to_numpy ().flatten ()
    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 9) )

    if is_FDR == True:
        # make scatterplot
        ax.scatter ( x, y, c = data_df["color_map_qval"].apply(lambda x: colors[x]), s = s*size_factor)
    else:
        # make scatterplot
        ax.scatter ( x, y, c=data_df["color_map_pval"].apply ( lambda x: colors[x] ), s = s*size_factor)

    ax.set_xscale ( "log" )

    # set xticks
    x_ticks = np.unique(x).tolist()
    ax.set_xticks ( x_ticks )  # set x ticks
    ax.set_xticklabels ( ax.get_xticks (), rotation=90 )
    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    # set yticks
    ax.set_yticks(y)
    ax.set_yticklabels ( y_labels, rotation=0 )  # y tick labels
    ax.invert_yaxis()

    ax.set ( ylabel="subjects" )
    ax.set ( xlabel="cycle period" )

    return ax


def violinplot_circ_pathology (data_df, ylim=[0, 1], x_var=None,
                     outcome_clrs=None, meas_ref=0.5, ax=None, size=7,
                     plot_type='beeswarm', overlay_violin=True,
                     violin_tint=0.9):
    if outcome_clrs is None:
        outcome_clrs = "#C5C9C7"

    clr_rgb = plt_colors.hex2color ( outcome_clrs )

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    ax.set_ylim ( ylim )
    ax.axhline ( meas_ref, color=(0.5, 0.5, 0.5), linestyle="--" )

    if plot_type == 'beeswarm':

        # violin
        if overlay_violin:
            # lighten clrs
            v_clrs = np.array ( [cc * 255 for cc in clr_rgb] )
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=data_df, y='drs', x=x_var,
                             color=v_clrs, ax=ax, saturation=1,
                             inner='quartiles', cut=0 )  # CAN USE cut = 0.5 AS WELL

        # beeswarm
        sns.swarmplot ( data=data_df, y='drs', x=x_var,
                        color=outcome_clrs,
                        size=size, ax=ax )

    ax.set_title ("Distribution of Drs by {}".format(x_var))
    ax.set_ylabel ( '$D_{RS}$' )
    ax.set_xlabel ( "{}".format(x_var) )

    if x_var == "cyclep_est":
        x_ticks_l = list ( data_df["cyclep_est"].drop_duplicates ().sort_values ( ascending=True ).to_numpy () )
        x_ticks = ax.get_xticks ()
        ax.set_xticks ( x_ticks )
        x_tick_labels = [format_global ( xt ) for xt in x_ticks_l]
        ax.set_xticklabels ( x_tick_labels )
   
    return ax


def scatterplot_corr (pnt_meas, pnt_var, ylim=[0, 1],
                               outcome_clrs=None, meas_ref=0.5, ax=None, size=7):
    if outcome_clrs is None:
        outcome_clrs = "#C5C9C7"

    clr_rgb = plt_colors.hex2color ( outcome_clrs )

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    if ylim is not None:
        ax.set_ylim ( ylim )

    ax.axhline ( meas_ref, color=(0.5, 0.5, 0.5), linestyle="--" )

    data_df = pd.DataFrame({"Drs": pnt_meas, "pnt_var": pnt_var})
    sns.scatterplot ( data=data_df, y='Drs', x=pnt_var, s = size,
                     color=clr_rgb, ax=ax)

    # Pearson correlation
    corr, p = scipy.stats.pearsonr(pnt_meas, pnt_var)

    ax.set_title ( "pearson corr: {}, p-value: {}".format ( corr, p ) )
    ax.set_ylabel ( '$D_{RS}$' )

    return ax

def dotplot_Drs_metadata(data_df, x,y, s, is_FDR = False, size_factor = 100, ax = None):

    y_labels = data_df["freq_band"].to_numpy ().flatten ()
    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 9) )

    if is_FDR == True:
        # make scatterplot
        p_scatter = ax.scatter (x, y, c = s, cmap = "viridis", vmin = 0, vmax = 1, s = size_factor)
    else:
        # make scatterplot
        p_scatter = ax.scatter ( x, y, c=s, cmap = "viridis", vmin = 0, vmax = 1, s = size_factor)

    ax.set_xscale ( "log" )
    plt.colorbar ( p_scatter )

    # set xticks
    x_ticks = np.unique(x).tolist()
    ax.set_xticks ( x_ticks )  # set x ticks
    ax.set_xticklabels ( ax.get_xticks (), rotation=90 )
    ax.xaxis.set_major_formatter ( plt.FuncFormatter ( format_func ) )

    # set yticks
    ax.set_yticks(y)
    ax.set_yticklabels ( y_labels, rotation=0 )  # y tick labels
    ax.invert_yaxis()

    ax.set ( ylabel="frequency bands" )
    ax.set ( xlabel="cycle period" )

    return ax

def density_drs_across_all(data_df, clr = "#C5C9C7", p_side = "less", ax = None):

    drs = data_df["drs"].to_numpy()

    # compute AUC
    if drs.shape[0] > 0:
        diff = 0.5 - drs
        _, p = scipy.stats.wilcoxon ( diff, alternative=p_side,
                                      mode='exact' )  # less = 0.5<patient D_RS
        # H0: Drs=0.5 with alternative that
        # H1: Drs>0.5
    else:
        p = np.nan

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 9) )

    my_bins = np.arange(0,1.1,0.1)
    alph = 0.6
    ax.hist ( drs, bins=my_bins,
               facecolor=clr, alpha=alph )

    # line representing the median value
    median_drs = np.median ( data_df["drs"] )
    ax.axvline ( median_drs, color=(0.5, 0.5, 0.5), linestyle="-", linewidth=6 )
    ax.set_title (
        f'H0:Drs = 0.5, H1:Drs > 0.5, p_value: {round ( p, 5 )} ({p_side}) \n Drs distribution with median Drs = {round(median_drs, 3)}' )

    ax.set_xlabel('$D_{RS}$')
    ax.set_xlim(0,1)
    ax.set_xticks((0,0.5,1))

    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


    return ax

def dotplot_Drs_metadata_fixed(data_df, x,y, s, is_FDR = False, size_factor = 100, ax = None):


    y_labels = data_df["freq_band"].to_numpy ().flatten ()
    x_labels = ['1h-3h', '3h-6h', '6h-9h', '9h-12h', '12h-19h', '19h-1.3d']
    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(7, 9) )

    if is_FDR == True:
        # make scatterplot
        p_scatter = ax.scatter (x, y, c = s, cmap = "RdBu", vmin = -1, vmax = 1, s = size_factor)
    else:
        # make scatterplot
        p_scatter = ax.scatter ( x, y, c=s, cmap = "RdBu", vmin = -1, vmax = 1, s = size_factor)

    plt.colorbar ( p_scatter )

    # set xticks
    x_ticks = np.unique(x).tolist()
    ax.set_xticks ( x_ticks )  # set x ticks
    ax.set_xticklabels ( x_labels, rotation=90 )

    # set yticks
    ax.set_yticks(y)
    ax.set_yticklabels ( y_labels, rotation=0 )  # y tick labels
    ax.invert_yaxis()

    ax.set ( ylabel="frequency bands" )
    ax.set ( xlabel="cycle period" )

    return ax


