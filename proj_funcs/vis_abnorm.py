"""

TODO: Check the commented code for modifying it and getting the appropriate plots; displaying all channels with offset

Some functions are modified from Gabrielle Schroeder's functions

"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import scipy.stats
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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



def plot_outcome_auc_lobe_surgery (pnt_ilae, pnt_meas, pnt_lobe_surgery, ilae_thresh=3, ylim=[0, 1], meas_name='$D_{RS}$',
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
        cmap = plt.get_cmap ( 'Set1', 8 )
        outcome_clrs = np.zeros ( (2, 3) )
        outcome_clrs[0, :] = cmap ( 1 )[:3]
        outcome_clrs[1, :] = cmap ( 0 )[:3]
        # colours for lobe_surgery variable
        outcome_clrs_lobe = np.zeros ( (2, 3) )
        outcome_clrs_lobe[0, :] = cmap ( 2 )[:3]
        outcome_clrs_lobe[1, :] = cmap ( 3 )[:3]

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
        dct = {'bad_outcome': bad_outcome, 'pnt_meas': pnt_meas, "pnt_lobe_surgery": pnt_lobe_surgery }
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
            sns.swarmplot ( data=df, x='bad_outcome', y='pnt_meas', hue='pnt_lobe_surgery',
                            size=size,
                            palette=outcome_clrs_lobe, ax=ax )

        elif plot_type == 'violin':
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, x='bad_outcome', y='pnt_meas',
                             palette=v_clrs, ax=ax, saturation=1,
                             inner='stick', cut=0 )
        # ax.get_legend ().remove ()
        ax.get_legend()
        ax.set_xticklabels ( ['good', f'bad (ILAE {ilae_thresh}+)'] )

    ax.set_title (
        f'{meas_name} \n AUC = {round ( auc, 3 )}, p = {round ( p, 5 )} ({p_side}) \n n = {len ( pnt_meas )}' )
    ax.set_ylabel ( "$D_{RS}$" )
    ax.set_xlabel ( 'surgical outcome' )

    return ax, auc, p

def plot_outcome_lobe_surgery (pnt_meas, pnt_lobe_surgery, ylim=[0, 1], meas_name='$D_{RS}$',
                      outcome_clrs=None, meas_ref=0.5, ax=None, size=7,
                      plot_type='beeswarm', p_side='less', overlay_violin=True,
                      violin_tint=0.9):

    # first remove any nans
    is_nan = np.isnan ( pnt_meas )
    pnt_meas = pnt_meas[np.invert ( is_nan )]
    print ( f'removed {np.sum ( is_nan )} patients due to NaN measure' )

    if outcome_clrs is None:
        cmap = plt.get_cmap ( 'Set1', 8 )
        # colours for lobe_surgery variable
        outcome_clrs_lobe = np.zeros ( (2, 3) )
        outcome_clrs_lobe[0, :] = cmap ( 2 )[:3]
        outcome_clrs_lobe[1, :] = cmap ( 3 )[:3]

    # compute AUC
    indx_0 = [ind for ind,d in enumerate(pnt_lobe_surgery) if d == 0]
    indx_1 = [ind for ind,d in enumerate(pnt_lobe_surgery) if d == 1]
    if  pnt_meas.shape[0] > 0:
        auc = sk_metrics.roc_auc_score ( pnt_lobe_surgery, pnt_meas )
        _, p = scipy.stats.ranksums ( pnt_meas[indx_0],
                                      pnt_meas[indx_1],
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
        dct = {'pnt_lobe_surgery': pnt_lobe_surgery, 'pnt_meas': pnt_meas}
        df = pd.DataFrame ( dct )
        if plot_type == 'beeswarm':

            # violin
            if overlay_violin:
                # lighten clrs
                v_clrs = outcome_clrs_lobe * 255
                v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
                v_clrs = v_clrs / 255

                sns.violinplot ( data=df, x='pnt_lobe_surgery', y='pnt_meas',
                                 palette=v_clrs, ax=ax, saturation=1,
                                 inner='quartiles', cut=0 ) # CAN USE cut = 0.5 AS WELL

                # beeswarm
            sns.swarmplot ( data=df, x='pnt_lobe_surgery', y='pnt_meas', hue='pnt_lobe_surgery',
                            size=size,
                            palette=outcome_clrs_lobe, ax=ax )

        elif plot_type == 'violin':
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, x='pnt_lobe_surgery', y='pnt_meas',
                             palette=v_clrs, ax=ax, saturation=1,
                             inner='stick', cut=0 )
        ax.get_legend ().remove ()
        ax.set_xticklabels ( ['eTLE', f'TLE'] )

    ax.set_title (
        f'{meas_name} \n AUC = {round ( auc, 3 )}, p = {round ( p, 5 )} ({p_side}) \n n = {len ( pnt_meas )}' )
    ax.set_ylabel ( "$D_{RS}$" )
    ax.set_xlabel ( 'epilepsy syndrome' )

    return ax, auc, p


def plot_outcome_patient (roi_is_resect, measure_patient, roi_names, d_rs, ilae_patient, ylim = None, ilae_thresh=3,
                      outcome_clrs=None, ax=None, size=7,
                      plot_type='beeswarm', overlay_violin=True,
                      violin_tint=0.9, ylabel = None):
    # ilae thresh = min ilae that is bad outcome

    bad_outcome = ilae_patient >= ilae_thresh
    if bad_outcome == True:
        surgery_outcome = "bad"
    else:
        surgery_outcome = "good"

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
    dct = {'is_resect': roi_is_resect, 'measure_patient': measure_patient, 'roi_names': roi_names}
    df = pd.DataFrame ( dct )

    if plot_type == 'beeswarm':
        # violin
        if overlay_violin:
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, x='is_resect', y='measure_patient',
                             palette=v_clrs, ax=ax, saturation=1,
                             inner='quartiles', cut = 0)

        # beeswarm
        sns.swarmplot ( data=df, x='is_resect', y='measure_patient', hue='is_resect',
                        size=size,
                        palette=outcome_clrs, ax=ax )

    elif plot_type == 'violin':
        # lighten clrs
        v_clrs = outcome_clrs * 255
        v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
        v_clrs = v_clrs / 255

        sns.violinplot ( data=df, x='is_resect', y='measure_patient',
                         palette=v_clrs, ax=ax, saturation=1,
                         inner='stick', cut = 0)
    ax.get_legend ().remove ()
    ax.set_xticklabels ( ['spared', 'resected'] )

    ax.set_title (
        f' Drs = {round ( d_rs, 2 )} ILAE = {ilae_patient} \n'
        f' {surgery_outcome} outcome, bad define (ILAE {ilae_thresh}+) \n # ROIs = {len(roi_names)}')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel ( "$D_{RS}$")

    ax.set_xlabel ( '' )

    return ax

def plot_outcome_patient_drs (pnt_meas, meas_name, d_rs, ylim = None,
                      outcome_clrs=None, ax=None, size=7,
                      plot_type='beeswarm', overlay_violin=True,
                      violin_tint=0.9, ylabel = None):
    # first remove any nans
    is_nan = np.isnan ( pnt_meas )
    pnt_meas = pnt_meas[np.invert ( is_nan )]
    print ( f'removed {np.sum ( is_nan )} patients due to NaN measure' )

    if outcome_clrs is None:
        cmap = plt.get_cmap ( 'Set1', 8 )
        outcome_clrs = np.asarray ( cmap ( 3 )[:3] )

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    if ylim is not None:
        ax.set_ylim ( ylim )


    if len ( pnt_meas ) > 0:

        # make dataframe
        dct = {'pnt_meas': pnt_meas}
        df = pd.DataFrame ( dct )
        if plot_type == 'beeswarm':
            # violin
            if overlay_violin:
                # lighten clrs
                v_clrs = outcome_clrs * 255
                v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
                v_clrs = v_clrs / 255

                sns.violinplot ( data=df,  y='pnt_meas',
                                 color=v_clrs, ax=ax, saturation=1,
                                 inner='quartiles', cut = 0)

            # beeswarm
            sns.swarmplot ( data=df, y='pnt_meas',
                            size=size,
                            color=outcome_clrs, ax=ax )

        elif plot_type == 'violin':
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, y='pnt_meas',
                             color=v_clrs, ax=ax, saturation=1,
                             inner='stick', cut = 0)

    ax.set_title (
        f'Drs = {round ( d_rs, 2 )} \n '
        f'{meas_name} \n n = {len ( dct )}' )
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel ( meas_name)

    ax.set_xlabel ( 'values' )

    return ax

def plot_outcome_drs (pnt_meas, meas_name, all_patients = None, ylim=[0, 1],
                      outcome_clrs=None, patients_clrs = None, meas_ref=0.5, ax=None, size=7,
                      plot_type='beeswarm', p_side='less', overlay_violin=True,
                      violin_tint=0.9):
    # ilae thresh = min ilae that is bad outcome

    # first remove any nans
    is_nan = np.isnan ( pnt_meas )
    pnt_meas = pnt_meas[np.invert ( is_nan )]
    print ( f'removed {np.sum ( is_nan )} patients due to NaN measure' )

    if outcome_clrs is None:
        cmap = plt.get_cmap ( 'Set1', 8 )
        outcome_clrs = np.asarray(cmap ( 3 )[:3])
    if patients_clrs is None:
        pat_colors = ["#a9a9a9", "#2f4f4f", "#556b2f", "#a0522d", "#191970", "#006400", "#8b0000",
                      "#808000", "#3cb371", "#bdb76b", "#008b8b", "#4682b4", "#00008b", "#32cd32", "#daa520",
                      "#800080", "#b03060", "#ff4500", "#ff8c00", "#ffff00", "#00ff00", "#00fa9a", "#dc143c",
                      "#00ffff", "#00bfff", "#f4a460", "#0000ff", "#a020f0", "#adff2f", "#da70d6", "#ff00ff",
                      "#1e90ff", "#fa8072", "#dda0dd", "#ff1493", "#7b68ee", "#afeeee", "#ffdab9", "#ffb6c1"]
        # make a color palette using those custom colors
        patients_clrs = sns.color_palette(pat_colors)
    # compute AUC
    if  pnt_meas.shape[0] > 0:
        diff = 0.5 - pnt_meas
        _, p = scipy.stats.wilcoxon ( diff, alternative=p_side,
                                          mode='exact' )  # less = 0.5<patient D_RSs
    else:
        p = np.nan

    # PLOT
    if ax is None:
        fig, ax = plt.subplots ( 1, 1, figsize=(4, 5) )

    ax.set_ylim ( ylim )
    ax.axhline ( meas_ref, color=(0.5, 0.5, 0.5) , linestyle ="--")
    if len ( pnt_meas ) > 0:

        # make dataframe
        dct = {'pnt_meas': pnt_meas, "all_patients": all_patients}
        df = pd.DataFrame ( dct )
        if plot_type == 'beeswarm':

            # violin
            if overlay_violin:
                # lighten clrs
                v_clrs = outcome_clrs * 255
                v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
                v_clrs = v_clrs / 255


                sns.violinplot ( data=df, y='pnt_meas',
                                 color=v_clrs, ax=ax, saturation=1,
                                 inner='quartiles', cut=0 ) # CAN USE cut = 0.5 AS WELL

            # beeswarm
            if all_patients is not None:
                sns.swarmplot ( data=df, y='pnt_meas', hue = "all_patients",
                                palette = patients_clrs,
                                size=size, ax=ax )
            else:
                sns.swarmplot ( data=df, y='pnt_meas',
                                size=size,
                                color=outcome_clrs, ax=ax )

        elif plot_type == 'violin':
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, y='pnt_meas',
                             color=v_clrs, ax=ax, saturation=1,
                             inner='stick', cut=0 )
    if all_patients is not None:
        ax.legend ( bbox_to_anchor=(1.04, 1), borderaxespad = 0, ncol=2 )

    ax.set_title (
        f'{meas_name} \n Drs > 0.5: {round ( p, 5 )} ({p_side}) \n n = {len ( pnt_meas )}' )
    ax.set_ylabel ( '$D_{RS}$')
    ax.set_xlabel ( 'values' )

    return ax, p


def plot_outcome_gender (pnt_meas, pnt_gender, ylim=[0, 1], meas_name='$D_{RS}$',
                      outcome_clrs=None, meas_ref=0.5, ax=None, size=7,
                      plot_type='beeswarm', p_side='less', overlay_violin=True,
                      violin_tint=0.9):

    # first remove any nans
    is_nan = np.isnan ( pnt_meas )
    pnt_meas = pnt_meas[np.invert ( is_nan )]
    print ( f'removed {np.sum ( is_nan )} patients due to NaN measure' )

    if outcome_clrs is None:
        cmap = plt.get_cmap ( 'Set1', 8 )
        # colours for lobe_surgery variable
        outcome_clrs_lobe = np.zeros ( (2, 3) )
        outcome_clrs_lobe[0, :] = cmap ( 2 )[:3]
        outcome_clrs_lobe[1, :] = cmap ( 3 )[:3]

    # compute AUC
    indx_0 = [ind for ind,d in enumerate(pnt_gender) if d == 0]
    indx_1 = [ind for ind,d in enumerate(pnt_gender) if d == 1]
    if  pnt_meas.shape[0] > 0:
        auc = sk_metrics.roc_auc_score ( pnt_gender, pnt_meas )
        _, p = scipy.stats.ranksums ( pnt_meas[indx_0],
                                      pnt_meas[indx_1],
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
        dct = {'pnt_gender': pnt_gender, 'pnt_meas': pnt_meas}
        df = pd.DataFrame ( dct )
        if plot_type == 'beeswarm':

            # violin
            if overlay_violin:
                # lighten clrs
                v_clrs = outcome_clrs_lobe * 255
                v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
                v_clrs = v_clrs / 255

                sns.violinplot ( data=df, x='pnt_gender', y='pnt_meas',
                                 palette=v_clrs, ax=ax, saturation=1,
                                 inner='quartiles', cut=0 ) # CAN USE cut = 0.5 AS WELL

                # beeswarm
            sns.swarmplot ( data=df, x='pnt_gender', y='pnt_meas', hue='pnt_gender',
                            size=size,
                            palette=outcome_clrs_lobe, ax=ax )

        elif plot_type == 'violin':
            # lighten clrs
            v_clrs = outcome_clrs * 255
            v_clrs = ((255 - v_clrs) * violin_tint) + v_clrs
            v_clrs = v_clrs / 255

            sns.violinplot ( data=df, x='pnt_gender', y='pnt_meas',
                             palette=v_clrs, ax=ax, saturation=1,
                             inner='stick', cut=0 )
        ax.get_legend ().remove ()
        ax.set_xticklabels ( ['M', 'F'] )

    ax.set_title (
        f'{meas_name} \n AUC = {round ( auc, 3 )}, p = {round ( p, 5 )} ({p_side}) \n n = {len ( pnt_meas )}' )
    ax.set_ylabel ( "$D_{RS}$" )
    ax.set_xlabel ( 'gender' )

    return ax, auc, p


def heatmap_data (roi_data, t_days, roi_names, roi_is_resect=None, ax=None,
                           center=0, cmap_y0 = 0, cmap='RdBu', resect_line_clr=(0, 0, 0),
                           cbar_kws=None, vmin=None, vmax=None):
    '''
    Plot a heatmap of ROI data (ROIs x time heatmap). Time units are
    hours. Will also sort ROIs by resected/spared and divide categories with
    a horizontal line if labels are provided.

    There must be an integer number of time windows in an hour.

    Parameters
    ----------
    roi_data : 2D numpy array, float
        ROI data, size number of ROIs x time of time windows.
    t_days : 1D numpy array, float
        Times corresponding to roi_data columns, in days.
    roi_names : list of strings
        List of ROI names corresponding to roi_data rows.
    roi_is_resect : 1D numpy array, boolean, optional
        Whether each ROI was resected. The default is None. If None, ROIs are
        not sorted by spared/resected in the heatmap.
    ax : matplotlib axes, optional
        Axes in which to plot heatmap. The default is None. If None, new figure
        is generated.
    center : int, float, or None, optional
        Value to put at center of heatmap colorbar. The default is 0. If None,
        default colorbar limits and centering are used.
    cmap_y0: the position y0 of the cmap within the bbox_to_anchor variable
    cmap : str, optional
        Heatmap colormap. The default is 'RdBu'.
    resect_line_clr : matplotlib color (e.g., tuple), optional
        Color of line separating resected and spared ROIs.
        The default is (0,0,0).

    Returns
    -------
    ax : matplotlib axes
        Axes in which heatmap is plotted.
    idx : 1D numpy array, int
        indices for sorting ROIs

    '''
    # number of ROIs
    n_roi, n_win = roi_data.shape

    # create new figure if none pass to function
    if ax is None:
        fig, ax = plt.subplots ()

    # indices for sorting ROIs by resected and spared, if specified
    # resected ROIs will be at top of plot
    if roi_is_resect is not None:
        idx = np.argsort ( np.invert ( roi_is_resect ) )
    else:
        idx = np.arange ( n_roi )

    # sort ROIs
    sorted_roi = [roi_names[i] for i in idx]

    # axes for colorbar
    cax = inset_axes ( ax, width='5%', height='100%',
                       loc=3,
                       bbox_to_anchor=(1, cmap_y0, 1, 1),
                       bbox_transform=ax.transAxes )
    # heatmap
    sns.heatmap ( roi_data[idx, :], center=center, cmap=cmap,
                  yticklabels=sorted_roi, ax=ax,
                  cbar_ax=cax, rasterized=True,
                  cbar_kws=cbar_kws, vmin=vmin, vmax=vmax )

    # labels
    win_per_hr = 1 / ((t_days[1] - t_days[0]) * 24)  # number of windows per hr
    if win_per_hr.is_integer ():
        win_per_hr = int ( win_per_hr )
    else:
        raise ValueError ( 'There must be an integer number of windows per hour' )

    tick_x = np.arange ( 0, n_win, win_per_hr * 12 )  # make tick mark every 12 hrs
    ax.set_xticks ( tick_x )  # set x ticks
    ax.set_xticklabels ( t_days[tick_x], rotation=0 )  # x tick labels
    ax.set_xlabel ( 'time (days)' )  # x axis label
    ax.set_ylabel ( 'ROI' )  # y axis label

    # frame
    for _, spine in ax.spines.items ():
        spine.set_visible ( True )

    # mark resected vs spared, if specified
    if roi_is_resect is not None:
        n_resect = np.sum ( roi_is_resect )
        ax.axhline ( n_resect, color=resect_line_clr )

    # return axis
    return ax, idx, cax
#



def plot_allcycles_allP(data_df, xlim=[0, 1],
                      outcome_clrs=None, meas_ref=0.5, ax=None, size=7, p_side='less'):

    if outcome_clrs is None:
        colors = {"ultradian": "#D2691E", "circadian": "#DBB40C", "multidien": "#6E750E"}

    # get variable for measure
    meas_var = data_df["Drs"].to_numpy().flatten()
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

    ax.set_xlim ( xlim )
    ax.axvline ( meas_ref, color=(0.5, 0.5, 0.5), linestyle="--" )

    # beeswarm
    sns.swarmplot ( data=data_df, x='Drs', hue="what_peak",
                    palette=colors,
                    size=size, ax=ax )

    ax.set_title (
        f'H0:Drs = 0.5, H1:Drs > 0.5: {round ( p, 5 )} ({p_side})' )
    ax.set_xlabel ( '$D_{RS}$' )
    ax.set_ylabel ( 'timescales' )

    return ax, p



# def plot_mean_abnorm (t_days, mean_ab_r, mean_ab_s, ax=None,
#                       clr_r=(195 / 255, 136 / 255, 19 / 255), clr_s=(35 / 255, 135 / 255, 129 / 255)):
#     # number of time points (windows)
#     n_win = len ( t_days )
#
#     # make new figure if axes not supplied
#     if ax is None:
#         fig, ax = plt.subplots ()
#
#     # plot
#     ax.plot ( t_days, mean_ab_s, linewidth=0.5, color=clr_s, label='spared' )  # spared
#     ax.plot ( t_days, mean_ab_r, linewidth=0.5, color=clr_r, label='resected' )  # resected
#     ax.legend ()
#
#     # limits
#     ax.set_xlim ( min ( t_days ), max ( t_days ) )
#     if np.nanmin ( mean_ab_s ) >= 0 and np.nanmin ( mean_ab_r ) >= 0:
#         ax.set_ylim ( 0, None )
#
#     # labels
#     win_per_hr = 1 / ((t_days[1] - t_days[0]) * 24)  # number of windows per hr
#     if win_per_hr.is_integer ():
#         win_per_hr = int ( win_per_hr )
#     else:
#         raise ValueError ( 'There must be an integer number of windows per hour' )
#
#     ax.set_ylabel ( 'mean abnormality' )
#     ax.set_title ( 'mean abnormality' )
#     tick_x = np.arange ( 0, n_win, win_per_hr * 12 )  # make tick mark every 12 hrs
#     ax.set_xticks ( t_days[tick_x] )  # set x ticks
#     ax.set_xlabel ( 'time (days)' )  # x axis label
#
#     return ax
#
#
# def plot_abnorm_contribution (t_days, ab_cont, ax=None, roi_is_resect=None,
#                               clr=(136 / 255, 33 / 255, 17 / 255)):
#     # number of time points (windows)
#     n_win = len ( t_days )
#
#     # make new figure if axes not supplied
#     if ax is None:
#         fig, ax = plt.subplots ()
#
#     # plot
#     ax.plot ( t_days, ab_cont, linewidth=0.5, color=clr )
#
#     # add proportion of ROIs resected if resection labels provided
#     if roi_is_resect is not None:
#         prop_resect = np.sum ( roi_is_resect ) / len ( roi_is_resect )
#
#         # horizontal line
#         ax.axhline ( prop_resect, color=(0, 0, 0), linewidth=0.25 )
#
#     # limits
#     ax.set_xlim ( min ( t_days ), max ( t_days ) )
#     ax.set_ylim ( 0, np.nanmax ( ab_cont ) )
#
#     # labels
#     win_per_hr = 1 / ((t_days[1] - t_days[0]) * 24)  # number of windows per hr
#     if win_per_hr.is_integer ():
#         win_per_hr = int ( win_per_hr )
#     else:
#         raise ValueError ( 'There must be an integer number of windows per hour' )
#
#     ax.set_title ( 'abnormality contribution of resected' )
#     ax.set_ylabel ( 'proportion abnormality' )
#     tick_x = np.arange ( 0, n_win, win_per_hr * 12 )  # make tick mark every 12 hrs
#     ax.set_xticks ( t_days[tick_x] )  # set x ticks
#     ax.set_xlabel ( 'time (days)' )  # x axis label
#
#     return ax
#
#
# def plot_NMF_results (W, H, W_meas, roi_data, t_days, roi_names, roi_is_resect=None,
#                       center=None, cmap='viridis', resect_line_clr=(0.9, 0, 0),
#                       W_meas_name='$D_{RS}$', W_meas_ylim=[0, 1], W_meas_ref=0.5):
#     # start fig
#     fig, axs = plt.subplots ( 3, 2, figsize=(12, 12),
#                               gridspec_kw={'height_ratios': [3, 1, 3], 'width_ratios': [1, 5]} )
#
#     # plot abnormalities
#     fig.tight_layout ( h_pad=5, w_pad=15, rect=[0.02, 0.03, 0.98,
#                                                 0.95] )  # need to call before heatmap since uses inset axes to set colorbar
#
#     axs[0, 1].set_title ( 'abnormalities' )
#     heatmap_abnormalities ( roi_data, t_days, roi_names,
#                             roi_is_resect=roi_is_resect, ax=axs[0, 1], center=center, cmap=cmap,
#                             resect_line_clr=resect_line_clr )
#
#     # plot W
#     axs[0, 0].set_title ( 'W' )
#     k = W.shape[1]
#     heatmap_abnormalities ( W, t_days, roi_names,
#                             roi_is_resect=roi_is_resect, ax=axs[0, 0], center=center, cmap=cmap,
#                             resect_line_clr=resect_line_clr )
#     # relabel
#     axs[0, 0].set_xticks ( np.arange ( k ) + 0.5 )
#     axs[0, 0].set_xticklabels ( np.arange ( 1, k + 1 ) )
#     axs[0, 0].set_xlabel ( 'basis vector' )
#     axs[0, 0].set_ylabel ( 'ROI' )
#
#     # plot H
#     axs[1, 1].set_title ( 'H' )
#     heatmap_abnormalities ( H, t_days, np.arange ( 1, k + 1 ), ax=axs[1, 1], center=center, cmap='magma' )
#     axs[1, 1].set_ylabel ( 'basis vector' )
#
#     # reconstruction
#     axs[2, 1].set_title ( 'reconstruction (WxH)' )
#     heatmap_abnormalities ( np.matmul ( W, H ), t_days, roi_names,
#                             roi_is_resect=roi_is_resect, ax=axs[2, 1], center=center, cmap=cmap,
#                             resect_line_clr=resect_line_clr )
#
#     # W measure (e.g., D_RS)
#     axs[1, 0].scatter ( np.arange ( k ) + 1, W_meas, s=50 )
#     axs[1, 0].axhline ( W_meas_ref, color=(0.5, 0.5, 0.5) )
#     axs[1, 0].set_xticks ( np.arange ( k ) + 1 )
#     axs[1, 0].set_xlabel ( 'basis vector' )
#     axs[1, 0].set_ylabel ( W_meas_name )
#     axs[1, 0].set_ylim ( W_meas_ylim )
#     axs[1, 0].set_xlim ( 0.5, k + 0.5 )
#     axs[1, 0].set_title ( W_meas_name )
#
#     # turn off remaining axis
#     axs[2, 0].axis ( 'off' )
#
#     return fig, axs
#
#
