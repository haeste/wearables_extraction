import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

def heatmap_cycle_data (roi_cycle_data, freq_band, roi_names, ax=None,
                           center=0, cmap_y0 = 0, cmap='RdBu',
                           cbar_kws=None, vmin=None, vmax=None):
    '''
    Plot a heatmap of ROI data (ROIs x time heatmap). Time units are
    hours. Will also sort ROIs by resected/spared and divide categories with
    a horizontal line if labels are provided.

    There must be an integer number of time windows in an hour.

    Parameters
    ----------
    roi_cycle_data : 2D numpy array, float
        ROI data, size number of ROIs x # variables ([Delta, Theta, Alpha, Beta, Gamma, ratio alpha/delta]).
    freq_band : 1D numpy array, float
        Variable names corresponding to roi_cycle_data columns.
    roi_names : list of strings
        List of ROI names corresponding to roi_data rows.
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

    # create new figure if none pass to function
    if ax is None:
        fig, ax = plt.subplots ()

     # axes for colorbar
    cax = inset_axes ( ax, width='5%', height='100%',
                       loc=3,
                       bbox_to_anchor=(1, cmap_y0, 1, 1),
                       bbox_transform=ax.transAxes )
    # heatmap
    sns.heatmap ( roi_cycle_data, center=center, cmap=cmap,
                  yticklabels=roi_names, xticklabels = freq_band, ax=ax,
                  cbar_ax=cax, rasterized=True,
                  cbar_kws=cbar_kws, vmin=vmin, vmax=vmax )

    # labels
    ax.set_xlabel ( 'Variables' )  # x axis label
    ax.set_ylabel ( 'ROI' )  # y axis label

    # frame
    for _, spine in ax.spines.items ():
        spine.set_visible ( True )

    # return axis
    return ax, cax