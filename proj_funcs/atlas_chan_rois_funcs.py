"""

@Mariellapanag

Created on March 2023
@author: mariella

Functions for computing abnormalities from continuous iEEG data.

Includes
    load_atlas
    load_patient_channel_json

    get_info_from_parc_scale

    match_most_common_channel_set
    keep_consensus_chan
    subset_channel_df
    map_chan2rois
    map_chan2rois_all_chan_sets
    match_final_channel_set

"""

# internal modules
import numpy as np
import scipy.io as sio
import json
import pandas as pd


def load_atlas (atlas_path):
    '''
    Load atlas info from MATLAB workspace (1 parcellation).
    Workspaces that are compatible with this function:
        atlas_scale36.mat
        atlas_scale60.mat
        atlas_scale125.mat
        atlas_scale250.mat
    See MATLAB script atlas_reformat.m for how ATLAS.mat, retains.mat, and
    atlasinfo.mat were reformatted into the above workspaces.

    Parameters
    ----------
    atlas_path : string
        Path to MATLAB workspace with atlas info for one parcellation,
        including the workspace name. MATLAB workspace should contain variables
        roi_names, roi_dists, roi_retain, roi_vol, roi_xyz, and roi_scale.

    Returns
    -------
    atlas_dict : dictionary
        Dictionary containing atlas info - keys should be
            'roi_names': list of ROI names, length # ROI
            'roi_dists': # ROI x # ROI array of distances between ROIs
            'roi_retain': # ROI x 1 boolean array of which ROIs to keep for
                analysis (i.e., not in white matter or ventricles).
            'roi_scale': string, parcellation scale label
            'roi_xyz': # ROI x 3 array of coordinates for each ROI
            'roi_vol': # ROI x 1 array of ROI volumes

    '''

    # load mat file
    atlas_dict = sio.loadmat ( atlas_path )

    # get keys
    atlas_keys = list ( atlas_dict.keys () )

    # remove keys that don't correspond to variables
    # will leave remainder in dictionary form for easy return variable
    rm_keys = [key for key in atlas_keys if key.startswith ( '__' )]
    for key in rm_keys:
        atlas_dict.pop ( key )

    # unpack nested array of ROI names into list
    atlas_dict['roi_names'] = [name[0][0].tolist () for name in atlas_dict['roi_names']]

    # unpack roi_scale string
    atlas_dict['roi_scale'] = atlas_dict['roi_scale'][0]

    return atlas_dict


def load_patient_channel_json (json_path):
    '''
    Load json file with patient's channel info

    Parameters
    ----------
    json_path : string
        Path to json file with channel info, including the file name.
    Returns
    -------
    channel_df : dataframe
        pandas dataframe of channel info.
    '''

    # load json file
    with open ( json_path ) as f_json:
        patient_json = json.load ( f_json )

    # convert to dataframe
    channel_df = pd.DataFrame ( patient_json )

    return channel_df


def get_info_from_parc_scale (parc_scale):
    '''
    Helper function - returns info about parcellation at a given scale so do
    not need to pass that needed info to the function.

    Parameters
    ----------
    parc_scale : string
        Parcellation scale - must be '36', '60', '125', or '250'

    Returns
    -------
    n_roi : int
        Number of ROIs in parcellation (before limit ROIs to retains ROIs).
    parc : int between 0 and 3, inclusive
        Index of parcellation - used for selecting parcellation info in array
        that contains info of all parcellations.

    '''
    all_scales = ['36', '60', '125', '250']
    parc = all_scales.index ( parc_scale )

    all_n_roi = np.array ( (114, 160, 265, 494) )
    n_roi = all_n_roi[parc]

    return n_roi, parc

def match_most_common_channel_set (chan_data):
    '''
    Determines which set of channels is most common in recording (ignoring windows
    with all nan values) and removes data from time windows with different sets
    of channels. Dimensions are not changed - data is removed by turning
    entries to nan.

    Note: only first dimension of chan_data is analysed; assumes other
    dimensions have the same pattern of channels and nan data.

    Parameters
    ----------
    chan_data : 3D numpy array
        Channel data, size # channels x # time windows x # measures (e.g.,
        frequency bands).


    Returns
    -------
    chan_data : 3D numpy array
        Channel data, size # channels x # time windows x # measures (e.g.,
        frequency bands). Same size as original chan_data, but now all columns
        have the same nan rows (excluding windows where all rows are nan).


    '''
    chan_data = chan_data.copy ()
    n_chan = chan_data.shape[0]

    # which entries are nan - just use first dimension
    # (assume other dimensions have same channel pattern)
    data_nan = np.isnan ( chan_data[:, :, 0] )

    # get unique columns, counts, and coresponding indices
    unique_col, idx, col_counts = np.unique ( data_nan, return_inverse=True,
                                              return_counts=True, axis=1 )

    # ignore all nan win when determining most common pattern
    nan_win = np.sum ( unique_col, 0 ) == n_chan
    col_counts[nan_win] = 0;

    # most common channel pattern
    chan_pat_idx = np.argmax ( col_counts )

    # nan windows that don't match most common pattern
    chan_data[:, idx != chan_pat_idx, :] = np.nan

    return chan_data


def keep_consensus_chan (chan, chan_data):
    '''
    Create consensus list of channels with data in every window of chan_data
    (with the exception of windows that are completely nan)

    Parameters
    ----------
    chan : list
        Channel names.
    chan_data : 3D numpy array
        Channel data, size # channels x # time windows x # measures (e.g., frequency bands).

    Returns
    -------
    chan : list
        List of consensus channel names.
    chan_data : 3D numpy array
        Channel data, now only containing data from consensus channels in chan.

    '''

    # copy
    chan = chan.copy ()
    chan_data = chan_data.copy ()

    # number of channels
    n_chan = chan_data.shape[0]

    # number of windows where all channels are nan
    n_nan_per_win = np.sum ( np.isnan ( chan_data ), 0 )
    n_nan_win = np.sum ( n_nan_per_win == n_chan )

    # number of nan windows per channel (across all frequency bands)
    n_nan_per_chan = np.sum ( np.isnan ( chan_data ), (1, 2) )

    # only keep channels with n_nan_win nan (= consensus channels)
    keep_chan = n_nan_per_chan == n_nan_win
    chan_data = chan_data[keep_chan, :, :]
    chan = [i for (i, v) in zip ( chan, keep_chan ) if v]
    n_removed = n_chan - len ( chan )
    print ( f'Consensus channels: {n_removed} channels removed' )

    return chan, chan_data


def subset_channel_df (channel_df, chan):
    '''
    Subset/sort channel dataframe to match separate list of channels.

    Parameters
    ----------
    channel_df : dataframe
        Channel information stored as a dataframe. Must contain variable 'name'.
    chan : list
        Channel names that channel_df will be matched to. All channel names
        must be present in channel_df['name'].

    Returns
    -------
    channel_df : datafame
        Channel dataframe now subsetted and/or sorted so channel_df['name']
        matches chan.

    '''
    # copy dataframe
    channel_df = channel_df.copy ()

    # get list of all channels from channels dataframe
    all_chan = list ( channel_df['name'].values.copy () )

    # get indices of array channels in channels dataframe
    idx = [all_chan.index ( i ) for i in chan]
    idx = np.array ( idx )

    # sort/subset dataframe by channel indices
    channel_df = channel_df.iloc[idx, :]

    # reset index
    channel_df = channel_df.reset_index ( drop=True )

    return channel_df


def map_chan2rois (chan_data, channel_df, parc_scale):
    '''
    Map channel data (e.g., time-varying band power) to ROI data by averaging
    values of channels within each ROI.

    Parameters
    ----------
    chan_data : 3D numpy array
        Channel data, size # channels x # time windows x # measures
        (e.g., frequency bands).
    channel_df : dataframe
        Channel information stored as a dataframe. Must contain variables
        'name', 'ROIids', and 'is_resected', and must also have the same number
        of rows as chan_data.
    parc_scale : string
        Parcellation scale to use for mapping - must be '36', '60', '125', or
        '250'

    Returns
    -------
    roi_data : 3D numpy array
        ROI data, size # ROI in parcellation x # time windows x # measures
        (e.g., frequency bands).
    prop_roi_resected : numpy array
        Proportion of channels in each ROI that was resected.

    '''
    # dimensions of chan_data
    n_chan, n_win, n_bands = chan_data.shape

    # number of ROIs and parcellation index
    n_roi, parc = get_info_from_parc_scale ( parc_scale )

    # get ROI IDs of channels from dataframe
    # channels in dataframe must match chan_data rows
    roi_ids = channel_df['ROIids'].values

    # make map of channels to ROIs
    # # ROI x # channels
    # (i,j) = 1 if channel j is in ROI i
    chan2roi_map = np.zeros ( (n_roi, n_chan) )
    for j in range ( n_chan ):
        j_id = roi_ids[j][parc]  # channel's ROI ID
        if j_id is not None:  # check that channel has a valid ROI ID
            chan2roi_map[j_id - 1, j] = 1  # minus 1 because IDs are numbered from 1 (MATLAB data)
        else:
            print ( (f"channel {channel_df['name'][j]} does not have a corresponding "
                     "ROI and will therefore be removed from analysis") )

    # number of channels per ROI (so can use map to compute mean signal)
    n_chan_per_roi = np.sum ( chan2roi_map, 1 )
    n_chan_per_roi = n_chan_per_roi[:, np.newaxis]  # add 2nd dimension so can tile

    # divide map by number of channels
    # --> map now includes contribution of each channel to that ROI
    # note - will divide by zero if no channels in an ROI --> missing ROIs will have nan values

    # chan_by_roi = np.tile ( n_chan_per_roi, (1, n_chan) )
    # chan_by_roi_ = chan_by_roi.copy()
    # map0 = np.where(chan_by_roi == 0)
    # chan_by_roi_[map0] = np.nan # replace all 0 values with NaNs
    #
    # chan2roi_map_ = np.where(0, np.nan, chan_by_roi)
    # chan2roi_map_[map0] = np.nan
    # chan2roi_map_[map0] = np.divide ( chan2roi_map_[~map0], chan_by_roi[~map0] )

    chan2roi_map = np.divide ( chan2roi_map, np.tile ( n_chan_per_roi, (1, n_chan) ) )
    if np.sum(np.isinf(chan2roi_map)) != 0:
        print("There are Inf values in the final chan2roi_map matrix.")
        chan2roi_map[np.isinf ( chan2roi_map)] = np.nan # make sure that even of inf will be returned by the division with 0 this will become nan
    else:
        print("There are no Inf values in the final chan2roi_map matrix.")
    # use map to compute mean ROI band power
    roi_data = np.zeros ( (n_roi, n_win, n_bands) )
    for b in range ( n_bands ):
        roi_data[:, :, b] = np.matmul ( chan2roi_map, chan_data[:, :, b] )

    # also map is_chan_resected (--> proportion of ROI resected)
    prop_roi_resected = np.matmul (
        chan2roi_map, channel_df['is_resected'].values.astype ( float ) )

    return roi_data, prop_roi_resected  # return both mappings


def map_chan2rois_all_chan_sets (chan_data, chan, channel_df, parc_scale):
    '''
    Map channel data (e.g., time-varying band power) to ROI data by averaging
    values of channels within each ROI.

    Works with data where sets of channels with non-nan values varies across
    the recording; the mapping will be performed for each channel set/pattern
    and the consolidated into a single output.

    Parameters
    ----------
    chan_data : 3D numpy array
        Channel data, size # channels x # time windows x # measures
        (e.g., frequency bands).
    chan : list
        Channel names that match rows of chan_data
    channel_df : dataframe
        Channel information stored as a dataframe. Must contain variables
        'name', 'ROIids', and 'is_resected', and must also have the same number
        of rows as chan_data.
    parc_scale : string
        Parcellation scale to use for mapping - must be '36', '60', '125', or
        '250'

    Returns
    -------
    roi_data : 3D numpy array
        ROI data, size # ROI in parcellation x # time windows x # measures
        (e.g., frequency bands).
    prop_roi_resected : numpy array
        Proportion of channels in each ROI that was resected (uses max
        proportion found in recording if varies)
    chan_set_info: dictionary with information about different channel
        sets/patterns in the data:
            ['unique_col']: different channel patterns (including all nan
            pattern)
            ['col_idx']: vector indicating which windows in roi_data and chan_data
            have each pattern in unique_col
            ['prop_roi_resected_all']: proportion of each ROI that is resected
            according to each channel pattern

    '''

    # dimensions
    n_chan, n_win, n_bands = chan_data.shape

    # which entries are nans (just use first dimension - assume all are the same)
    data_nan = np.isnan ( chan_data[:, :, 0] )

    # get unique columns, counts, and coresponding indices
    unique_col, idx, col_counts = np.unique ( data_nan, return_inverse=True,
                                              return_counts=True, axis=1 )

    # number of channel patterns
    n_chan_pattern = unique_col.shape[1]

    # initialse array for mapping each channel pattern
    n_roi, _ = get_info_from_parc_scale ( parc_scale )
    roi_data_all = np.zeros ( (n_roi, n_win, n_bands, n_chan_pattern) ) * np.nan
    prop_roi_resected_all = np.zeros ( (n_roi, n_chan_pattern) ) * np.nan

    # map each channel pattern
    for i in range ( n_chan_pattern ):  # (n_chan_pattern)

        # map if not all nan
        if np.sum ( unique_col[:, i] ) != n_chan:
            # nan windows not in pattern
            chan_data_i = chan_data.copy ()
            chan_data_i[:, idx != i, :] = np.nan

            # remove nan channels
            chan_i, chan_data_i = keep_consensus_chan ( chan, chan_data_i )
            channel_df_i = subset_channel_df ( channel_df, chan_i )

            # map and store mapping
            roi_data_all[:, :, :, i], prop_roi_resected_all[:, i] = map_chan2rois (
                chan_data_i, channel_df_i, parc_scale )

            del chan_data_i, chan_i, channel_df_i

    # consolidate mappings across different channels
    roi_data = np.zeros ( (n_roi, n_win, n_bands) ) * np.nan
    for i in range ( n_chan_pattern ):
        data_tmp = roi_data_all[:, :, :, i]
        roi_data[:, idx == i, :] = data_tmp[:, idx == i, :]
        del data_tmp

    # use max proportion resected across different channel patterns
    prop_roi_resected = np.nanmax ( prop_roi_resected_all, axis=1 )

    # save channel pattern info in dictionary
    chan_set_info = dict ()
    chan_set_info['unique_col'] = unique_col
    chan_set_info['col_idx'] = idx
    chan_set_info['prop_roi_resected_all'] = prop_roi_resected_all

    # return
    return roi_data, prop_roi_resected, chan_set_info


def match_final_channel_set (chan_data):
    '''
    Determines final set of channels present in recording (ignoring windows
    with all nan values) and removes data from time windows with different sets
    of channels. Dimensions are not changed - data is removed by turning
    entries to nan.

    Note: only first dimension of chan_data is analysed; assumes other
    dimensions have the same pattern of channels and nan data.
    Parameters
    ----------
    chan_data : 3D numpy array
        Channel data, size # channels x # time windows x # measures (e.g.,
        frequency bands).
    Returns
    -------
    chan_data : 3D numpy array
        Channel data, size # channels x # time windows x # measures (e.g.,
        frequency bands). Same size as original chan_data, but now all columns
        have the same nan rows (excluding windows where all rows are nan).
    '''
    # copy
    chan_data = chan_data.copy ()

    # dimensions
    n_chan, n_win, _ = chan_data.shape

    # which channels are nan (just use first dimension - assume rest have same pattern)
    nan_chan = np.isnan ( chan_data[:, :, 0] )

    # time windows where channels are not all nan
    not_all_nan = np.sum ( nan_chan, 0 ) < n_chan

    # last window that does not have all NaNs
    idx = n_win - np.argmax ( not_all_nan[::-1] ) - 1;

    # find time windows that are not all nan AND match final set of channels
    final_chan = nan_chan[:, idx]
    matching_win = np.sum ( nan_chan == final_chan[:, None], 0 ) == n_chan
    keep_win = np.logical_and ( matching_win, not_all_nan )

    # turn other windows to all nan
    chan_data[:, np.invert ( keep_win ), :] = np.nan
    n_removed = np.sum ( not_all_nan ) - np.sum ( keep_win )
    print ( f'Removed {n_removed} time windows that had different channels' )

    return chan_data
