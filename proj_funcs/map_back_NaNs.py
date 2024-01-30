
"""
@Mariellapanag

Created on Feb 2023
@author: mariella

Functions for computing abnormalities from continuous iEEG data.

Includes
        map_NaN_values_channel



TODO: Write down comments on this script
"""

import numpy as np


def map_NaN_values_IMFchar(data, nan_all):

    [n_modes, n_roi, time] = data.shape

    data_nan = data.copy ()

    for channel in range ( n_roi ):
        #print ( n_roi )
        for i in range ( len ( nan_all[channel] ) ):
            print ( i )
            mask = nan_all[channel][i][1]
            mask_nan_block = np.empty ( len ( mask ))
            mask_nan_block[:] = np.nan
            data_nan[:, channel, mask] = mask_nan_block

    return data_nan


def map_NaN_values_channel(data, nan_all):
    '''
    Map back the NaN values!

    Args:
        data: a 2d array, size: # ROIs x # time windows
        nan_all:

    Returns:

    '''

    [n_roi, time] = data.shape
    data_nan = data.copy()

    for channel in range ( n_roi ):
        #print ( n_roi )
        for i in range ( len ( nan_all[channel] ) ):
            #print ( i )
            mask = nan_all[channel][i][1]
            mask_nan_block = np.empty ( len ( mask ))
            mask_nan_block[:] = np.nan
            data_nan[channel, mask] = mask_nan_block

    return data_nan


def map_NaN_consec_end_F_and_A(data_part, n_roi, n_win):
    n_modes = data_part.shape[0]
    n_win_part = data_part.shape[2]
    data_f = np.zeros ( (n_modes, n_roi, n_win) )

    for cc in range ( n_roi ):

        min_ind_NaN_map = n_win_part
        # induce all values till the start of the NaN based on the output before within the final 2D array
        data_f[:, cc, :min_ind_NaN_map] = data_part[:, cc, :]
        mask_nan_block = np.empty ( n_win-n_win_part )
        mask_nan_block[:] = np.nan
        data_f[:, cc, min_ind_NaN_map: n_win] = mask_nan_block

    return data_f
def map_NaN_consec_end_imfchar(data_part, n_roi, n_win, nan_l):

    n_modes = data_part.shape[0]
    data_f = np.zeros ( (n_modes, n_roi, n_win) )


    for cc in range ( n_roi ):
        for nan_block in range ( len ( nan_l[cc] ) ):
            ind_NaN_map = nan_l[cc][nan_block][1]
            min_ind_NaN_map = min( ind_NaN_map )
            # induce all values till the start of the NaN based on the output before within the final 2D array
            data_f[:, cc, :min_ind_NaN_map] = data_part[:, cc, :]
            mask_nan_block = np.empty ( len ( ind_NaN_map ) )
            mask_nan_block[:] = np.nan
            data_f[:, cc, ind_NaN_map] = mask_nan_block

    return data_f

def map_NaN_consec_end(data_part, nrows_f, ncols_f, nan_l):


    data_f = np.zeros ( (nrows_f, ncols_f) )

    for cc in range ( nrows_f ):
        for nan_block in range ( len ( nan_l[cc] ) ):
            ind_NaN_map = nan_l[cc][nan_block][1]
            min_ind_NaN_map = min( ind_NaN_map )
            # induce all values till the start of the NaN based on the output before within the final 2D array
            data_f[cc, :min_ind_NaN_map] = data_part[cc, :]
            data_f[cc, ind_NaN_map] = np.nan

    return data_f