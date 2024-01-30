
"""

@Mariellapanag

Created on Feb 2023
@author: mariella

Functions for computing abnormalities from continuous iEEG data.

Includes
    power signal : compute power of signal
    energy_range : compute energy range for each signal

    get_cycles: obtain cycles using bandpass filter

TODO:

"""
# internal modules
import numpy as np

# external modules
import proj_funcs.imputeMissing as impute
import proj_funcs.map_back_NaNs as map_nan
import proj_funcs.FilterEEG as filter_funcs


def power_signal_through_time(array_3d):

    [n_roi, n_win, n_cycles] = array_3d.shape

    results_array = np.zeros((n_roi, n_win, n_cycles))
    for dd in range(n_cycles):
        data_power = array_3d[:,:,dd]**2 # compute power

        results_array[:, :, dd] = data_power

    return results_array

def power_signal(array_3d):

    [n_roi, n_win, n_cycles] = array_3d.shape

    results_array = np.zeros((n_roi, n_cycles))
    for dd in range(n_cycles):
        data_tmp = array_3d[:,:,dd]**2 # compute power
        power = np.nanmean ( data_tmp, axis=1 )

        results_array[:, dd] = power

    return results_array

def energy_range(array_3d):

    [d1, d2, d3] = array_3d.shape

    results_array = np.zeros((d1, d3))
    for dd in range(d3):
        row_range = np.nanmax ( array_3d[:,:,dd], axis=1 ) - np.nanmin ( array_3d[:,:,dd], axis=1 )

        results_array[:, dd] = row_range

    return results_array


def get_cycles_bandpass(roi_data, fluct_name, fluct_narrowband, srate):
    '''
    This function extracts the cycles from each channel (row)

    Args:
        roi_data: # ROI x # time-windows
        roi_num:
        is_missing:
        is_missing_edges:

    Returns:
    '''

    [n_roi, n_win] = roi_data.shape
    # check if there are missing data
    is_missing = np.sum ( np.isnan ( roi_data ) )

    if (is_missing != 0):

        # Impute the data
        # Imputed data from l1-norm
        [roi_data_imp, nan_all, nan_around] = impute.imputed_surround_all ( roi_data )

        # check if there are still missing data after the imputation; this would mean that there are missing values
        # at the end of the recording
        is_missing_edges = np.sum ( np.isnan ( roi_data_imp ) )

        if is_missing_edges != 0:

            # find the NaN values and where those are located
            nan_l = []
            nan_count = []
            for cc in range ( n_roi ):
                # use one of the dataframes above to get the list with consecutive NaN values
                nan_tmp = impute.find_nan ( roi_data_imp[cc,:] )  # function for detecting consecutive NaN values in one single array (in that case each channel)
                nan_l.append ( nan_tmp )
                nan_c = len ( nan_tmp )
                nan_count.append ( nan_c )

            nan_indices = np.isnan ( roi_data_imp ).all ( axis=0 )
            # Rerun the imputation with the max_absZscore_l1 without the NaN values at the end; running it again
            # to obtain the nan_all and nan_around variables correctly
            [roi_data_imp, nan_all, nan_around] = impute.imputed_surround_all (
                roi_data[:, ~nan_indices] )

            # temporary file just for plotting later
            roi_data_imp_tmp = map_nan.map_NaN_consec_end ( roi_data_imp, n_roi, n_win, nan_l )
        else:
            roi_data_imp_tmp = None

        # final cycles with imputed NaNs values
        cycles_output_imp = np.zeros ( (n_roi, n_win, len ( fluct_name )) )
        # final cycle with mapping back NaNs
        cycles_output = np.zeros ( (n_roi, n_win, len ( fluct_name )) )
        jj = 0
        for keyname in fluct_name:

            cycles_tmp = filter_funcs.FilterEEG ( roi_data_imp, cutoff=fluct_narrowband[keyname],
                                                  sample_rate=srate,
                                                  butterworth_type="bandpass", order=2 )
            cycles_tmp_imp = cycles_tmp
            cycles_nan = map_nan.map_NaN_values_channel ( cycles_tmp, nan_all )

            if is_missing_edges != 0:
                # for the consecutive NaN values at the end of the recording if those exist
                cycles_nan = map_nan.map_NaN_consec_end(cycles_nan, n_roi, n_win, nan_l)
                # in the case of NaNs at the end of the recording the imputed data will have NaNs at the end
                # we don't impute those data
                cycles_tmp_imp = map_nan.map_NaN_consec_end(cycles_tmp_imp, n_roi, n_win, nan_l)
            cycles_output_imp[:,:,jj] = cycles_tmp_imp
            cycles_output[:, :, jj] = cycles_nan
            jj = jj + 1
    else:
        roi_data_imp = None
        roi_data_imp_tmp = None
        # final cycles with imputed NaNs values
        cycles_output_imp = np.zeros ( (n_roi, n_win, len ( fluct_name )) )
        # final cycle with mapping back NaNs
        cycles_output = np.zeros ( (n_roi, n_win, len ( fluct_name )) )
        jj = 0
        for keyname in fluct_name:
            cycles_tmp = filter_funcs.FilterEEG ( roi_data, cutoff=fluct_narrowband[keyname],
                                                  sample_rate=srate,
                                                  butterworth_type="bandpass", order=2 )

            cycles_output_imp[:, :, jj] = cycles_tmp
            cycles_output[:, :, jj] = cycles_tmp

            jj = jj + 1

    return cycles_output, cycles_output_imp, roi_data_imp, roi_data_imp_tmp
