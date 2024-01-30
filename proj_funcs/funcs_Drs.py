"""

@Mariellapanag

Created on Feb 2023
@author: mariella

Functions for computing metrics related to Drs (AUC) and statistical tests

Includes
    calculate_DRS


TODO:

"""


# internal modules
import sklearn.metrics
import numpy as np




# Function for calculating DRS (and other metrics) for each subject
def calculate_DRS (cycles_measures_patient, roi_is_resect):
    '''
    Compute D_RS (AUC) of resected vs. spared ROIs based on ROI measure

    D_RS > 0.5 indicates measure higher in spared ROIs
    D_RS is <0.5 when measure is higher for resected ROIs
    Parameters
    ----------
    cycles_measures_patient : 3D numpy array, float
        size: # ROIs x # cycles # signal measures
    roi_is_resect : 1D numpy array, boolean
        Whether each ROI was resected (True if resected, False if spared).
    Returns
    -------
    d_rs : 2D numpy array, float
        AUC at each cycle and measure, size # cycles x
        # measures.
    '''

    # data dimensions
    n_roi, n_cycles, n_measures = cycles_measures_patient.shape
    # Print progress message
    print ( "Calculating DRS and other metrics ..." )

    # initalise d_rs array
    d_rs = np.zeros ((n_cycles, n_measures))

    # check that have both classes
    if np.sum ( roi_is_resect ) > 0 and np.sum ( np.invert ( roi_is_resect ) ) > 0:
        # compute Drs for each measure and cycle
        for cc in range(n_cycles):
            for mm in range(n_measures):
                # specified column corresponding to the cycle and measure
                measure = cycles_measures_patient[:, cc, mm]

                # Calculate DRS
                d_rs[cc, mm] = 1 - sklearn.metrics.roc_auc_score ( roi_is_resect, measure)
    else:
        d_rs = d_rs * np.nan
    # Print progress message
    print ( "Successfully computed DRS values and other metrics." )

    # Return DRS
    return (d_rs)

def calculate_DRS_power (cycles_power_pat, roi_is_resect):
    '''
    Compute D_RS (AUC) of resected vs. spared ROIs based on ROI measure

    D_RS > 0.5 indicates measure higher in spared ROIs
    D_RS is <0.5 when measure is higher for resected ROIs
    Parameters
    ----------
    cycles_measures_patient : 3D numpy array, float
        size: # ROIs x # cycles # signal measures
    roi_is_resect : 1D numpy array, boolean
        Whether each ROI was resected (True if resected, False if spared).
    Returns
    -------
    d_rs : 2D numpy array, float
        AUC at each cycle and measure, size # cycles x
        # measures.
    '''

    # data dimensions
    n_roi, n_cycles = cycles_power_pat.shape
    # Print progress message
    print ( "Calculating DRS and other metrics ..." )

    # initalise d_rs array
    d_rs = np.zeros (n_cycles)

    # check that have both classes
    if np.sum ( roi_is_resect ) > 0 and (roi_is_resect[roi_is_resect == 0]).shape[0] > 0:
        # compute Drs for each measure and cycle
        for cc in range(n_cycles):
            # specified column corresponding to the cycle and measure
            measure = cycles_power_pat[:, cc]

            # Calculate DRS
            d_rs[cc] = 1 - sklearn.metrics.roc_auc_score ( roi_is_resect, measure)
    else:
        d_rs = d_rs * np.nan
    # Print progress message
    print ( "Successfully computed DRS values and other metrics." )

    # Return DRS
    return (d_rs)


def compute_d_rs_power_time (roi_data, roi_is_resect):
    '''
    Compute D_RS (AUC) of resected vs. spared ROIs based on ROI measure

    D_RS is <0.5 when measure is higher for resected ROIs

    Parameters
    ----------
    roi_data : 3D numpy array, float
        ROI data, size # ROIs x # time windows # frequency bands.
    roi_is_resect : 1D numpy array, boolean
        Whether each ROI was resected (True if resected, False if spared).

    Returns
    -------
    d_rs : 2D numpy array, float
        AUC at each time window and frequency band, size # time windows x
        # frequency bands.

    '''
    # data dimensions
    n_roi, n_win, n_bands = roi_data.shape

    # initalise d_rs array
    d_rs = np.zeros ( (n_win, n_bands) )

    # check that have both classes
    if np.sum ( roi_is_resect ) > 0 and (roi_is_resect[roi_is_resect == 0]).shape[0] > 0:
        # compute d_rs at each time window and for each frequency band
        for b in range ( n_bands ):
            print ( b )
            for i in range ( n_win ):

                # z-scores from specified time window and frequency band
                z = roi_data[:, i, b].copy ()

                # keep non-nan ROI
                keep_roi = np.invert ( np.isnan ( z ) )
                z = z[keep_roi]
                spared_i = roi_is_resect[keep_roi] == 0  # also invert so indicates spared
                resected_i = roi_is_resect[keep_roi] == 1

                # check that still have both classes and window is not completely nan
                if (np.sum ( spared_i ) > 0) and (np.sum ( resected_i ) > 0) and (np.sum ( keep_roi ) > 0):
                    d_rs[i, b] = 1 - sklearn.metrics.roc_auc_score (
                        roi_is_resect[keep_roi], z )  # compute D_RS
                else:
                    d_rs[i, b] = np.nan
    else:
        d_rs = d_rs * np.nan

    return d_rs
