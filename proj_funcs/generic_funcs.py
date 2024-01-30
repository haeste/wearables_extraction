import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import pandas as pd
import json


def format_global(value):

    if (value >= 0.8):
        return "{}d".format ( round ( value, 1 ) )
    elif (value < 0.8) and (value >= 1 / 24):
        return "{}h".format ( round ( value * 24, 1 ) )
    elif (value < 1 / 24):
        return "{}m".format ( round ( value * 24 * 60, 1 ) )


def pickle_save (filename, glob, *args):
    d = {}
    for k in args:
        d[k] = glob[k]
    with open ( filename, 'wb' ) as f:
        pickle.dump ( d, f )


def pickle_load (filename, glob, *args):
    with open ( filename, 'rb' ) as f:

        # pickle dictionary
        pickle_d = pickle.load ( f )

    # make each dictionary entry a variable

    # load all variables in pickle file
    if not args:
        for k, v in pickle_d.items ():
            glob[k] = v

    # load specified variables
    else:
        for k in args:
            glob[k] = pickle_d[k]


def mat_save (filename, glob, *args):
    # make dictionary to save
    d = {}
    for k in args:
        d[k] = glob[k]

    # save dictionary
    sio.savemat ( filename, d, do_compression=True )



def mat_load (filename, glob, *args):
    # get dictionary
    d = sio.loadmat ( filename )

    # make each dictionary entry a variable

    # load all variables
    if not args:
        for k, v in d.items ():
            glob[k] = v

    # load specified variables
    else:
        for k in args:
            glob[k] = d[k]


def mat_load_as_dict (filename, *args):
    d = sio.loadmat ( filename, squeeze_me=True, chars_as_strings=True, mat_dtype=False )

    if not args:
        # load all variables
        mat_dict = d
    else:
        # only keep specified variables in dictionary
        mat_dict = dict ()
        for k in args:
            mat_dict[k] = d[k]

    return mat_dict


def save_plot (plot_dir, fname, plot_formats=None):
    # plot_formats is list

    # ensure plot_dir directory exists
    if not os.path.exists ( plot_dir ):
        os.makedirs ( plot_dir )

    # default format is png
    if plot_formats is None:
        plot_formats = ['png']

    # save plot in each requested format
    n_formats = len ( plot_formats )
    for i in range ( n_formats ):
        plt.savefig ( os.path.join(plot_dir, "{}.{}".format(fname, plot_formats[i])), format=plot_formats[i], bbox_inches='tight' )


def load_patient_json (json_path):
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



def add_elements_in_dict(d, glob, *args):

    # make each dictionary entry a variable
    # add elements/variables
    if not args:
        print('nothing else is added')
    # load specified variables
    else:
        for k in args:
            if k not in d:
                d[k] = glob[k]

    return d
