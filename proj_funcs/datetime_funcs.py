
import datetime

def isbetween(time: datetime.datetime, time_range: tuple):
    r"""
    Function for finding if a certain datetime object exists in a specified datetime range.
    The boundaries of the time range are also included.

    Args:
        time: a datetime object
        time_range: a tuple depicting a time range. Each time in the time range is a datetime object.

    Returns:
        boolean: True if the condition is met. Otherwise, the function returns False
    """
    if time_range[1] < time_range[0]:
        return time >= time_range[0] or time <= time_range[1]
    return time_range[0] <= time <= time_range[1]

# def get_sz_start_win(pnt, fname_times, fname_sz_table):
#
#     # load time segments - windows
#     pnt_times = pd.read_csv ( fname_times, index_col=0 )
#
#     # load seizure severity table
#     sz_table = pd.read_excel ( fname_sz_table , sheet_name="Sheet1")
#
#     # detect the seizures for the specific subject
#     pnt_sz = sz_table[sz_table["patient_id"] == pnt]
#     pnt_sz = pnt_sz.reset_index ()  # make sure indexes pair with number of rows
#     pnt_sz = pnt_sz[['db_ind', 'patient_id', 'start', 'duration', 'ilae_sz_type']]
#
#     sz_table['start'] = sz_table['start'].apply ( pd.to_datetime )
#
#     sz_time_start = sz_table["start"]
#     # sz_time_stop = sz_time_start + datetime.timedelta(seconds=pnt_sz.duration)
#     pnt_times["start"] = pd.to_datetime ( pnt_times['start'] )  # , format='%Y%m%d_%H%M%S')
#     pnt_times["stop"] = pd.to_datetime ( pnt_times['stop'] )  # , format='%Y%m%d_%H%M%S')
#
#
#     start_sz_window = list ()
#     for sz in range ( len ( sz_time_start ) ):
#         win = 0
#         for ii in range ( len ( pnt_times ) ):
#             if isbetween ( time=sz_time_start.iloc[sz],
#                                    time_range=(pnt_times["start"].iloc[ii], pnt_times["stop"].iloc[ii]) ):
#                 print ( sz )
#                 start_sz_window.append ( win )
#             win = win + 1
#
#     return start_sz_window
#
