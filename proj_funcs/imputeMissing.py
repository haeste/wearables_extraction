import numpy as np
import pandas as pd


def find_nan(BPall_channel):

    # select one channel
    BP_all_df = pd.Series(BPall_channel)
    na_groups = BP_all_df.isnull().astype(int).groupby(BP_all_df.notnull().astype(int).cumsum()).sum()

    nan_cons_values = na_groups[na_groups>0]
    # na_groups = BP_all_df.notna().cumsum()[BP_all_df.isna()]
    # lengths_consecutive_na = na_groups.groupby(na_groups).agg(len)
    # longest_na_gap = lengths_consecutive_na.max()

    # The indices of the NaN values of the first row if band power (all band power in different frequencies share the same NaN values)
    indices_nan = np.where(np.isnan(BPall_channel))

    # The onset of NaN values
    cum_idx_start = 0
    cum_idx_start_l = [0]
    for i in range(nan_cons_values.values.shape[0]):
        cum_idx_start = cum_idx_start + nan_cons_values.values[i]
        cum_idx_start_l.append(cum_idx_start)

    # The index_start:index_end of each block of consecutive NaN values (note that the index_end is not included as index when used to extract part of array)
    range_nan_idx = [(cum_idx_start_l[i],cum_idx_start_l[i+1]) for i in range(len(cum_idx_start_l)-1)]

    # A list of tuples (n, [indices ... ]) for each NaN block
    nan_l = [(nan_cons_values.iloc[i], indices_nan[0][range_nan_idx[i][0]:range_nan_idx[i][1]].tolist()) for i in range(nan_cons_values.values.shape[0])]

    return nan_l

def imputed_surround(BPall_channel, std_frac = 0.6):

    nan_l = find_nan(BPall_channel)
    # Firstly, we will impute single NaNs with the mean values of the surroundings

    # Find which of the NaN blocks contain 1 NaN value
    indx_1NaN = [nan_l[i][1] for i in range(len(nan_l)) if nan_l[i][0] == 1]

    # Make list of lists flatten
    def flatten(l):
        result = [element for sublist in l for element in sublist]
        return result

    indx_1NaN_flat = flatten(indx_1NaN)

    BPall_imputed_channel = BPall_channel.copy()
    if indx_1NaN_flat != None:
        for i in indx_1NaN_flat:
            data_temp = np.transpose(np.vstack([BPall_imputed_channel[i-1],BPall_imputed_channel[i+1]]))
            mean_temp = np.mean(data_temp)
            BPall_imputed_channel[i] = mean_temp

    # Find which of the NaN blocks contain more than 1 NaN value
    NaN_info_greaterNaN = [nan_l[i] for i in range(len(nan_l)) if nan_l[i][0] != 1]

    # Go through the blocks of NaN and impute them with a linear interpolation of the mean of the surroundings
    for i in range(len(NaN_info_greaterNaN)):
        mask = NaN_info_greaterNaN[i][1]
        miss_len = NaN_info_greaterNaN[i][0]

        # if there aren't enough values before the missing part
        # start from the beginning of the recording and take the mean
        if min(mask) <= miss_len:
            if min(mask)!=0:
                data_temp = np.hstack([BPall_imputed_channel[0:min(mask)],BPall_imputed_channel[(max(mask)+1):max(mask)+miss_len+1]])
                mean_temp = np.nanmean(data_temp)
                std_temp = np.nanstd(data_temp)*std_frac
                noise = np.random.normal(0, std_temp, len(mask))
                BPall_imputed_channel[mask] = np.repeat(mean_temp, len(mask)) + noise
            if min(mask) == 0:
                # if missing data are in the start of the recording then use the segment
                # after the end of the consecutive missing period
                data_temp = BPall_imputed_channel[(max(mask)+1):max(mask)+miss_len+1]
                mean_temp = np.nanmean(data_temp)
                std_temp = np.nanstd(data_temp)*std_frac
                noise = np.random.normal(0, std_temp, len(mask))
                BPall_imputed_channel[mask] = np.repeat(mean_temp, len(mask)) + noise
        else:
            # define pre and post segments of values of length miss_len
            segment_pre = BPall_imputed_channel[(min(mask)-miss_len):(min(mask))]
            segment_post = BPall_imputed_channel[(max(mask)+1):max(mask)+miss_len+1]
            # Check if there are NaN values in segments before and after
            # If there are NaNs which corresponds to <50% then continue the calculation
            # if ((np.sum(np.isnan(segment_pre))/segment_pre.shape[0] < 0.5) and (np.sum(np.isnan(segment_post))/segment_post.shape[0] < 0.5)):
            # Compute the mean values for the surrounding segments
            mean_pre = np.nanmean(segment_pre)
            mean_post = np.nanmean(segment_post)
            # Compute the standard deviation of the surrounding segments
            sd = np.nanstd(np.hstack([segment_pre, segment_post]))

            y_all = np.hstack([mean_pre, BPall_imputed_channel[mask], mean_post])
            x_all = range(0, len(y_all))
            y_known = [mean_pre, mean_post]
            x_known = [0, len(x_all)]
            x_unknown = x_all[1:len(x_all)-1]

            y_new = np.interp(x_unknown, x_known, y_known)

            #plt.plot(x_known, y_known, "og-", x_unknown, y_new, "or");
            t = x_all[1:len(x_all)-1]
            n = np.random.normal(0, sd, len(t)) * std_frac
            y_new_noise = y_new + n

            BPall_imputed_channel[mask] = y_new_noise
            # else:
            #     # Compute the mean value of the available surrounding segments and use that to impute
            #     temp_mean = np.nanmean(np.hstack([segment_pre, segment_post]))
            #     y_imp = np.repeat(temp_mean, miss_len)
            #     std_temp = np.nanstd(temp_mean)*std_frac
            #     noise = np.random.normal(0, std_temp, len(mask))
            #     BPall_imputed_channel[mask] = y_imp + noise
    return BPall_imputed_channel, nan_l

def imputed_surround_all(BPall):
    [n_chan, time] = BPall.shape
    nan_around = list()
    # find consecutive NaN values at the start of the recording
    nan_start_dur = list()
    cc_id = list()
    for cc in range(n_chan):
        f_nan = find_nan(BPall[cc,:])
        # is the index less than the length of the missing block?
        if np.min(f_nan[0][1])<f_nan[0][0]:
            #print(f_nan[0])
            nan_start_dur.append(f_nan[0][0])
            cc_id.append(cc)

    if len(nan_start_dur)!=0:
        id_max = np.argmax(nan_start_dur)
        cc_max = cc_id[id_max]
        f_nan = find_nan(BPall[cc_max,:])
        nan_around.append(f_nan[0][1])
    else:
        nan_around = None

    nan_all = list()
    if nan_around !=None:
        indices_keep = [ii for ii in range(time) if ii not in nan_around]
        BPall_new = BPall[:,indices_keep]
        [n_chan, time_new] = BPall_new.shape
        BPallimputed = np.empty((n_chan, time_new))
        nan_all = list()
        for cc in range(n_chan):
            [BPtemp, nan_l] = imputed_surround(BPall_new[cc,:])
            BPallimputed[cc,:] = BPtemp
            nan_all.append(nan_l)
    else:
        BPallimputed = np.empty((n_chan, time))

        for cc in range(n_chan):
            [BPtemp, nan_l] = imputed_surround(BPall[cc,:].flatten())
            BPallimputed[cc,:] = BPtemp
            nan_all.append(nan_l)

    return BPallimputed, nan_all, nan_around

def impute_mean(BPall):

    [n_rows, n_cols] = BPall.shape
    BPall_imputed = np.empty((n_rows, n_cols))
    nan_all = list()
    for rr in range(n_rows):
        f_nan = find_nan(BPall[rr,:])
        nan_all.append(f_nan)
        df = pd.Series(BPall[rr,:])
        BPall_imputed[rr,:] = df.fillna(df.mean())

    return BPall_imputed, nan_all