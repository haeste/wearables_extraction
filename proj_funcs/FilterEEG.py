from scipy import signal
import numpy as np

def FilterEEG(EEG: np.array, cutoff: list, sample_rate: float, butterworth_type: str, order: int):
    r"""

    Simple Butterworth filter, with zero-phase filtering
    Importance of use of 2nd order sections:
    https://stackoverflow.com/questions/21862777/bandpass-butterworth-filter-frequencies-in-scipy
    Args:
        EEG: matrix of EEG signals (rows = channels, cols = time points)
        cutoff: list specifying cutoff of lowpass or highpass filter, or
        two-element vector for bandpass filter (in Hz)
        sample_rate: sampling frequency, in Hz
        butterworth_type: the type of filter; 'low' for lowpass, 'high' for highpass,
        'bandpass' for bandpass, 'stop' for bandstop
        order: the order of the Butterworth filter (order is 2*order for the resulting filter,
        in the case of bandpass filter

    Returns:
        (digital) filtered EEG signal as a matrix object (rows = channels, cols = time points)

    """
    sos = signal.butter(N = order, Wn = cutoff, btype = butterworth_type, analog = False,
                           fs = sample_rate, output = "sos") # create the filter
    filtered = np.empty((EEG.shape[0], EEG.shape[1]))

    # Apply filter in every signal (#rows = #individual signals)
    for i in range(EEG.shape[0]):
        filtered[i,:] = signal.sosfiltfilt(sos, EEG[i,:]) # Apply a
        # forward-backward digital filter using cascaded second-order sections.
    return filtered


def FilterEEG_Channel(EEG: np.array, cutoff: list, sample_rate: float, butterworth_type: str, order: int):
    r"""
    Simple Butterworth filter, with zero-phase filtering
    Importance of use of 2nd order sections:
    https://stackoverflow.com/questions/21862777/bandpass-butterworth-filter-frequencies-in-scipy

    Args:
        EEG: one single EEG signal (row = 1 channel, cols = time points)
        cutoff: list specifying cutoff of lowpass or highpass filter, or
        two-element vector for bandpass filter (in Hz)
        sample_rate: sampling frequency, in Hz
        butterworth_type: the type of filter; 'low' for lowpass, 'high' for highpass,
        'bandpass' for bandpass, 'stop' for bandstop
        order: the order of the Butterworth filter (order is 2*order for the resulting filter,
        in the case of bandpass filter

    Returns:
        (digital) filtered EEG signal as a single array (time points)

    """
    sos = signal.butter(N = order, Wn = cutoff, btype = butterworth_type, analog = False,
                        fs = sample_rate, output = "sos") # create the filter
    # Apply filter in every signal (#individual signal)
    # Apply a forward-backward digital filter using cascaded second-order sections.
    filtered = signal.sosfiltfilt(sos, EEG)

    return filtered


# Check the filter
#
# def butter_bandpass(cutoff, sample_rate, butterworth_type, order):
#     nyq = sample_rate/2 # Nyquist frequency
#
#         # If cutoff is a list (bandpass), else just the lowpass or highpass frequency as a single number...
#     if isinstance(cutoff, list):
#         Wn = [freq/nyq for freq in cutoff]
#     else:
#         Wn = cutoff/nyq
#
#     sos = signal.butter(N = order, Wn = Wn, btype = butterworth_type, analog = False,
#                         fs = sample_rate, output = "sos") # create the filter
#     return sos
#
# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.signal import freqz
#     # Sample rate and desired cutoff frequencies (in Hz).
#     fs = 400
#     # Plot the frequency response for a few different orders.
#     plt.figure(1)
#     plt.clf()
#     for order in [1, 2, 3, 4, 5, 6, 9, 10]:
#         sos = butter_bandpass([4, 8], fs, "bandpass",order=order)
#         w, h = signal.sosfreqz(sos, worN=1500)#np.logspace(-4, 3, 2000))
#         plt.semilogx((fs * 0.5 / np.pi) * w, abs(h), label="order = {}".format(order*2))
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Gain')
#     plt.title("fs:{}".format(fs))
#     plt.grid(True)
#     plt.legend(loc='best')
#

    # plt.figure(2)
    # plt.clf()
    # for order in [1, 3, 4, 5, 6, 9, 10]:
    #     sos = butter_bandpass([0.05, 0.4], fs,"bandpass", order=order)
    #     w, h = signal.sosfreqz(sos, worN=2000)#np.logspace(-4, 3, 2000))
    #     plt.semilogx((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain')
    # plt.grid(True)
    # plt.legend(loc='best')
    #
    # plt.show()

    # def butter_bandpass1(cutoff, sample_rate, butterworth_type, order):
    #     nyq = sample_rate/2 # Nyquist frequency
    #
    #     # If cutoff is a list (bandpass), else just the lowpass or highpass frequency as a single number...
    #     if isinstance(cutoff, list):
    #         Wn = [freq/nyq for freq in cutoff]
    #     else:
    #         Wn = cutoff/nyq
    #
    #     b, a = signal.butter(N = order, Wn = Wn, btype = butterworth_type, analog = False,
    #                         fs = sample_rate) # create the filter
    #     return b,a
    #
    # t = np.linspace(0, 1, 400, False)  # 1 second
    # sig = np.sin(2*np.pi*1*t) + np.sin(2*np.pi*2*t) #+ np.sin(2*np.pi*5*t) #+ np.sin(2*np.pi*20*t)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # ax1.plot(t, sig)
    # ax1.set_title('10 Hz and 20 Hz sinusoids')
    # ax1.axis([0, 1, -2, 2])
    #
    # sos = signal.butter(2, [1, 4], 'bp', fs=400, output='sos')
    # filtered = signal.sosfilt(sos, sig)
    # ax2.plot(t, filtered)
    # ax2.set_title('After band-pass filter')
    # ax2.axis([0, 1, -2, 2])
    # ax2.set_xlabel('Time [seconds]')
    # plt.tight_layout()
    # plt.show()



