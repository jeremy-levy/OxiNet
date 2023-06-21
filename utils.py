import numpy as np
from pobm.prep import set_range, median_spo2
from scipy.signal import savgol_filter


def split_signal(array, len_chunk, len_sep):
    """Returns a matrix of all full overlapping chunks of the input `array`, with a chunk
    length of `len_chunk` and a separation length of `len_sep`. Begins with the first full
    chunk in the array. """

    n_arrays = int(np.ceil((array.size - len_chunk + 1) / len_sep))

    array_matrix = np.tile(array, n_arrays).reshape(n_arrays, -1)

    columns = np.array(((len_sep * np.arange(0, n_arrays)).reshape(n_arrays, -1) + np.tile(
        np.arange(0, len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    rows = np.array((np.arange(n_arrays).reshape(n_arrays, -1) + np.tile(
        np.zeros(len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    return array_matrix[rows, columns]


def split_window(X, params):
    if len(X) >= params['padding_size']:
        X = X[0: params['padding_size']]
    else:
        X = np.pad(X, ((0, params['padding_size'] - len(X)),), constant_values=0)

    X = X[np.newaxis, :]

    window_size = params['window_size']
    if params['overlap'] == 0:
        len_sep = window_size
        nb_windows = int(round(params['padding_size'] / window_size))
    else:
        len_sep = window_size / 2
        nb_windows = (int(round(params['padding_size'] / window_size)) * 2) - 1

    do_again = False
    new_X = np.zeros(shape=(X.shape[0], nb_windows, window_size))
    for i in range(X.shape[0]):
        x_splitted = split_signal(X[i, :], window_size, len_sep)
        try:
            new_X[i, :, :] = x_splitted
        except ValueError:
            do_again = True
            break

    if do_again is True:
        nb_windows -= 1
        new_X = np.zeros(shape=(X.shape[0], nb_windows, window_size))
        for i_inter in range(X.shape[0]):
            x_splitted = split_signal(X[i_inter, :], window_size, len_sep)
            new_X[i_inter, :, :] = x_splitted

    return new_X


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def linear_interpolation(signal):
    signal = np.array(signal)

    nans, x = nan_helper(signal)
    signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])

    return signal


def preprocess_oximetry(oximetry_signal, params):
    signal = set_range(oximetry_signal)

    if True == params['apply_median_spo2']:
        signal = median_spo2(signal)

    if True == params['apply_savgol_filter']:
        signal = savgol_filter(signal, window_length=5, polyorder=3)

    if True == params['apply_linear_interpolation']:
        signal = linear_interpolation(signal)

    if True == params['apply_int_convertor']:
        signal = np.rint(signal)

    return signal
