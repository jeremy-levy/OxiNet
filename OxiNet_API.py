import os

import pandas as pd
from joblib import load
import numpy as np
import argparse

from architecture import get_duplo
from utils import split_window, preprocess_oximetry


def get_nb_window(overlap, signal_size, window_size, padding_size=-1):
    signal_size = max(signal_size, padding_size)

    if overlap == 0:
        nb_windows = int(round(signal_size / window_size))
    else:
        nb_windows = (int(round(signal_size / window_size)) * 2) - 1
    return nb_windows


def load_(params_path, model_path, scaler_path):
    oximetry_scaler = load(scaler_path)

    params = pd.read_csv(os.path.join(params_path)).iloc[0].to_dict()
    nb_windows = get_nb_window(params['overlap'], params['signal_size'], params['window_size'], params['padding_size'])
    params['shape'] = (int(nb_windows), int(params['window_size']))
    params['regularizer'] = 0.1

    for key in ['padding_size', 'n_filters_lstm', 'nb_conv_lstm', 'num_features', 'dilations_num',
                'residual_convolution_n_convs', 'num_residual_convolution', 'first_conv_n_filters', 'window_size']:
        params[key] = int(params[key])
    for key in ['first_conv_kernel_size', 'residual_convolution_kernel_size', 'dilated_block_kernel_size', 'dilation',
                'kernel_size_lstm']:
        params[key] = (int(params[key]),)

    model = get_duplo(params)
    model.load_weights(model_path)

    return oximetry_scaler, params, model


def apply_oxinet(oximetry_signal, oximetry_scaler, params, model):
    oximetry_signal = split_window(oximetry_signal, params)

    original_shape_test = oximetry_signal.shape
    oximetry_signal = oximetry_signal.reshape(-1, original_shape_test[-1])
    oximetry_signal = oximetry_scaler.transform(oximetry_signal)
    oximetry_signal = oximetry_signal.reshape(original_shape_test)

    ahi_predicted = model.predict([oximetry_signal, np.zeros(shape=(1, 176))])
    return ahi_predicted


def run_model(oximetry_signal):
    """
    Run the OxiNet model, for AHI prediction from oximetry time series

    :param oximetry_signal: Oximetry signal. Numpy array of shape (len_signal, 1).
                            Needs to be raw data, no pre-processed.
    """

    params_path = os.path.join('saved_model', 'oxinet_config.csv')
    model_path = os.path.join('saved_model', 'duplo_1.h5')
    scaler_path = os.path.join('saved_model', 'oximetry_scaler.joblib')

    oximetry_scaler, params, model = load_(params_path, model_path, scaler_path)
    oximetry_signal = preprocess_oximetry(oximetry_signal, params)
    ahi = apply_oxinet(oximetry_signal, oximetry_scaler, params, model)[2][0][0]

    print(ahi)
    return ahi


def load_sleep_stages(sleep_stage_path, oximetry_path):
    sleep_stages = np.load(sleep_stage_path)
    oximetry_signal = np.load(oximetry_path)
    print(len(sleep_stages), len(oximetry_signal))

    if sleep_stages.shape[0] < oximetry_signal.shape[0]:
        sleep_stages = np.concatenate((sleep_stages, np.zeros(shape=oximetry_signal.shape[0] - sleep_stages.shape[0])))
    elif sleep_stages.shape[0] > oximetry_signal.shape[0]:
        sleep_stages = sleep_stages[0: oximetry_signal.shape[0]]
    assert sleep_stages.shape[0] == oximetry_signal.shape[0]

    itemindex = np.where(sleep_stages != 0)[0]

    if len(itemindex) != 1:
        oximetry_signal = oximetry_signal[itemindex[0]: itemindex[-1]]

    return sleep_stages, oximetry_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_signal', type=str, help='Name of the input recording', default='354'),
    args = parser.parse_args()

    oximetry_path = os.path.join('data', 'spo2', args.input_signal + '.npy')
    sleep_stage_path = os.path.join('data', 'sleep_stages', args.input_signal + '.npy')
    sleep_stages, oximetry = load_sleep_stages(sleep_stage_path, oximetry_path)

    run_model(oximetry)


if __name__ == '__main__':
    main()
