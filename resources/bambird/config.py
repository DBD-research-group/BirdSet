#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions to load the configuration file
"""
#
# Authors:  Felix MICHAUD   <felixmichaudlnhrdt@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

# %%
# general packages
import sys
import os

# basic packages
import yaml

# %%

RANDOM_SEED = 1979  # Fix the random seed to be able to repeat the results

PARAMS_XC = {
    'PARAM_XC_LIST': ['len:"20-60"', 'q:">C"', 'type:"song"'],
    'NUM_FILES': 20,
    'CSV_XC_FILE': 'xc_metadata.csv'
}
PARAMS_EXTRACT = {
    #"FUNC": bambird.extract_rois_full_sig,
    # Extract Audio resampling
    "SAMPLE_RATE": 32000,  # Sampling frequency in Hz
    # Audio preprocess
    "LOW_FREQ": 250,  # Low frequency in Hz of the bandpass filter applied to the audio
    "HIGH_FREQ": 13000,  # High frequency in Hz of the bandpass filter applied to the audio
    # butterworth filter order to select the bandwidth corresponding to the ROI
    "BUTTER_ORDER": 1,
    # Max duration of the audio files that we will use to compute the features
    #"AUDIO_DURATION": 60,
    # Split the audio signal of chunk with duration = SIGNAL LENGTH (in second)
    "CHUNK_DURATION": 10,
    "OVLP": 0.05,  # Define the overlap ratio between each chunk
    # Spectrogram
    # Mode to compute the remove_background ('mean', 'median')
    "MODE_RMBCKG": "median",
    # Number of points used to compute the running mean of the noise profil
    "N_RUNNING_MEAN": 10,
    "NFFT": 1024,  # Number of points of the spectrogram
    # Combination of parameters for ROIs extraction
    "MASK_PARAM1": 26,  # 30 37
    "MASK_PARAM2": 10,  # 20 33
    # Select and merge bbox parameters
    "MAX_RATIO_YX": 7,  # ratio Y/X between the high (Y in px) and the width (X in px) of the ROI
    "MIN_DURATION": 0.5,  # minimum event duration in s
    "MARGIN_T_LEFT": 0.2,
    "MARGIN_T_RIGHT": 0.2,
    "MARGIN_F_TOP": 250,
    "MARGIN_F_BOTTOM": 250,
    # save parameters
    "MARGIN_T": 0.1,  # time margin in s around the ROI
    "MARGIN_F": 250,  # frequency margin in Hz around the ROI
    # butterworth filter order to select the bandwidth corresponding to the ROI
    "FILTER_ORDER": 5,
}


PARAMS_FEATURES = {
    # Extract Audio resampling
    "SAMPLE_RATE": 32000,  # Sampling frequency in Hz
    # Audio preprocess
    "LOW_FREQ": 250,  # Low frequency in Hz of the bandpass filter applied to the audio
    "HIGH_FREQ": 13000,  # High frequency in Hz of the bandpass filter applied to the audio
    # butterworth filter order to select the bandwidth corresponding to the ROI
    "BUTTER_ORDER": 5,
    # Spectrogram
    "NFFT": 1024,
    # Number of points of the spectrogram
    "SHAPE_RES": "high",
}

PARAMS_CLUSTER = {
    "FEATURES": ['shp', 'centroid_f', 'peak_f', 'duration_t', 'bandwidth_f', 'bandwidth_min_f', 'bandwidth_max_f'],
    # choose the features used to cluster {'shp', 'centroid_f', 'peak_f', 'duration_t', 'bandwidth_f', 'bandwidth_min_f', 'bandwidth_max_f', 'min_f', 'max_f' }
    "PERCENTAGE_PTS": 5,
    # minimum number of ROIs to form a cluster (in % of the total number of ROIs) {number between 0 and 1 or blank}
    "MIN_PTS": None,  # minimum number of ROIs to form a cluster {integer or blank}
    "METHOD": "DBSCAN",  # HDBSCAN or DBSCAN
    "SCALER": "MINMAXSCALER",  # STANDARDSCALER or ROBUSTSCALER or MINMAXSCALER
    "KEEP": "BIGGEST",  # ALL or BIGGEST
    "EPS": "auto"  # set the maximum distance between elements in a single clusters {a number or 'auto'}
}

PARAMS = {
    'RANDOM_SEED': RANDOM_SEED,
    'PARAMS_XC': PARAMS_XC,
    'PARAMS_EXTRACT': PARAMS_EXTRACT,
    'PARAMS_FEATURES': PARAMS_FEATURES,
    'PARAMS_CLUSTER': PARAMS_CLUSTER
}

# %%

""" ===========================================================================

                    Private functions 

============================================================================"""


def _fun_call_by_name(val):
    if '.' in val:
        module_name, fun_name = val.rsplit('.', 1)
        # Restrict which modules may be loaded here to avoid safety issue
        # Put the name of the module
        assert module_name.startswith('bambird')
    else:
        module_name = '__main__'
        fun_name = val
    try:
        __import__(module_name)
    except ImportError:
        raise ("Can''t import {} while constructing a Python object".format(val))
    module = sys.modules[module_name]
    fun = getattr(module, fun_name)
    return fun


def _fun_constructor(loader, node):
    val = loader.construct_scalar(node)
    print(val)
    return _fun_call_by_name(val)


def _get_loader():
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!FUNC", _fun_constructor)
    return loader


""" ===========================================================================

                    Public function 

============================================================================"""


def load_config(fullfilename=None):
    """
    Load the configuration file to set all the parameters of bambird

    Parameters
    ----------
    fullfilename : string, optional
        Path to the configuration file.
        if no valid configuration file is given, the parameters are set to the
        default values.

    Returns
    -------
    PARAMS : dictionary
        Dictionary with all the parameters that are required for the bambird's
        functions
    """

    global PARAMS
    global RANDOM_SEED
    global PARAMS_XC
    global PARAMS_EXTRACT
    global PARAMS_FEATURES
    global PARAMS_CLUSTER

    if os.path.isfile(str(fullfilename)):
        with open(fullfilename) as f:
            PARAMS = yaml.load(f, Loader=_get_loader())
            RANDOM_SEED = PARAMS['RANDOM_SEED']
            PARAMS_XC = PARAMS['PARAMS_XC']
            PARAMS_EXTRACT = PARAMS['PARAMS_EXTRACT']
            PARAMS_FEATURES = PARAMS['PARAMS_FEATURES']
            PARAMS_CLUSTER = PARAMS['PARAMS_CLUSTER']
    else:
        print("The config file {} could not be loaded. Default parameters are loaded".format(fullfilename))

    return PARAMS


def get_config():
    PARAMS = {
        'RANDOM_SEED': RANDOM_SEED,
        'PARAMS_XC': PARAMS_XC,
        'PARAMS_EXTRACT': PARAMS_EXTRACT,
        'PARAMS_FEATURES': PARAMS_FEATURES,
        'PARAMS_CLUSTER': PARAMS_CLUSTER
    }
    return PARAMS


