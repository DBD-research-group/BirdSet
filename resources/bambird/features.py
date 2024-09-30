from . import config
import maad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_features(sig,
                     params=config.PARAMS['PARAMS_FEATURES'],
                     display=False):
    """
    Compute features such as shape (wavelets), centroid and bandwidth

    Parameters
    ----------
    audio_path : string
        full path the audio file
    params : dictionnary, optional
        contains all the parameters to compute the features
    display : boolean, optional
        if true, display the spectrograms and the signals at each step of
        the process. The default is False.
    verbose : boolean, optional
        if true, print information. The default is False.

    Returns
    -------
    df_features : pandas dataframe
        dataframe with all the features computed for each roi found in the audio.
    """

    # 2.   Bandpass the audio
    # 3.   Compute the spectrogram
    # 4.   Convert to dB
    # 5.   Remove background noise
    # 6.   Compute the features shape of the ROIs
    # 7.   Compute the centroid and bandwidth of the ROIs
    # 8.   Add the features into a dataframe

    plt.style.use("default")

    # 2. bandpass filter around birds frequencies
    fcut_max = min(params["HIGH_FREQ"], params["SAMPLE_RATE"] // 2 - 1)
    fcut_min = params["LOW_FREQ"]
    sig = maad.sound.select_bandwidth(sig,
                                params["SAMPLE_RATE"],
                                fcut=[fcut_min, fcut_max],
                                forder=params["BUTTER_ORDER"],
                                fname="butter",
                                ftype="bandpass")

    # 3. compute the spectrogram
    Sxx, tn, fn, ext = maad.sound.spectrogram(sig,
                                         params["SAMPLE_RATE"],
                                         nperseg=params["NFFT"],
                                         noverlap=params["NFFT"] // 2)

    # set to zero the DC value (= the first row of the spectrogram)
    Sxx[0, :] = 0
    Sxx[1, :] = 0

    if display:
        # prepare 4 figures in a row
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        # plot the spectrogram into a bigger plot full of zeros with fixed
        # size in order to compare the spectrograms
        WIDTH_T = 20  # in s. Fix the width of the spectrogram.
        # width in pixels
        width_x = round(WIDTH_T / (tn[1] - tn[0]))
        # zero padding before and after the spectrogram if the duration time
        # is below width_x
        if int(round((width_x - Sxx.shape[1]) / 2)) > 0 :
            margin = np.zeros((Sxx.shape[0],
                              int(round((width_x - Sxx.shape[1]) / 2))), int)
            Sxx_to_display = np.concatenate((margin, Sxx, margin), axis=1)
        else :
            Sxx_to_display = Sxx

        maad.util.plot_spectrogram(Sxx_to_display,
                              extent=[0, WIDTH_T, ext[2], ext[3]],
                              ax=ax[0, 0],
                              title="original ROI",
                              now=False)

    # 4. convert to dB
    Sxx_dB = maad.util.power2dB(Sxx, db_range=96) + 96

    # 5. Clean spectrogram : remove background
    Sxx_clean_dB, _, _ = maad.sound.remove_background(Sxx_dB)
    Sxx_clean_dB = Sxx_clean_dB * (Sxx_clean_dB > 6)

    if display:
        maad.util.plot_spectrogram(
            np.concatenate((margin, Sxx_clean_dB, margin), axis=1),
            # [left, right, bottom, top]
            extent=[0, WIDTH_T, ext[2], ext[3]],
            log_scale=False,
            ax=ax[0, 1],
            title="cleaned ROI",
            now=False)

    # initialize a dataframe with ROIs with the coordinate of the
    # bounding box
    # !!! it does not work when using the original bbox values found during
    # the segmentation. This might be due to the different resolution which
    # creates pixel shift.

    # test if we can find 0 pixels. If not, set the min_y to 0 and max_y to
    # the vertical size of the spectrogram
    if (len(np.where(maad.sound.avg_amplitude_spectro(Sxx_clean_dB) != 0)[0])) <= 2:
        min_y = 0
        max_y = Sxx_clean_dB.shape[1]
    else:
        min_y = np.where(maad.sound.avg_amplitude_spectro(Sxx_clean_dB) != 0)[0][2]  # 2 to avoid to take indice 0 or 1
        max_y = np.where(maad.sound.avg_amplitude_spectro(Sxx_clean_dB) != 0)[0][-1]

    df_rois_for_shape = pd.DataFrame([[min_y, 0, max_y, Sxx.shape[1] - 1]],
                                     columns=["min_y", "min_x", "max_y", "max_x"])

    # 6. Compute acoustic features (MAAD)
    df_shape, params_shape = maad.features.shape_features(Sxx_clean_dB,
                                                     resolution=params["SHAPE_RES"],
                                                     rois=df_rois_for_shape)

    if display:
        maad.util.plot_shape(df_shape, params_shape, ax=ax[1, 0])

    # Keep columns ['min_y', 'min_x', 'max_y', 'max_x'] in case of these
    # features are not in the original dataset. If there are already
    # in the original dataset, these columns will be deleted and we keep
    # the original columns ['min_y', 'min_x', 'max_y', 'max_x'] found during
    # the extraction of the ROIs
    df_shape = maad.util.format_features(df_shape, tn, fn)

    # 7. Compute the centroid (t,f), the bandwidth and the duration of the ROI
    # initialize the dataframe df_rois with duration and centroid from the
    # size of the spectrogram
    df_rois = pd.DataFrame([[tn[-1], tn[-1] / 2]], columns=["duration_t", "centroid_t"])

    # compute the mean spectrum and smooth the spectrum with a running mean
    Sxx_clean = maad.util.dB2power(Sxx_clean_dB)
    mean_spectrum = maad.util.running_mean(maad.sound.avg_power_spectro(Sxx_clean), N=35)

    # Convert the spectrum in dB
    mean_spectrum_dB = maad.util.power2dB(mean_spectrum)

    # find the frequency position and amplitude corresponding to the
    # frequency peak
    f_peak = fn[np.argmax(mean_spectrum_dB)]
    f_peak_amplitude_dB = np.max(mean_spectrum_dB)

    # find the frequencies (- and +) at -6dB
    index = [i for i, x in enumerate(mean_spectrum_dB) if x > (f_peak_amplitude_dB-6)]
    f_peak_minus_6dB = fn[index[0]]
    f_peak_plus_6dB = fn[index[-1]]

    # add the values to the dataframe
    df_rois["peak_f"] = f_peak
    df_rois["centroid_f"] = (f_peak_plus_6dB + f_peak_minus_6dB) / 2
    df_rois["bandwidth_f"] = f_peak_plus_6dB - f_peak_minus_6dB
    df_rois["bandwidth_min_f"] = f_peak_minus_6dB
    df_rois["bandwidth_max_f"] = f_peak_plus_6dB
    df_rois["snr"] = maad.util.power2dB(np.percentile(mean_spectrum, 99))

    if display:
        maad.util.plot_spectrum(mean_spectrum_dB, fn, ax=ax[1, 1])
        ax[1, 1].axvline(x=f_peak_minus_6dB, color="red", linestyle="dotted", linewidth=2)
        ax[1, 1].axvline(x=f_peak_plus_6dB, color="red", linestyle="dotted", linewidth=2)
        ax[1, 1].plot([f_peak], [f_peak_amplitude_dB], "r+")
        ax[1, 1].plot([(f_peak_plus_6dB + f_peak_minus_6dB) /2],[f_peak_amplitude_dB], "bo")

    # 8. concat df_rois and df_shape
    df_features = pd.concat([df_rois, df_shape], axis=1)

    return df_features


def compute_multiple_features(df_rois,
                              sig,
                              params=config.PARAMS['PARAMS_FEATURES'],
                              verbose=False):
    """
    Parameters
    ----------
    df_rois : string or pandas dataframe
        if it's a string it should be either a directory where are the ROIs
        files to process or a full path to a csv file containing a column
        "filename_ts" and a column "fullfilename_ts" with the full path to the
        ROIS files to process
        if it's a dataframe, the dataframe should contain a column
        "filename_ts" and a column "fullfilename_ts" with the full path to the audio
        files to process. This dataframe can be obtained by called the function
        grab_audio_to_df
    params : dictionnary, optional
        contains all the parameters to compute the features
    save_path : string, default is None
        Path to the directory where the csv file with the features will be saved
    save_csv_filename: string, optional
        csv filename that contains all the features that will be saved. The default
        is None, meaning that the name will be automatically created from the
        parameters to compute the features as :
            'features_'
            + params["SHAPE_RES"]
            + "_NFFT"
            + str(params["NFFT"])
            + "_SR"
            + str(params["SAMPLE_RATE"])
            + ".csv"
    overwrite : boolean, optional
        if a directory already exists with the rois, if false, the process is
        aborted, if true, new features will eventually be added in the directory and
        in the csv file.
    nb_cpu : integer, optional
        number of cpus used to compute the features. The default is None which means
        that all cpus will be used
    verbose : boolean, optional
        if true, print information. The default is False.

    Returns
    -------
    df_features_sorted : pandas dataframe
        dataframe containing all the features computed for each roi.
    csv_fullfilename : string
        full path the csv file with all the features computed for each roi.
        if the file already exists, the new features will be appended to the file.
    """

    df_rois = df_rois.copy()
    if len(df_rois) == 0:
        return df_rois

    # if necessary compute features using multicpu
    # test if the dataframe contains files to compute the features
    if verbose:
        print('Composition of the dataset : ')
        print(f'number of files: {len(df_rois)}2.0f')

    l = []
    sr = params["SAMPLE_RATE"]
    for idx, row in df_rois.iterrows():
        wav = sig[int(row["min_t"]*sr): int(row["max_t"]*sr)]
        f = compute_features(wav, params)
        l.append(f)

    df_features = pd.concat(l, ignore_index=True)
    # Merge the result df_features with the df_rois

    df_rois.reset_index(inplace=True)
    df_features.reset_index(inplace=True)

    common_columns = df_features.columns.intersection(df_rois.columns)

    # Drop common columns from df_features
    df_features = df_features.drop(columns=common_columns)

    # Merge the DataFrames based on the index
    df_features = pd.merge(df_rois, df_features, how="outer", left_index=True, right_index=True)

    df_features.drop(columns=["index"], inplace=True)
    return df_features
