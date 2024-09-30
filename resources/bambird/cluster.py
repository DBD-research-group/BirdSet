from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import maad
from . import config
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
import hdbscan
import warnings

warnings.simplefilter('ignore', np.RankWarning)


def _prepare_features(df_features,
                     scaler = "STANDARDSCALER",
                     features = ["shp", "centroid_f"]):
    """

    Prepare the features before clustering

    Parameters
    ----------
    df_features : pandas dataframe
        the dataframe should contain the features
    scaler : string, optional {"STANDARDSCALER", "ROBUSTSCALER", "MINMAXSCALER"}
        Select the type of scaler uses to normalize the features.
        The default is "STANDARDSCALER".
    features : list of features, optional
        List of features will be used for the clustering. The name of the features
        should be the name of a column in the dataframe. In case of "shp", "shp"
        means that all the shpxx will be used.
        The default is ["shp","centroid_f"].

    Returns
    -------
    X : pandas dataframe
        the dataframe with the normalized features

    """

    # select the scaler
    #----------------------------------------------
    if scaler == "STANDARDSCALER":
        scaler = StandardScaler()
    elif scaler == "ROBUSTSCALER":
        scaler = RobustScaler()
    elif scaler == "MINMAXSCALER":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        print("*** WARNING *** the scaler {} does not exist. StandarScaler was choosen".format(scaler))

    X = pd.DataFrame()
    f = features.copy()

    # Normalize the shapes
    if "shp" in f:
        X = df_features.loc[:, df_features.columns.str.startswith("shp")]
        X_vect = X.to_numpy()
        X_shape = X_vect.shape
        X_vect = X_vect.reshape(X_vect.size, -1)
        X_vect = scaler.fit_transform(X_vect)
        X = X_vect.reshape(X_shape)
        f.remove('shp')

    X2 = pd.DataFrame()

    # Normalize the other features (centroid, bandwidth...)
    # test if the features list is not null
    if len(f) > 0:
        # add other features like frequency centroid
        X2 = df_features[f]
        X2 = scaler.fit_transform(X2)

    return np.concatenate((X, X2), axis=1)


def find_cluster(
        df_features,
        params=config.PARAMS['PARAMS_CLUSTER'],
        display=False,
        verbose=False):
    """

    Clustering of ROIs

    We will use DBSCAN or HDSCAN clustering method for several reasons :
        * DBSCAN does not need the number of clusters to do the clustering
        * DBSCAN is able to deal with noise and keep them outside any clusters.

    So, the goal of the clustering is to aggregate similar ROIs
    which might correspond to the main call or song of a species. If several
    clusters are found, which means that we might have ROIs corresponding to
    different calls and/or songs for the species, we can keep the cluster with
    the highest number of ROIs or all the clusters.

    Parameters
    ----------
    dataset : string or pandas dataframe
        if it's a string it should be a full path to a csv file with the features
        containing a column "filename_ts" and a column "fullfilename_ts" with
        the full path to the roi
        if it's a dataframe, the dataframe should contain the features and
        a column "filename_ts" and a column "fullfilename_ts" with the full
        path to the roi.
    params : dictionnary, optional
        contains all the parameters to perform the clustering
        The default is DEFAULT_PARAMS_CLUSTER.
    save_path : string, default is None
        Path to the directory where the result of the clustering will be saved
    save_csv_filename: string, optional
        csv filename that contains all the rois with their label and cluster number
        that will be saved. The default is cluster.csv
    display : boolean, optional
        if true, display the features vectors, the eps and 2D representation of
        the DBSCAN or HDBSCAN results. The default is False.
    verbose : boolean, optional
        if true, print information. The default is False.

    Returns
    -------
    df_cluster : pandas dataframe
        Dataframe with the label found for each roi.

    """

    # drop NaN rows
    df_features = df_features.dropna(axis=0)

    if display:
        fig2, ax2 = plt.subplots(1, len(df_features.categories.unique()))
        fig2.set_size_inches(len(df_features.categories.unique()) * 6, 5)

        fig3, ax3 = plt.subplots(1, len(df_features.categories.unique()))
        fig3.set_size_inches(len(df_features.categories.unique()) * 6, 5)

        try:
            len(ax2)
        except:
            ax2 = [ax2]

        try:
            len(ax3)
        except:
            ax3 = [ax3]

    # initialize df_cluster
    df_cluster = df_features[['min_f',
                              'min_t',
                              'max_f',
                              'max_t']].copy()

    count = 0
    # test if the number of ROIs is higher than 2.
    # If not, it is impossible to cluster ROIs. It requires at least 3 ROIS
    if len(df_features) < 3:
        df_cluster["cluster_number"] = -1  # noise
        df_cluster["auto_label"] = 0  # noise

        if verbose:
            print(f"Only {len(df_features)} ROIs. It requires at least 3 ROIs to perform clustering")

        return df_cluster

    # Prepare the features of that categories
    #-------------------------------------------------------
    X = _prepare_features(df_features,
                         scaler=params['SCALER'],
                         features=params['FEATURES'])

    if display:
        # Plot the features
        ax3[count].imshow(
            X,
            interpolation="None",
            cmap="viridis",
            vmin=np.percentile(X, 10),
            vmax=np.percentile(X, 90),
        )
        ax3[count].set_xlabel("features")
        ax3[count].set_title("Shapes")

    # Select the minimum of points for a cluster
    #-------------------------------------------------------
    if params["PERCENTAGE_PTS"] is not None:
        min_points = round(params["PERCENTAGE_PTS"] / 100 * len(df_features))
    elif params["MIN_PTS"] is not None :
        min_points = round(params["MIN_PTS"])
    else :
        min_points = 2

    if min_points < 2:
        min_points = 2

    # automatic estimation of the maximum distance eps
    #-------------------------------------------------------
    if params["EPS"] == 'auto':
        # Calculate the average distance between each point in the data set and
        # its N nearest neighbors (N corresponds to min_points).
        neighbors = NearestNeighbors(n_neighbors=min_points)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)

        # Sort distance values by ascending value and plot
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # find the knee (curvature inflexion point)
        kneedle = KneeLocator(
            x=np.arange(0, len(distances), 1),
            y=distances,
            interp_method="polynomial",
            # online = False,
            # S=10,
            curve="convex",
            direction="increasing",
        )

        if display:
            # plot the distance + the knee
            ax2[count].set_xlabel("cumulative number of ROIs", fontsize=10)
            ax2[count].set_ylabel("eps", fontsize=10)
            ax2[count].axhline(y=kneedle.knee_y, xmin=0,xmax=len(distances), color="r")
            ax2[count].set_title("sorted k-dist graph", fontsize=12)
            ax2[count].plot(distances)

        # first find the maximum distance that corresponds to 95% of observations
        eps = kneedle.knee_y if kneedle.knee_y is not None else 0

        if eps == 0:
            eps = distances.max()

    # set eps manually
    #-------------------------------------------------------
    else :
        eps = params["EPS"]

    # find the number of clusters and the rois that belong to the cluster
    #--------------------------------------------------------------------
    if params["METHOD"] == "DBSCAN":
        cluster = DBSCAN(eps=eps, min_samples=min_points).fit(X)

        if verbose:
            print(f"DBSCAN eps {eps} min_points {min_points} Number of soundtypes found for {len(df_features)} : {np.unique(cluster.labels_).size}")

    elif params["METHOD"] == "HDBSCAN":
        cluster = hdbscan.HDBSCAN(
            min_cluster_size=min_points,
            min_samples=round(min_points / 2),
            cluster_selection_epsilon=float(eps),
            # cluster_selection_method = 'leaf',
            #allow_single_cluster=True,
        ).fit(X)

        if verbose:
            print(f"HDBSCAN eps {eps} min_points {min_points} Number of soundtypes found for {len(df_features)} : {np.unique(cluster.labels_).size}")

    #with pd.option_context('mode.chained_assignment', None):
        # add the cluster label into the label's column of the dataframe
    df_cluster["cluster_number"] = cluster.labels_.reshape(-1, 1)

    # add the automatic label (SIGNAL = 1 or NOISE = 0) into the auto_label's column of
    # the dataframe
    # Test if we want to consider only the biggest or all clusters
    # that are not noise (-1) to be signal
    if params["KEEP"] == 'BIGGEST':
        # set by default to 0 the auto_label of all
        df_cluster["auto_label"] = 0
        # find the cluster ID of the biggest cluster that is not noise
        try:
            biggest_cluster_ID = df_cluster.loc[df_cluster["cluster_number"] >= 0]["cluster_number"].value_counts().idxmax()
            # set by default to 1 the auto_label of the biggest cluster
            df_cluster.loc[df_cluster["cluster_number"] == biggest_cluster_ID, "auto_label"] = int(1)
        except:
            # if there is only noise
            pass
    elif params["KEEP"] == 'ALL':
        # set by to 0 the auto_label of the noise (cluster ID = -1)
        df_cluster.loc[df_cluster["cluster_number"] < 0, "auto_label"] = int(0)
        # set by to 1 the auto_label of the signal (cluster ID >= 0)
        df_cluster.loc[df_cluster["cluster_number"] >= 0, "auto_label"] = int(1)

    count += 1

    return df_cluster


def overlay_rois(cluster,
                 wave,
                 column_labels ='cluster_number',
                 params=config.PARAMS['PARAMS_EXTRACT'],
                 verbose=False,
                 **kwargs):

    color_labels=['tab:red', 'tab:green', 'tab:orange', 'tab:blue',
                  'tab:purple','tab:pink','tab:brown','tab:olive',
                  'tab:cyan','tab:gray','yellow']

    df_cluster = cluster.copy(deep=True)
    unique_labels = np.sort(df_cluster.cluster_number.unique())
    if verbose :
        print('\n')
        print('============== OVERLAY ROIS ON THE ORIGINAL FILE ==============\n')

    df_cluster['label'] = df_cluster[column_labels]

    sig = wave

    fig = plt.figure(figsize=kwargs.pop("figsize", ((len(wave)/params["SAMPLE_RATE"])/60*15, 7)))

    ax0 = plt.subplot2grid((5, 1), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)

    maad.util.plot_wave(s=sig, fs=params["SAMPLE_RATE"], ax=ax0, now = False)

    # 2. compute the spectrogram
    Sxx, tn, fn, ext = maad.sound.spectrogram(
        sig,
        params["SAMPLE_RATE"],
        nperseg=params["NFFT"],
        noverlap=params["NFFT"] // 2,
    )

    df_single_file = maad.util.format_features(df_cluster, tn, fn)

    # 3.
    # Convert in dB
    X = maad.util.power2dB(Sxx, db_range=96) + 96

    kwargs.update({"vmax": np.max(X)})
    kwargs.update({"vmin": np.min(X)})
    kwargs.update({"extent": ext})
    maad.util.plot_spectrogram(X,
                          log_scale=False,
                          colorbar=False,
                          ax=ax1,
                          now = False,
                          **kwargs)

    # 4.
    if unique_labels is None:
        unique_labels = list(df_cluster[column_labels].unique())

    # overlay
    maad.util.overlay_rois(im_ref=X,
                          rois = df_single_file,
                          ax=ax1,
                          fig=fig,
                          unique_labels=unique_labels,
                           edge_color=color_labels,
                          textbox_label=True)

    fig.tight_layout()
    plt.show()

    return

