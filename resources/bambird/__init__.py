from . import segment
from . import config
from . import features
from . import cluster


def findCluster(wave, params=config.PARAMS, display=False):

    df_rois = segment.extract_rois_full_sig(
        sig=wave,
        params=params["PARAMS_EXTRACT"],
        display=False,
        verbose=False
    )

    df_features = features.compute_multiple_features(
        df_rois=df_rois,
        sig=wave,
        params=params["PARAMS_FEATURES"],
        verbose=False
    )

    df_cluster = cluster.find_cluster(
        df_features=df_features,
        params=params["PARAMS_CLUSTER"],
        display=False,
        verbose=False
    )

    if display:
        cluster.overlay_rois(df_cluster, wave)

    return df_cluster
