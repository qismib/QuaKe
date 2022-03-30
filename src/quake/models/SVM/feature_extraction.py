import logging
from sklearn.preprocessing import MinMaxScaler
from ..CNN.CNN_dataloading import load_data
from ..CNN.CNN_data_preprocessing import prepare, normalize
from .load_CNN import load_cnn
from quake import PACKAGE

import numpy as np

logger = logging.getLogger(PACKAGE + ".SVM")


def extract_feats(data_folder, opts, extra, detector):
    logger.info("Loading data ...")
    sig, bkg = load_data(data_folder, detector)
    logger.info("Preparing data ...")
    data, labels = prepare(sig, bkg, detector)
    feature_layer, tr_map, val_map, te_map = load_cnn(data_folder)
    feat_size = opts["feature_number"]

    """
    Making batches for better memory allocation
    """
    nbatches = 100
    data_feats = np.zeros((0, feat_size))
    for i in range(0, labels.shape[0], nbatches):
        input = [
            data[0][i : i + nbatches],
            data[1][i : i + nbatches],
            data[2][i : i + nbatches],
        ]
        data_feats = np.vstack((data_feats, feature_layer(input).numpy()))

    if extra:
        f = extrafeatures(data)
        data_feats = np.hstack((data_feats, f))

    """ Rescaling features from 0 to 1 seems to lower performances
    scaler = MinMaxScaler()
    scaler.fit(data_feats)
    data_feats = scaler.transform(data_feats)
    """

    s_tr = data_feats[tr_map]
    s_val = data_feats[val_map]
    s_te = data_feats[te_map]
    l_tr = labels[tr_map]
    l_val = labels[val_map]
    l_te = labels[te_map]

    s_tr, l_tr = outlier_removal(s_tr, l_tr)

    return s_tr, l_tr, s_val, l_val, s_te, l_te


def extrafeatures(data):
    f = np.zeros((data[0].shape[0], 2))
    for i, particle in enumerate(data[0]):
        f[i, 0] = np.argwhere(particle).shape[0]
        f[i, 1] = particle[:, :, 0].sum()
    return f


def outlier_removal(s_tr, l_tr):
    means = np.mean(s_tr, axis=0)
    stds = np.std(s_tr, axis=0)
    outmask = (s_tr - means) / stds < 3.5
    om = np.ones(l_tr.shape[0])
    for i in range(0, means.shape[0]):
        om = np.logical_and(om, outmask[:, i])
    return s_tr[om], l_tr[om]
