import logging
from ..CNN.CNN_dataloading import load_data
from ..CNN.CNN_data_preprocessing import prepare
from .load_CNN import load_cnn
from quake import PACKAGE

import numpy as np

logger = logging.getLogger(PACKAGE + ".SVM")

def extract_feats(data_folder, opts, extra):
    logger.info("Loading data ...")
    sig, bkg = load_data(data_folder)
    logger.info("Preparing data ...")
    data, labels = prepare(sig, bkg, opts)
    feature_layer, tr_map, val_map, te_map = load_cnn(data_folder)
    feat_size = opts["feature_number"]
    
    # Making batches for better memory allocation
    nbatches = 100
    data_feats = np.zeros((0, feat_size))
    for i in range(0, data.shape[0], nbatches):
        data_feats = np.vstack((data_feats, feature_layer(data[i:i+nbatches]).numpy()))

    if extra:
        f = extrafeatures(data, opts)
        data_feats = np.hstack((data_feats, f))
        
    s_tr = data_feats[tr_map]
    s_val = data_feats[val_map]
    s_te = data_feats[te_map]
    l_tr = labels[tr_map]
    l_val = labels[val_map]
    l_te = labels[te_map]

    return s_tr, l_tr, s_val, l_val, s_te, l_te

def extrafeatures(data, opts):
    dim = opts["dim"]
    f = np.zeros((data.shape[0], 2))
    for i, particle in enumerate(data):
        f[i, 0] = np.argwhere(particle).shape[0]
        if dim == 2:
            f[i, 1] = particle[:,:,0].sum()
        elif dim == 3:
            f[i, 1] = particle.sum()
    return f


