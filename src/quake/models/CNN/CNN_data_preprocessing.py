import logging
import numpy as np
from sklearn.model_selection import train_test_split
from quake import PACKAGE

logger = logging.getLogger(PACKAGE + ".CNN")


def normalize(data):  # Not in use at the moment
    for i, particle in enumerate(data):
        data[i] = particle / particle.sum()
    return data


def roll_crop(d, crop):
    for i, particle in enumerate(d):
        c = np.argwhere(particle)
        if list(c):
            cm1 = np.min(c[:, 0])
            cm2 = np.min(c[:, 1])
            d[i] = np.roll(d[i], -cm1, axis=0)
            d[i] = np.roll(d[i], -cm2, axis=1)
    d = d[:, 0 : crop[0], 0 : crop[1]]
    return d


def prepare(sig, bkg, detector):
    res = np.array(detector["resolution"])
    dx = np.concatenate((sig[0], bkg[0]), axis=0)
    dy = np.concatenate((sig[1], bkg[1]), axis=0)
    dz = np.concatenate((sig[2], bkg[2]), axis=0)
    hist_crop = (20 / res).astype(int)
    dx = roll_crop(dx, hist_crop[[1, 2]])
    dy = roll_crop(dy, hist_crop[[0, 2]])
    dz = roll_crop(dz, hist_crop[[0, 1]])

    nsig = sig[0].shape[0]
    nbkg = bkg[0].shape[0]

    labels = np.concatenate([np.ones(nsig), np.zeros(nbkg)])

    data = [dx, dy, dz]

    return data, labels


def tr_val_te_split(data, labels, opts, seed):
    test_size = opts["val_te_ratio"]
    dx = data[0]
    dy = data[1]
    dz = data[2]
    ntot = dx.shape[0]
    idx = np.arange(0, ntot)

    dx_idx = np.vstack((dx.reshape(ntot, -1).T, idx)).T
    logger.info(data[0].shape)

    dx_tr, dx_vt, l_tr, l_vt = train_test_split(
        dx_idx, labels, test_size=2 * test_size, random_state=seed
    )
    dx_val, dx_te, l_val, l_te = train_test_split(
        dx_vt, l_vt, test_size=0.5, random_state=seed
    )

    idx_tr = dx_tr[:, -1].astype(int)
    idx_val = dx_val[:, -1].astype(int)
    idx_te = dx_te[:, -1].astype(int)

    shape = np.array(dx.shape[1:]).astype(int)

    dx_tr = dx_tr[:, :-1].reshape(np.append(dx_tr.shape[0], shape))
    dx_val = dx_val[:, :-1].reshape(np.append(dx_val.shape[0], shape))
    dx_te = dx_te[:, :-1].reshape(np.append(dx_te.shape[0], shape))

    dy_tr, dz_tr = dy[idx_tr], dz[idx_tr]
    dy_val, dz_val = dy[idx_val], dz[idx_val]
    dy_te, dz_te = dy[idx_te], dz[idx_te]

    s_tr = [dx_tr, dy_tr, dz_tr]
    s_val = [dx_val, dy_val, dz_val]
    s_te = [dx_te, dy_te, dz_te]

    return s_tr, l_tr, s_val, l_val, s_te, l_te, idx_tr, idx_val, idx_te
