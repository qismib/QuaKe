import numpy as np
from sklearn.model_selection import train_test_split

def normalize(data, dim):
    if dim == 2:
        return data/data.sum(axis = (1,2))[:, None, None]
    elif dim == 3:
        return data/data.sum(axis = (1,2,3))[:, None, None, None]
    else:
        return -1

def prepare(sig, bkg, opts):
    dim = opts["dim"]
    data = np.concatenate((sig, bkg), axis = 0)
    #data[data < 1e-3] = 0 
    for i, particle in enumerate(data):
        c = np.argwhere(particle)
        if list(c):
            cmx = np.min(c[:,0])
            cmy = np.min(c[:,1])
            cmz = np.min(c[:,2])
            data[i] = np.roll(data[i], -cmx , axis = 0)
            data[i] = np.roll(data[i], -cmy , axis = 1)
            data[i] = np.roll(data[i], -cmz , axis = 2)
    data = data[:, 0:4, 0:4, 0:20]  

    nsig = sig.shape[0]
    nbkg = sig.shape[0]

    labels = np.concatenate([np.ones(nsig), np.zeros(nbkg)])

    if dim == 2:
        dx = np.expand_dims(data.sum(axis = 1), 3)
        dy = np.expand_dims(data.sum(axis = 2), 3)
        data = np.concatenate([dx, dy], axis = 3)
        data = data.reshape(nsig+nbkg, 4, 20, 2)
    elif dim == 3:
        data = np.expand_dims(data, axis = 4)
        print(data.shape)
    return data, labels

def tr_val_te_split(data, labels, opts, seed):
    test_size = opts["val_te_ratio"]
    val_size = test_size/(1-test_size)
    ntot = data.shape[0]
    idx = np.arange(0, ntot)

    data_idx = np.vstack((data.reshape(ntot, -1).T, idx)).T
    print(data.shape)

    s_tv, s_te, l_tv, l_te = train_test_split(data_idx, labels, test_size = test_size, random_state=seed)
    s_tr, s_val, l_tr, l_val = train_test_split(s_tv, l_tv, test_size = val_size, random_state=seed)

    idx_tr = s_tr[:,-1].astype(int)
    idx_val = s_val[:,-1].astype(int)
    idx_te = s_te[:,-1].astype(int)

    if data.ndim == 4:
        shape = np.array([4,20,2]).astype(int)
    elif data.ndim == 5:
        shape = np.array([4, 4, 20, 1]).astype(int)

    s_tr = s_tr[:, :-1].reshape(np.append(s_tr.shape[0], shape))
    s_val = s_val[:, :-1].reshape(np.append(s_val.shape[0], data.shape[1:]))
    s_te = s_te[:, :-1].reshape(np.append(s_te.shape[0], data.shape[1:]))

    return s_tr, l_tr, s_val, l_val, s_te, l_te, idx_tr, idx_val, idx_te