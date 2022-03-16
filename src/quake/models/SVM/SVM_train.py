import logging
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


from .feature_extraction import extract_feats

frac = lambda l, n: 100*np.sum(l == n)/len(l)

accuracy = lambda y, l: np.sum(1*y == l)/l.shape[0]
sensitivity = lambda y, l: np.sum(1*np.logical_and(y == 1, l == 1))/np.sum(1*l == 1)
specificity = lambda y, l: np.sum(1*np.logical_and(y == 0, l == 0))/np.sum(1*l == 0)

def SVM_train(data_folder: Path, opts, extra):
    print("Loading data and extracting features from CNN:")
    s_tr, l_tr, s_val, l_val, s_te, l_te = extract_feats(data_folder, opts, extra)
    classical_kernels = ['linear', 'poly', 'rbf']

    ntot = len(l_tr) + len(l_val) + len(l_te)
    s_tr_svm, l_tr_svm = train_test_split(s_tr, l_tr, train_size = 0.1, random_state = 42)[::2]

    print("%i samples for train (%0.1f %% of total) (%0.1f signal, %0.1f background)" 
    %(len(l_tr_svm), 100*len(l_tr_svm)/ntot, frac(l_tr_svm, 1), frac(l_tr_svm, 0)))
    print("%i samples for validation (%0.1f %% of total) (%0.1f signal, %0.1f background)" 
    %(len(l_val), 100*len(l_val)/ntot, frac(l_val, 1), frac(l_val, 0)))
    print("%i samples for test (%0.1f %% of total) (%0.1f signal, %0.1f background)" 
    %(len(l_te), 100*len(l_te)/ntot, frac(l_te, 1), frac(l_te, 0)))

    s = [s_tr, s_val, s_te]
    l = [l_tr, l_val, l_te]

    k_len = len(classical_kernels)

    acc = np.zeros((k_len, 3))
    sen = np.zeros((k_len, 3))
    spec = np.zeros((k_len, 3))
    auc = np.zeros((k_len, 3))

    print("Training, validating, testing SVMs with linear, poly, rbf kernels ...")

    if extra:
        print("Using %i features extracted from CNN + Total event energy + Nhits" %(s_tr.shape[1])-2)
    else:
        print("Using %i features extracted from CNN" %(s_tr.shape[1]))

    for k, kernel in enumerate(classical_kernels):
        classical_svc = SVC(kernel=kernel, probability=True)
        classical_svc.fit(s_tr_svm, l_tr_svm)
        for j in range(0, 3):
            y = classical_svc.predict(s[j])
            acc[k, j] = accuracy(y, l[j])
            sen[k, j] = sensitivity(y, l[j])
            spec[k, j] = specificity(y, l[j])
            y_prob = classical_svc.predict_proba(s[j])[:,1]
            auc[k, j] = roc_auc_score(l[j],y_prob)

    np.set_printoptions(precision=3)
    print("Metrics matrices. Rows: linear, poly, rbf. Columns: train, validation, test")
    print("Accuracy: \n", acc)
    print("Sensitivity: \n", sen)
    print("Specificity: \n", spec)
    print("AUC: \n", auc)  