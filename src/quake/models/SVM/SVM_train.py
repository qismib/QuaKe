import logging
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from quake import PACKAGE

from .feature_extraction import extract_feats

logger = logging.getLogger(PACKAGE + ".CNN")

frac = lambda l, n: np.sum(l == n)/len(l)

accuracy = lambda y, l: np.sum(y == l)/l.shape[0]
sensitivity = lambda y, l: np.sum(np.logical_and(y == 1, l == 1))/np.sum(l == 1)
specificity = lambda y, l: np.sum(np.logical_and(y == 0, l == 0))/np.sum(l == 0)

def SVM_train(data_folder: Path, opts, extra):
    np.set_printoptions(precision=3)

    logger.info("Loading data and extracting features from CNN:")
    s_tr, l_tr, s_val, l_val, s_te, l_te = extract_feats(data_folder, opts, extra)
    classical_kernels = ['linear', 'poly', 'rbf']

    ntot = len(l_tr) + len(l_val) + len(l_te)
    s_tr_svm, l_tr_svm = train_test_split(s_tr, l_tr, train_size = 0.1, random_state = 42)[::2]

    logger.info(f"{len(l_tr_svm)} samples for train ("f"{len(l_tr_svm)/ntot:.1%} of total) ("
                f"{frac(l_tr_svm, 1):.1%} signal, "f"{frac(l_tr_svm, 0):.1%} background)")
    logger.info(f"{len(l_val)} samples for validation ("f"{len(l_val)/ntot:.1%} of total) ("
                f"{frac(l_val, 1):.1%} signal, "f"{frac(l_val, 0):.1%} background)")
    logger.info(f"{len(l_te)} samples for test ("f"{len(l_te)/ntot:.1%} of total) ("
                f"{frac(l_te, 1):.1%} signal, "f"{frac(l_te, 0):.1%} background)")

    s = [s_tr, s_val, s_te]
    l = [l_tr, l_val, l_te]

    k_len = len(classical_kernels)

    acc = np.zeros((k_len, 3))
    sen = np.zeros((k_len, 3))
    spec = np.zeros((k_len, 3))
    auc = np.zeros((k_len, 3))

    logger.info("Training, validating, testing SVMs with linear, poly, rbf kernels ...")

    feature_size = opts["feature_number"]
    logger.info("Using "f"{feature_size} features extracted from CNN + Total event energy + Nhits")

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

    logger.info("Metrics matrices. Rows: linear, poly, rbf. Columns: train, validation, test")
    logger.info("Accuracy: \n"f"{acc}")
    logger.info("Sensitivity: \n"f"{sen}")
    logger.info("Specificity: \n"f"{spec}")
    logger.info("AUC: \n"f"{auc}")  