import logging
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import numpy as np

from quake import PACKAGE
from .CNN_dataloading import load_data
from .CNN_data_preprocessing import prepare
from .CNN_data_preprocessing import tr_val_te_split
from .CNN_Net import buildCNN
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(PACKAGE + ".CNN")

frac = lambda l, n: np.sum(l == n)/len(l)

accuracy = lambda y, l: np.sum(y == l)/l.shape[0]
sensitivity = lambda y, l: np.sum(np.logical_and(y == 1, l == 1))/np.sum(l == 1)
specificity = lambda y, l: np.sum(np.logical_and(y == 0, l == 0))/np.sum(l == 0)

def CNN_train(data_folder: Path, opts, seed):
    batch_size = opts["batch_size"]
    epochs = opts["epochs"]

    logger.info("Loading data ...")
    sig, bkg = load_data(data_folder)

    logger.info("Preparing data ...")
    data, labels = prepare(sig, bkg, opts) 

    s_tr, l_tr, s_val, l_val, s_te, l_te, idx_tr, idx_val, idx_te = tr_val_te_split(data, labels, opts, seed)
    ntot = len(l_tr) + len(l_val) + len(l_te)

    logger.info(f"{len(l_tr)} samples for train ("f"{len(l_tr)/ntot:.1%} of total) ("
                f"{frac(l_tr, 1):.1%} signal, "f"{frac(l_tr, 0):.1%} background)")
    logger.info(f"{len(l_val)} samples for validation ("f"{len(l_val)/ntot:.1%} of total) ("
                f"{frac(l_val, 1):.1%} signal, "f"{frac(l_val, 0):.1%} background)")
    logger.info(f"{len(l_te)} samples for test ("f"{len(l_te)/ntot:.1%} of total) ("
                f"{frac(l_te, 1):.1%} signal, "f"{frac(l_te, 0):.1%} background)")
   
    model = buildCNN(s_tr, opts)
    model.summary()

    logger.info("Training the CNN")
    model.fit(
        s_tr, to_categorical(l_tr), batch_size=batch_size, 
        epochs=epochs, validation_data=(s_val, to_categorical(l_val)), verbose=1)

    scores = model.predict(s_te)
    y_prob = scores[:,1]
    y = np.argmax(scores, axis = 1)

    results = [accuracy(y, l_te), sensitivity(y, l_te),
    specificity(y, l_te), roc_auc_score(l_te,y_prob)]

    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'auc']
    logger.info("Performances on test set:")
    for i, m in enumerate(metrics):
        logger.info(f"{m}"f"{results[i]:{10}.{3}}")
    
    logger.info("Saving model as 'CNN.h5'")
    model.save(str(data_folder.parent)+"/models/cnn/CNN.h5")
    logger.info("Saving train-validation-test partition for the SVM model")
    np.savetxt(str(data_folder.parent)+"/models/cnn/train_map", idx_tr, fmt='%i')
    np.savetxt(str(data_folder.parent)+"/models/cnn/validation_map", idx_val, fmt='%i')
    np.savetxt(str(data_folder.parent)+"/models/cnn/test_map", idx_te, fmt='%i')
