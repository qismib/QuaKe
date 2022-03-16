import logging
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import numpy as np

from .CNN_dataloading import load_data
from .CNN_data_preprocessing import prepare
from .CNN_data_preprocessing import tr_val_te_split
from .CNN_Net import buildCNN
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score

frac = lambda l, n: np.sum(l == n)/len(l)

accuracy = lambda y, l: np.sum(1*y == l)/l.shape[0]
sensitivity = lambda y, l: np.sum(1*np.logical_and(y == 1, l == 1))/np.sum(1*l == 1)
specificity = lambda y, l: np.sum(1*np.logical_and(y == 0, l == 0))/np.sum(1*l == 0)

def CNN_train(data_folder: Path, opts, seed):
    batch_size = opts["batch_size"]
    epochs = opts["epochs"]

    print("Loading data ...")
    sig, bkg = load_data(data_folder)
    print("Preparing data ...")
    data, labels = prepare(sig, bkg, opts) 
    s_tr, l_tr, s_val, l_val, s_te, l_te, idx_tr, idx_val, idx_te = tr_val_te_split(data, labels, opts, seed)

    print("%i samples for train (%0.3f) (%0.3f signal, %0.3f background)" 
    %(len(l_tr), len(l_tr)/len(labels), frac(l_tr, 1), frac(l_tr, 0)))
    print("%i samples for validation (%0.3f) (%0.3f signal, %0.3f background)" 
    %(len(l_val), len(l_val)/len(labels), frac(l_val, 1), frac(l_val, 0)))
    print("%i samples for test (%0.3f) (%0.3f signal, %0.3f background)" 
    %(len(l_te), len(l_te)/len(labels), frac(l_te, 1), frac(l_te, 0)))
   
    model = buildCNN(s_tr, opts)
    model.summary()

    print("Training the CNN")
    model.fit(
        s_tr, to_categorical(l_tr), batch_size=batch_size, 
        epochs=epochs, validation_data=(s_val, to_categorical(l_val)), verbose=1)

    scores = model.predict(s_te)
    y_prob = scores[:,1]
    y = np.argmax(scores, axis = 1)

    results = [accuracy(y, l_te), sensitivity(y, l_te),
    specificity(y, l_te), roc_auc_score(l_te,y_prob)]

    metrics = ['accuracy', 'sensitivity', 'specificity', 'auc']
    print("Performances on test set:")
    for i, m in enumerate(metrics):
        print(m, results[i])
    
    print("Saving model as 'CNN.h5'")
    model.save(str(data_folder.parent)+"/models/cnn/CNN.h5")
    print("Saving train-validation-test partition for the SVM model")
    np.savetxt(str(data_folder.parent)+"/models/cnn/train_map", idx_tr, fmt='%i')
    np.savetxt(str(data_folder.parent)+"/models/cnn/validation_map", idx_val, fmt='%i')
    np.savetxt(str(data_folder.parent)+"/models/cnn/test_map", idx_te, fmt='%i')