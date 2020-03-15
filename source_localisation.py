import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load train data
X_train = pd.read_csv(os.path.join('data', 'train.csv'))
y_train = sparse.load_npz(os.path.join('data', 'train_target.npz')).toarray()

# Visualize
if 0:
    import mne  # noqa
    data_path = mne.datasets.sample.data_path()
    fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    info = mne.read_evokeds(fname)[0].pick('eeg').info
    n_classes = y_train.shape[1]
    fig, axes = plt.subplots(5, n_classes, figsize=(16, 4))

    for k in range(n_classes):
        X_k = X_train.iloc[np.argmax(y_train, axis=1) == k]
        for i, ax in enumerate(axes[:, k]):
            mne.viz.plot_topomap(X_k.iloc[i].values, info, axes=ax)
    plt.tight_layout()
    plt.show()

clf = KNeighborsClassifier(3)
multi_target_ridge = MultiOutputClassifier(clf, n_jobs=-1)
multi_target_ridge.fit(X_train, y_train)

# Load test data
X_test = pd.read_csv(os.path.join('data', 'test.csv'))
y_test = sparse.load_npz(os.path.join('data', 'test_target.npz')).toarray()
print(multi_target_ridge.score(X_test, y_test))
