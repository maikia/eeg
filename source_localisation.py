import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load train data
# X_train = pd.read_csv(os.path.join('data', 'train.csv'))
# y_train = sparse.load_npz(os.path.join('data', 'train_target.npz')).toarray()

# Visualize
if 0:
    import mne  # noqa
    fig_dir = 'figs'
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

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
    plt.save(os.path.join(fig_dir, 'visualize.png'))
    plt.show()

clf = KNeighborsClassifier(3)
multi_target_ridge = MultiOutputClassifier(clf, n_jobs=-1)
# multi_target_ridge.fit(X_train, y_train)

## Load test data
# X_test = pd.read_csv(os.path.join('data', 'test.csv'))
# y_test = sparse.load_npz(os.path.join('data', 'test_target.npz')).toarray()
# print(multi_target_ridge.score(X_test, y_test))

data_samples = np.logspace(1,4,num=10,base=10,dtype='int')

# check for all the data directories
'''
import glob
scores_all = {}
for data_dir in os.listdir('.'):
    if data_dir[:4] == 'data':
        scores = []
        print('working on %s' % (data_dir))
        no_parcels = int(data_dir[5:])

        X_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        y_train = sparse.load_npz(os.path.join(data_dir,
                                            'train_target.npz')).toarray()
        X_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        y_test = sparse.load_npz(os.path.join(data_dir, 'test_target.npz')).toarray()

        for no_samples in data_samples[data_samples < 4641]: #len(X_train)]:
            no_samples_test = int(no_samples * 0.2)
            multi_target_ridge.fit(X_train.head(no_samples),
                                   y_train[:no_samples])
            score = multi_target_ridge.score(X_test.head(no_samples_test),
                                             y_test[:no_samples_test])
            scores.append(score)
        scores_all[str(no_parcels)] = scores

import matplotlib.pylab as plt
plt.figure()

for s in scores_all.keys():
    plt.plot(data_samples[:len(scores_all[s])], scores_all[s], label = s + ' parcels')
plt.legend()
plt.xlabel('number of samples used')
plt.ylabel('score (on Kneighours)')
plt.savefig('score.png')
'''

data_dir = 'data_15'
L = np.load(os.path.join(data_dir, 'lead_field.npz'))
L = L['arr_0']

X_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
y_train = sparse.load_npz(os.path.join(data_dir,
                                            'train_target.npz')).toarray()
X_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
y_test = sparse.load_npz(os.path.join(data_dir, 'test_target.npz')).toarray()

x = X_train.iloc[[0]]
pd.DataFrame(L.T @ x.T).groupby(parcel_indices).argmax()
# look for the parcel that has the source with the highest correlation with
# the data


