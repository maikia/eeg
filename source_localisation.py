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
model = MultiOutputClassifier(clf, n_jobs=-1)
# model.fit(X_train, y_train)

## Load test data
# X_test = pd.read_csv(os.path.join('data', 'test.csv'))
# y_test = sparse.load_npz(os.path.join('data', 'test_target.npz')).toarray()
# print(model.score(X_test, y_test))

data_samples = np.logspace(1, 4, num=10, base=10, dtype='int')

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
            model.fit(X_train.head(no_samples),
                                   y_train[:no_samples])
            score = model.score(X_test.head(no_samples_test),
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

import pickle
with open(os.path.join(data_dir, 'labels.pickle'), 'rb') as outfile:
    #pickle.dump(parcel_vertices, outfile)
    labels = pickle.load(outfile)

# reading forward matrix and saving
import mne
data_path = mne.datasets.sample.data_path()
subject = 'sample'
fwd_fname = os.path.join(data_path, 'MEG', subject,
                         subject + '_audvis-meg-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
lead_field = fwd['sol']['data']

# now we make a vector of size n_vertices for each surface of cortex
# hemisphere and put a int for each vertex that says it which label
# it belongs to.
parcel_indices_lh = np.zeros(len(fwd['src'][0]['inuse']), dtype=int)
parcel_indices_rh = np.zeros(len(fwd['src'][1]['inuse']), dtype=int)
for label_name, label_idx in labels.items():
    label_id = int(label_name[:-3])
    if '-lh' in label_name:
        parcel_indices_lh[label_idx] = label_id
    else:
        parcel_indices_rh[label_idx] = label_id

# Make sure label numbers different for each hemisphere
#parcel_indices_rh[parcel_indices_rh != 0] += np.max(parcel_indices_lh)
parcel_indices = np.concatenate((parcel_indices_lh,
                                 parcel_indices_rh), axis=0)

# Now pick vertices that are actually used in the forward
inuse = np.concatenate((fwd['src'][0]['inuse'],
                        fwd['src'][1]['inuse']), axis=0)

parcel_indices_leadfield = parcel_indices[np.where(inuse)[0]]

assert len(parcel_indices_leadfield) == L.shape[1]

# take one sample and look at the column of L that is the most
# correlated with it. The predict the label idx of the max column.
score = 0
for idx in range(0, len(X_train)): #enumerate(X_train.iterrows()):
    x = X_train.iloc[[idx]]
    y_pred = pd.DataFrame(L.T @ x.T).groupby(parcel_indices_leadfield).max().idxmax().values[0]
    y_true = np.where(y_train[idx])[0][0] + 1
    print('True: %d : Pred %d' %(y_true, y_pred))

    if y_pred == y_true:
        # predicted correctly
        score += 1

final_score = score/(idx+1)
print('Score: %f ' %final_score)

# look for the parcel that has the source with the highest correlation with
# the data


