import os
import numpy as np
import pandas as pd
import pickle
# import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from simulation.lead_correlate import LeadCorrelate
import simulation.metrics as met

# Load train data
# X_train = pd.read_csv(os.path.join('data', 'train.csv'))
# y_train = sparse.load_npz(os.path.join('data', 'train_target.npz')).toarray()

'''
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
'''

clf = KNeighborsClassifier(3)
model = MultiOutputClassifier(clf, n_jobs=-1)
# model.fit(X_train, y_train)

# Load test data
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
        y_test = sparse.load_npz(os.path.join(data_dir,
                                 'test_target.npz')).toarray()

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
    plt.plot(data_samples[:len(scores_all[s])], scores_all[s],
             label = s + ' parcels')
plt.legend()
plt.xlabel('number of samples used')
plt.ylabel('score (on Kneighours)')
plt.savefig('score.png')
'''

y_test_score = []
y_train_score = []
max_parcels_all = []
for data_dir in os.listdir('.'):
    if 'data_15_2' in data_dir:
        max_parcels = data_dir[8:]
        lead_matrix = np.load(os.path.join(data_dir, 'lead_field.npz'))
        parcel_indices_leadfield = lead_matrix['parcel_indices']
        L = lead_matrix['lead_field']

        X_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        y_train = sparse.load_npz(os.path.join(data_dir,
                                  'train_target.npz')).toarray()
        X_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        y_test = sparse.load_npz(os.path.join(data_dir,
                                 'test_target.npz')).toarray()

        with open(os.path.join(data_dir, 'labels.pickle'), 'rb') as outfile:
            labels = pickle.load(outfile)

        lc = LeadCorrelate(L, parcel_indices_leadfield)
        lc.fit(X_train, y_train)

        y_pred1 = lc.predict(X_test)
        y_pred2 = lc.predict(X_train)
        # plotFROC()

        # y_pred = lc.predict(X_train)
        # y_pred2 = lc.predict(X_test)

        # score_test = lc.score(X_test, y_test)
        # score_train = lc.score(X_train, y_train)

        # calculating
        from sklearn.metrics import hamming_loss
        hl = hamming_loss(y_test, y_pred1)

        from sklearn.metrics import jaccard_score
        js = jaccard_score(y_test, y_pred1, average='samples')
        print('score: hamming: {:.2f}, jaccard: {:.2f}'.format(hl, js))

        from sklearn.model_selection import cross_validate
        lc2 = LeadCorrelate(L, parcel_indices_leadfield)
        # cross_val = cross_validate(lc2, X_train, y_train, cv=3)
        # print('cross validation (smaller the better): {}'.format(cross_val))

        from sklearn.metrics import make_scorer
        scoring = {'froc_score': make_scorer(met.froc_score),
                   'afroc_score': make_scorer(met.afroc_score),
                   'jaccard': make_scorer(jaccard_score)}

        cross_validate(lc2, X_train, y_train, cv=3,
                       scoring=scoring)

        froc = met.froc_score(X_test, y_test)
        area = met.calc_froc_area(X_test, y_test)

        # y_test_score.append(score_test)
        # y_train_score.append(score_train)
        # max_parcels_all.append(max_parcels)

'''
plt.figure()
plt.plot(max_parcels_all, y_test_score, 'ro')
plt.plot(max_parcels_all, y_train_score, 'ro')
plt.xlabel('max parcels')
plt.ylabel('score (avg #errors/sample/max parcels): higher is worse')

plt.title('Results for 15 parcels')
plt.show()
'''
