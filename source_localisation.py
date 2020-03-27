import glob
import os
import numpy as np
import pandas as pd
import pickle

from scipy import sparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from simulation.lead_correlate import LeadCorrelate
import simulation.metrics as met
from simulation.plot_signal import plot_sources_at_activation
from simulation.plot_signal import plot_samples_vs_score

plot_data = True


def score_on_KNeighbours(X, y, X_test, y_test, neighbours = 3):
    clf = KNeighborsClassifier(neighbours)
    model = MultiOutputClassifier(clf, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def check_data(data_dir = '.', model = None):
    # runs given model (if None KNeighbours = 3 will be used) with the data
    # with different number of max sources and different number of brain
    # parcels and plots their score depending on number of samples used.

    # number of samples selected at each run

    data_samples = np.logspace(1, 4, num=10, base=10, dtype='int')
    scores_all = pd.DataFrame(columns=['n_parcels', 'max_sources', 'scores'])

    if model is None:
        clf = KNeighborsClassifier(3)
        model = MultiOutputClassifier(clf, n_jobs=-1)

    # check for all the data directories
    for data_dir in os.listdir(data_dir):
        if data_dir[:4] == 'data':
            dir_name = data_dir.split('_')
            max_sources = dir_name[-1]
            n_parcels = dir_name[-2]
            scores = []
            print('working on %s' % (data_dir))

        if data_dir[:4] == 'data':
            X_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
            y_train = sparse.load_npz(os.path.join(data_dir,
                                                'train_target.npz')).toarray()
            X_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
            y_test = sparse.load_npz(os.path.join(data_dir,
                                    'test_target.npz')).toarray()

            for no_samples in data_samples[data_samples]: #len(X_train)]:
                no_samples_test = int(no_samples * 0.2)
                model.fit(X_train.head(no_samples),
                                    y_train[:no_samples])
                score = model.score(X_test.head(no_samples_test),
                                                y_test[:no_samples_test])
                scores.append(score)

            scores_all = scores_all.append({'n_parcels': int(n_parcels),
                               'max_sources': int(max_sources),
                               'scores': scores}, ignore_index=True)
    return scores_all, data_samples


if plot_data:
    scores_all, data_samples = check_data('.')
    plot_samples_vs_score(scores_all, data_samples)

if plot_data:
    plot_sources_at_activation(X_train, y_train)

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

        y_pred_test = lc.predict(X_test)
        y_pred_train = lc.predict(X_train)

        # y_df_test = lc.decision_function(X_test)
        # y_df_train = lc.decision_function(X_train)

        score_test = lc.score(X_test, y_test)
        score_train = lc.score(X_train, y_train)

        # calculating
        from sklearn.metrics import hamming_loss
        hl = hamming_loss(y_test, y_pred_test)

        from sklearn.metrics import jaccard_score
        js = jaccard_score(y_test, y_pred_test, average='samples')
        print('score: hamming: {:.2f}, jaccard: {:.2f}'.format(hl, js))

        from sklearn.model_selection import cross_validate
        lc2 = LeadCorrelate(L, parcel_indices_leadfield)

        from sklearn.metrics import make_scorer
        scoring = {'froc_score': make_scorer(met.froc_score,
                                             needs_threshold=True),
                   'afroc_score': make_scorer(met.afroc_score,
                                              needs_threshold=True),
                   'jaccard': make_scorer(jaccard_score,
                                          average='samples'),
                   'hamming': make_scorer(hamming_loss,
                                          greater_is_better=False)}

        scores = cross_validate(lc2, X_train, y_train, cv=3, scoring=scoring)

        # froc = met.froc_score(X_test, y_test)
        # area = met.calc_froc_area(X_test, y_test)

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
