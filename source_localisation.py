import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn import linear_model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from simulation.lead_correlate import LeadCorrelate
from simulation.sparse_regressor import SparseRegressor
import simulation.metrics as met


from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split

if os.environ.get('DISPLAY'):  # display exists
    from simulation.plot_signal import plot_sources_at_activation
    visualize_data = True
    N_JOBS = 1
else:
    # running on the server, no display
    visualize_data = False
    N_JOBS = -1


plot_data = True
# data_dir = 'data_CC120008_26_3'
data_dir = 'data_sample_26_3'

def learning_curve(X, y, model=None, model_name=''):
    # runs given model (if None KNeighbours = 3 will be used) with the data
    # with different number of max sources and different number of brain
    # parcels and plots their score depending on number of samples used.

    # number of samples selected at each run

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    n_samples_grid = np.logspace(1, np.log10(len(X_train)),
                                 num=10, base=10, dtype='int')
    scores_all = pd.DataFrame(columns=['n_samples_train', 'score_test'])

    if model is None:
        clf = KNeighborsClassifier(3)
        model = MultiOutputClassifier(clf, n_jobs=N_JOBS)
        model_name = 'Kneighbours3'

    for n_samples_train in n_samples_grid:
        model.fit(X_train.head(n_samples_train), y_train[:n_samples_train])
        score = model.score(X_test, y_test)
        scores_all = scores_all.append({'n_samples_train': n_samples_train,
                                        'score_test': score},
                                       ignore_index=True)
    n_parcels = int(y_train.shape[1])
    max_sources = int(y_train.sum(axis=1).max())

    scores_all['n_parcels'] = n_parcels
    scores_all['max_sources'] = max_sources
    scores_all['model_name'] = model_name
    scores_all['model'] = str(model)

    return scores_all


def load_data(data_dir):
    lead_matrix = np.load(os.path.join(data_dir, 'lead_field.npz'))
    parcel_indices_leadfield = lead_matrix['parcel_indices']
    signal_type = lead_matrix['signal_type']
    L = lead_matrix['lead_field']

    X = pd.read_csv(os.path.join(data_dir, 'X.csv'))
    y = sparse.load_npz(os.path.join(data_dir, 'target.npz')).toarray()
    return X, y, L, parcel_indices_leadfield, signal_type

'''
X, y, L, parcel_indices_leadfield, signal_type = load_data(data_dir)

if plot_data and visualize_data:
    plot_sources_at_activation(X, y, signal_type)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

lc = LeadCorrelate(L, parcel_indices_leadfield)
lc.fit(X_train, y_train)

y_pred_test = lc.predict(X_test)
y_pred_train = lc.predict(X_train)


# calculating
hl = hamming_loss(y_test, y_pred_test)
js = jaccard_score(y_test, y_pred_test, average='samples')
print('score: hamming: {:.2f}, jaccard: {:.2f}'.format(hl, js))

scoring = {'froc_score': make_scorer(met.froc_score,
                                     needs_threshold=True),
           'afroc_score': make_scorer(met.afroc_score,
                                      needs_threshold=True),
           'jaccard': make_scorer(jaccard_score,
                                  average='samples'),
           'hamming': make_scorer(hamming_loss,
                                  greater_is_better=False)}

scores = cross_validate(lc, X_train, y_train, cv=3, scoring=scoring)

scores = pd.DataFrame(scores)
scores[['test_%s' % s for s in scoring]]
print(scores.agg(['mean', 'std']))
'''

# Do learning curve for all models and all datasets
scores_all = []

# data_dirs = sorted(glob.glob('data_*'))
data_dirs = [data_dir]
for idx, data_dir in enumerate(data_dirs):
    print('{}/{} processing {} ... '.format(idx+1, len(data_dirs), data_dir))
    X, y, L, parcel_indices, signal_type = load_data(data_dir)

    lc = LeadCorrelate(L, parcel_indices)
    lasso_lars = SparseRegressor(L, parcel_indices, linear_model.LassoLarsCV())
    lasso = SparseRegressor(L, parcel_indices, linear_model.LassoCV())
    models = {'': None, 'lead correlate': lc, 'lasso lars': lasso_lars}
    # models = {'lasso lars': lasso_lars}

    for name, model in models.items():
        scores_all.append(learning_curve(X, y, model=model, model_name=name))

scores_all = pd.concat(scores_all, axis=0)
scores_all.to_pickle("scores_all.pkl")

diff_parcels = scores_all['n_parcels'].unique()
fig, ax = plt.subplots(nrows=len(diff_parcels), ncols=1)
for cond, df in scores_all.groupby(['n_parcels', 'max_sources', 'model_name',
                                    'model']):
    sub = np.where(diff_parcels == cond[0])[0][0]

    if type(ax) == np.ndarray:
        ax[sub].plot(df.n_samples_train, df.score_test,
                     label=str(cond[1])+cond[2])
    else:
        ax.plot(df.n_samples_train, df.score_test,
                     label=str(cond[1])+cond[2])
for idx, parcel in enumerate(diff_parcels):
    if type(ax) == np.ndarray:
        ax[idx].set(xlabel='n_samples_train', ylabel='score',
                    title='Parcels: '+str(parcel))
    else:
        ax.set(xlabel='n_samples_train', ylabel='score',
                    title='Parcels: '+str(parcel))
    plt.legend()
plt.tight_layout()
plt.savefig('figs/learning_curves.png')
