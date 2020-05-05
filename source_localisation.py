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


# data_dir = 'all' if all directories starting with 'data_' should be simulated
# otherwise give name of the directory
# e.g data_dir = 'data/data_grad_sample_26_3' or 'all'
data_dir = 'data/data_grad_all_26_3'
signal_type = 'grad'


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

    # TODO: remove this line
    n_samples_grid = n_samples_grid[n_samples_grid < 5000]
    for n_samples_train in n_samples_grid:
        # for test use either all test samples or n_samples_train
        n_samples_test = min(len(X_test), n_samples_train)
        print('fitting {} using {} train samples, {} test samples'.format(
              model_name, n_samples_train, n_samples_test))
        model.fit(X_train.head(n_samples_train), y_train[:n_samples_train])

        score = model.score(X_test.head(n_samples_test),
                            y_test[:n_samples_test])
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
    # find all the files with lead_field
    # lead_matrix = np.load(os.path.join(data_dir, 'lead_field.npz'))
    lead_field_files = os.path.join(data_dir, '*lead_field.npz')
    lead_field_files = sorted(glob.glob(lead_field_files))
    subject_name = data_dir.split('_')[2]

    assert len(lead_field_files) >= 1

    parcel_indices_leadfield, L = [], []
    subj_dict = {}
    for idx, lead_file in enumerate(lead_field_files):
        lead_matrix = np.load(lead_file)

        if subject_name == 'all':
            lead_file = os.path.basename(lead_file)
            subj_dict[lead_file.split('_')[0]] = idx
        else:
            subj_dict[subject_name] = idx
        parcel_indices_leadfield.append(lead_matrix['parcel_indices'])
        L.append(lead_matrix['lead_field'])
        assert parcel_indices_leadfield[idx].shape[0] == L[idx].shape[1]
    signal_type = lead_matrix['signal_type']

    assert len(parcel_indices_leadfield) == len(L) == idx + 1
    assert len(subj_dict) >= 1  # at least a single subject

    X = pd.read_csv(os.path.join(data_dir, 'X.csv'))

    if subject_name == 'all':
        X['subject'] = X['subject'].map(subj_dict)
    else:
        X['subject'] = idx
    X.astype({'subject': 'int32'}).dtypes
    y = sparse.load_npz(os.path.join(data_dir, 'target.npz')).toarray()

    # Scale data to avoid tiny numbers
    X.loc[:, X.columns != 'subject'] /= np.max(X.loc[:,
                                                     X.columns != 'subject'])
    assert y.shape[0] == X.shape[0]
    return X, y, L, parcel_indices_leadfield, signal_type


def calc_scores_for_leadcorrelate(data_dir, plot_data=False):
    X, y, L, parcel_indices_leadfield, signal_type = load_data(data_dir)

    if plot_data:
        plot_sources_at_activation(X, y, signal_type)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    lc = LeadCorrelate(L, parcel_indices_leadfield)
    lc.fit(X_train, y_train)

    y_pred_test = lc.predict(X_test)
    # y_pred_train = lc.predict(X_train)

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


def make_learning_curve_for_all(data_dir):
    # Do learning curve for all models and all datasets
    scores_all = []

    if data_dir == 'all':
        data_dir = 'data/data_' + signal_type + '_*'
        data_dirs = sorted(glob.glob(data_dir))
        data_dir_all = 'data/data_' + signal_type + '_all'
        [data_dirs.remove(all_file) for all_file in data_dirs if data_dir_all
         in all_file]
    else:
        data_dirs = [data_dir]

    for idx, data_dir in enumerate(data_dirs):
        print('{}/{} processing {} ... '.format(idx+1,
                                                len(data_dirs), data_dir))
        subject = data_dir.split('_')[2]
        X, y, L, parcel_indices, signal_type_data = load_data(data_dir)
        assert signal_type == signal_type_data

        lc = LeadCorrelate(L, parcel_indices)
        lasso_lars = SparseRegressor(L, parcel_indices,
                                     linear_model.LassoLarsCV(max_iter=10,
                                                              n_jobs=N_JOBS))
        # lasso = SparseRegressor(L, parcel_indices, linear_model.LassoCV())
        # models = {'': None, 'lead correlate': lc, 'lasso lars': lasso_lars}
        models = {'lead correlate': lc, 'lasso lars': lasso_lars}
        # models = {'lasso lars': lasso_lars}

        for name, model in models.items():
            score = learning_curve(X, y, model=model, model_name=name)
            score['subject'] = subject
            scores_all.append(score)

    scores_all = pd.concat(scores_all, axis=0)
    scores_all.to_pickle("scores_all.pkl")  # TODO: save in another location


# plot the results from all the calculated data
def plot_scores(scores_all, file_name='learning_curves', ext='.png'):
    diff_parcels = scores_all['n_parcels'].unique()
    fig, ax = plt.subplots(nrows=len(diff_parcels), ncols=1)
    for cond, df in scores_all.groupby(['n_parcels', 'max_sources',
                                        'model_name', 'model']):
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
    plt.savefig('figs/' + file_name + ext)


plot_data = False
calc_scores_for_leadcorrelate(data_dir, (plot_data and visualize_data))

# make_learning_curve_for_all(data_dir)


if plot_data:
    scores_all = pd.read_pickle("scores_all.pkl")
    plot_scores(scores_all, file_name='learning_curves', ext='.png')

    # plot the results for each subject separately
    for subject in np.unique(scores_all['subject']):
        scores = scores_all[scores_all['subject'] == subject]
        plot_scores(scores, file_name='learning_curves_'+subject)
