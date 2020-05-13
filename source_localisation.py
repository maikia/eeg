import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn import linear_model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from simulation.lead_correlate import LeadCorrelate
from simulation.parcels import dist_calc
from simulation.sparse_regressor import SparseRegressor
import simulation.metrics as met


from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split

if os.environ.get('DISPLAY'):  # display exists
    from simulation.plot_signal import plot_sources_at_activation
    from simulation.plot_signal import plot_y_pred_true_parcels
    visualize_data = True
    N_JOBS = 1
else:
    # running on the server, no display
    visualize_data = False
    N_JOBS = -1


def calc_distance_matrix(data_dir, subjects):
    # calculates distance matrix
    # TODO: subdivide to multiple functions
    import nibabel as nib
    idx = 0

    for subject in subjects:
        # only left hemisphere for now
        base_dir = 'mne_data/MNE-sample-data/subjects/' + subject
        surf_lh = nib.freesurfer.read_geometry(os.path.join(base_dir,
                                               'surf/lh.pial'))
        surf_rh = nib.freesurfer.read_geometry(os.path.join(base_dir,
                                               'surf/rh.pial'))
        labels_x = np.load(os.path.join(data_dir, subject + '_labels.npz'),
                               allow_pickle=True)

        labels_x = labels_x['arr_0']
        labels_x_lh = [s for s in labels_x if s.hemi == 'lh']
        labels_x_rh = [s for s in labels_x if s.hemi == 'rh']
        distance_matrix_lh = dist_calc(surf=surf_lh, cortex=None,
                                   source_nodes=labels_x_lh, dist_type = "min")
        distance_matrix_rh = dist_calc(surf=surf_rh, cortex=None,
                                   source_nodes=labels_x_rh, dist_type = "min")
        save_path = os.path.join(data_dir, subject + '_dist_matrix.npz')
        np.savez(save_path, dist_matrix_lh=distance_matrix_lh,
                            dist_matrix_rh=distance_matrix_rh)
    '''
    # TO plot:
    import pdb; pdb.set_trace()
    mne_dir = mne.get_config('MNE_DATASETS_SAMPLE_PATH')
    data_path = mne_dir + '/MNE-sample-data'
    # subjects_dir = os.path.join(data_path, 'subjects')
    subjects_dir = 'mne_data/MNE-sample-data/subjects/'
    hemi = 'both'
    brain = Brain(subject, hemi, 'inflated', subjects_dir=subjects_dir,
                  cortex='low_contrast', background='white')
    idx = 0
    distance_matrix_lh_norm = distance_matrix_lh/np.max(distance_matrix_lh)
    brain.add_label(labels_x_lh[idx], alpha=1, color='blue') #str(1 -
    (distance_matrix_lh_norm[0, idx])))
    '''

def calc_distance_matrix_subject():
    pass

def calc_dist_matrix_labels():
    pass

def display_true_pred_parcels(X, y, data_dir, model, model_name='',
                              n_samples='all'):
    # draw a brain with y_pred in red and y_test in green

    if model_name == 'K-neighbours(3)':
        print('this function does not work with k-neighbours')
        return

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    if n_samples != 'all':
        len_train, len_test = len(X_train), len(X_test)
        X_train = X_train.head(min(n_samples, len_train))
        y_train = y_train[:min(n_samples, len_train)]
        X_test = X_test.head(min(n_samples, len_test))
        y_test = y_test[:min(n_samples, len_test)]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = [[] for i in range(len(np.unique(X_test['subject'])))]

    for idx, (y_p, y_t) in enumerate(zip(y_pred, y_test)):
        x = X_test.iloc[idx]
        subject = x['subject']
        subject_id = x['subject_id']

        if len(labels[subject_id]) == 0:
            labels_x = np.load(os.path.join(data_dir, subject + '_labels.npz'),
                               allow_pickle=True)
            labels_x = labels_x['arr_0']
            labels[subject_id] = labels_x

        idx_lab_pred = labels[subject_id][y_p == 1]
        idx_lab_true = labels[subject_id][y_t == 1]

        plot_y_pred_true_parcels(subject, idx_lab_pred, idx_lab_true)


def learning_curve(X, y, model=None, model_name='', n_samples_grid='auto'):
    # runs given model with the data
    # with different number of max sources and different number of brain
    # parcels and plots their score depending on number of samples used.

    # number of samples selected at each run
    if model_name == 'K-neighbours(3)':
        X = X.loc[:, X.columns != 'subject']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    if n_samples_grid == 'auto':
        n_samples_grid = np.logspace(1, np.log10(len(X_train)),
                                     num=10, base=10, dtype='int')

    scores_all = pd.DataFrame(columns=['n_samples_train', 'score_test'])

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
        X['subject_id'] = X['subject'].map(subj_dict)
    else:
        X['subject'] = subject_name
        X['subject id'] = idx

    X.astype({'subject_id': 'int32'}).dtypes
    y = sparse.load_npz(os.path.join(data_dir, 'target.npz')).toarray()

    # Scale data to avoid tiny numbers
    X.iloc[:, :-2] /= np.max(X.iloc[:, :-2])
    assert y.shape[0] == X.shape[0]
    return X, y, L, parcel_indices_leadfield, signal_type


def calc_scores_for_model(X, y, model, n_samples=-1):
    '''
    TODO: add doc
    '''
    print('calculating various scores for the model')
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    if n_samples > -1:
        # use only subset of the data
        X_train = X_train[:min(len(X_train), n_samples)]
        X_test = X_test[:min(len(X_test), n_samples)]
        y_train = y_train[:min(len(y_train), n_samples)]
        y_test = y_test[:min(len(y_test), n_samples)]

    model.fit(X_train, y_train)

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

    scores = cross_validate(model, X_train, y_train, cv=3, scoring=scoring)

    scores = pd.DataFrame(scores)
    scores[['test_%s' % s for s in scoring]]
    print(scores.agg(['mean', 'std']))
    return scores


def make_learning_curve_for_all(X, y, models):
    # Do learning curve for all models and all datasets
    scores_all = []

    for name, model in models.items():
        score = learning_curve(X, y, model=model, model_name=name,
                               n_samples_grid=n_samples_grid)

        scores_all.append(score)

    scores_all = pd.concat(scores_all, axis=0)
    return scores_all


# plot the results from all the calculated data
def plot_scores(scores_all, file_name='learning_curves', ext='.png'):
    diff_parcels = scores_all['n_parcels'].unique()
    fig, ax = plt.subplots(nrows=len(diff_parcels), ncols=1)
    for cond, df in scores_all.groupby(['n_parcels', 'max_sources',
                                        'model_name', 'model']):
        sub = np.where(diff_parcels == cond[0])[0][0]

        if type(ax) == np.ndarray:
            ax[sub].plot(df.n_samples_train, df.score_test,
                         label=str(cond[1]) + cond[2])
        else:
            ax.plot(df.n_samples_train, df.score_test,
                    label=str(cond[1]) + cond[2])
    for idx, parcel in enumerate(diff_parcels):
        if type(ax) == np.ndarray:
            ax[idx].set(xlabel='n_samples_train', ylabel='score',
                        title='Parcels: ' + str(parcel))
        else:
            ax.set(xlabel='n_samples_train', ylabel='score',
                   title='Parcels: ' + str(parcel))
        plt.legend()
    plt.tight_layout()
    plt.savefig('figs/' + file_name + ext)


if __name__ == "__main__":
    plot_data = False
    calc_scores_for_lc = False
    calc_learning_rate = False

    data_dir = 'data/data_grad_all_26_3'
    signal_type = 'grad'

    # n_samples_grid = 'auto'
    n_samples_grid = [300]
    subject = data_dir.split('_')[-3]

    # load data
    print('processing {} ... '.format(data_dir))

    X, y, L, parcel_indices, signal_type_data = load_data(data_dir)
    assert signal_type == signal_type_data

    # define models
    # Lasso lars
    model = make_pipeline(
            StandardScaler(with_mean=False),
            linear_model.LassoLarsCV(max_iter=3, n_jobs=N_JOBS,
                                     normalize=False, fit_intercept=False)
        )

    lasso_lars = SparseRegressor(L, parcel_indices, model)
    # lasso = SparseRegressor(L, parcel_indices, linear_model.LassoCV())

    # Lead COrrelate
    lc = LeadCorrelate(L, parcel_indices)

    # K-means
    clf = KNeighborsClassifier(3)
    kneighbours = MultiOutputClassifier(clf, n_jobs=N_JOBS)

    if calc_scores_for_lc:
        # calculate various scores for Lead Correlate model
        if n_samples_grid != 'auto':
            n_samples = n_samples_grid[-1]
            # n_samples = 10
        else:
            n_samples = -1
        calc_scores_for_model(X, y, model=lc, n_samples=n_samples)

    scores_save_file = os.path.join(data_dir, "scores_all.pkl")
    if calc_learning_rate:
        # make learning curve for selected models
        models = {'lead correlate': lc, 'lasso lars': lasso_lars,
                  'K-neighbours(3)': kneighbours}
        scores_all = make_learning_curve_for_all(X, y, models)
        scores_all.to_pickle(scores_save_file)

        print(scores_all.tail(len(models)))

    plot_data = plot_data and visualize_data
    if plot_data:
        # plot sources at the activation
        plot_sources_at_activation(X, y, signal_type)

    if plot_data:
        # plot scores
        scores_all = pd.read_pickle(scores_save_file)
        plot_scores(scores_all, file_name='learning_curves', ext='.png')

    if plot_data:
        # plot parcels
        display_true_pred_parcels(X, y, data_dir, model=lasso_lars,
                                  model_name='lasso lars',
                                  n_samples=300)

    calc_distance_matrix(data_dir, np.unique(X['subject']))
