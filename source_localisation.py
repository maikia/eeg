import os
import glob

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn import linear_model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split

from simulation.lead_correlate import LeadCorrelate
from simulation.parcels import find_shortest_path_between_hemi
from simulation.sparse_regressor import SparseRegressor, ReweightedLasso
import simulation.metrics as met

if os.environ.get('DISPLAY'):  # display exists
    from simulation.plot_signal import plot_sources_at_activation
    from simulation.plot_signal import plot_y_pred_true_parcels
    from simulation.plot_signal import plot_distance
    visualize_data = True
    N_JOBS = 1
else:
    # running on the server, no display
    visualize_data = False
    N_JOBS = -1


def calc_distance_matrix(data_dir, subjects):
    # calculates distance matrix

    for subject in subjects:
        # only left hemisphere for now
        save_path_lh = os.path.join(data_dir, subject + '_dist_matrix_lh.csv')
        save_path_rh = os.path.join(data_dir, subject + '_dist_matrix_rh.csv')
        if os.path.exists(save_path_lh) and os.path.exists(save_path_rh):
            pass
            # continue
        else:
            print('calculating distance matrix for {}'.format(subject))

        dist_matrix = find_shortest_path_between_hemi(data_dir, subject)
        dist_matrix_lh, dist_matrix_rh = dist_matrix

        # np.savez(save_path, dist_matrix_lh=distance_matrix_lh,
        #                    dist_matrix_rh=distance_matrix_rh)

        dist_matrix_lh.to_csv(save_path_lh)
        dist_matrix_rh.to_csv(save_path_rh)


def display_distances_on_brain(data_dir, subject='CC110033'):

    calc_distance_matrix(data_dir, [subject])
    labels_x = np.load(os.path.join(data_dir, subject + '_labels.npz'),
                       allow_pickle=True)['arr_0']
    plot_distance(subject, data_dir, labels_x)


def plot_all_parcels(data_dir, subject):
    labels_x = np.load(os.path.join(data_dir,
                                    subject + '_labels.npz'),
                       allow_pickle=True)

    labels_x = labels_x['arr_0']
    plot_y_pred_true_parcels(subject, labels_x, [])


def display_true_pred_parcels(X, y, data_dir, model, model_name='',
                              n_samples='all'):
    # draw a brain with y_pred in red and y_test in green

    if model_name == 'K-neighbours(3)':
        print('this function does not work with display true pred parcels')
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
            labels_x = np.load(os.path.join(data_dir,
                                            subject + '_labels.npz'),
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
    print(n_samples_grid)
    scores_all = pd.DataFrame(columns=['n_samples_train', 'score_test'])

    for n_samples_train in n_samples_grid:
        # for test use either all test samples or n_samples_train
        n_samples_test = min(len(X_test), n_samples_train)
        print('fitting {} using {} train samples, {} test samples'.format(
              model_name, n_samples_train, n_samples_test))

        model.fit(X_train.head(n_samples_train), y_train[:n_samples_train])

        # score = model.score(X_test.head(n_samples_test),
        #                     y_test[:n_samples_test])
        y_pred = model.predict(X_test.head(n_samples_test))
        score = hamming_loss(y_test[:n_samples_test], y_pred)
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
        X['subject_id'] = idx

    X.astype({'subject_id': 'int32'}).dtypes
    y = sparse.load_npz(os.path.join(data_dir, 'target.npz')).toarray()

    # Scale data and L to avoid tiny numbers
    # X.iloc[:, :-2] /= np.max(X.iloc[:, :-2])
    L = 1e8 * np.array(L)
    X.iloc[:, :-2] *= 1e12
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


def make_learning_curve_for_all(X, y, models, n_samples_grid):
    # Do learning curve for all models and all datasets
    # returns data frame with names of the models and the hamming score
    # calculated on the predictions of this model
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
                         label=str(cond[1]) + 's: ' + cond[2])
        else:
            ax.plot(df.n_samples_train, df.score_test,
                    label=str(cond[1]) + 's: ' + cond[2])

    for idx, parcel in enumerate(diff_parcels):
        if type(ax) == np.ndarray:
            ax[idx].set(xlabel='n_samples_train', ylabel='score',
                        title='Parcels: ' + str(parcel))
        else:
            ax.set(xlabel='n_samples_train', ylabel='score',
                   title='Parcels: ' + str(parcel))
        plt.legend()
    plt.tight_layout()
    fig_path = os.path.join('figs', file_name + ext)
    plt.savefig(fig_path)
    print(('figure saved in {}').format(fig_path))


if __name__ == "__main__":
    plot_data = False
    calc_scores_for_lc = False
    calc_learning_rate = False
    save_y_pred = False
    score_on_predicted = False
    plot_parcels = False

    username = os.environ.get('USER')
    data_dir = 'data_grad_sample_450_3'
    # data_dir = 'data_grad_CC120008_80_1'

    if "mtelen" in username or 'maja' in username:
        data_dir_base = 'data'
    elif "hjana" in username:
        data_dir_base = "/storage/store/work/hjanati/datasets"
    else:
        pass

    data_dir = os.path.join(data_dir_base, data_dir)
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
    model_lars = linear_model.LassoLars(max_iter=3, normalize=False,
                                        fit_intercept=False)

    lasso_lars = SparseRegressor(L, parcel_indices, model_lars)  # , data_dir)

    model_reweighted = ReweightedLasso(alpha_fraction=.8, max_iter=20,
                                       max_iter_reweighting=10, tol=1e-4)
    lasso_reweighted = SparseRegressor(L, parcel_indices, model_reweighted)

    model_reweighted_not = ReweightedLasso(alpha_fraction=.01, max_iter=20,
                                           max_iter_reweighting=1, tol=1e-4)
    lasso_reweighted_not = SparseRegressor(L, parcel_indices,
                                           model_reweighted_not)

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
        calc_scores_for_model(X, y, model=lc, n_samples=n_samples_grid)

    models = {'K-neighbours(3)': kneighbours,
              'lead correlate': lc,
              'lasso lars': lasso_lars,
              '1 lasso reweighted': lasso_reweighted_not,
              '10 lasso reweighted': lasso_reweighted,
              }

    scores_save_file = os.path.join(data_dir, "scores_all.pkl")
    if calc_learning_rate:
        # make learning curve for selected models
        # models = {'lasso reweighted': lasso_reweighted}

        scores_all = make_learning_curve_for_all(X, y, models, n_samples_grid)
        scores_all.to_pickle(scores_save_file)

        print(scores_all.tail(len(models)))

    models_pred_file = os.path.join(data_dir, "models_pred_all.pkl")
    if save_y_pred:
        # split the data
        model_pred = {}
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42)
        n_samples_train = len(X_train)
        n_samples_test = len(X_test)
        for name, model in models.items():
            print('running predictions on {}'.format(name))
            if name == 'K-neighbours(3)':
                model.fit(X_train.loc[:, X_train.columns != 'subject'].head(
                          n_samples_train), y_train[:n_samples_train])
                y_pred = model.predict(X_test.loc[:,
                                       X_test.columns != 'subject'].head(
                                                            n_samples_test))
            else:
                model.fit(X_train.head(n_samples_train),
                          y_train[:n_samples_train])
                y_pred = model.predict(X_test.head(n_samples_test))
            model_pred[name] = y_pred

        model_pred['y_true'] = y_test[:n_samples_test]
        model_pred['subject'] = X_test.head(
                                    n_samples_test)['subject'].to_numpy()
        with open(models_pred_file, 'wb') as handle:
            pickle.dump(model_pred, handle)
        print('saved the predictions to {}'.format(models_pred_file))

    if score_on_predicted:
        try:
            with open(models_pred_file, 'rb') as handle:
                model_pred = pickle.load(handle)
        except FileNotFoundError:
            print('{} cannot be found; try setting `save_y_predict` to True'
                  .format(models_pred_file))

        models_score = {}
        y_true = model_pred.pop('y_true')
        subjects = model_pred.pop('subject')

        for model_name in model_pred.keys():
            models_score[model_name] = {}

            y_pred = model_pred[model_name]
            models_score[model_name]['jaccard_score'] = jaccard_score(
                y_true, y_pred, average='samples')
            models_score[model_name]['hamming_loss'] = hamming_loss(
                y_true, y_pred)
            models_score[model_name]['emd'] = met.emd_score_subjects(
                subjects, y_true, y_pred, data_dir)

        models_score = pd.DataFrame(models_score)
        print(models_score)
        models_score.to_csv(os.path.join(data_dir, 'score_per_model.csv'))

    plot_predicted_score = True
    if plot_predicted_score:
        test_dataset = 'data_grad_sample_450'
        data_dirs = [data_dir for data_dir in os.listdir(data_dir_base) if
                     data_dir.startswith(test_dataset)]
        data_dirs.sort()
        pad = 5
        import matplotlib.pylab as plt
        n_s = []
        # prepare the axes
        for idx, data_dir in enumerate(data_dirs):
            score_file = os.path.join(data_dir_base, data_dir,
                                      'score_per_model.csv')
            if not os.path.exists(score_file):
                print('You need to calculate score to use {score_file}')
                break

            scores = pd.read_csv(score_file, index_col=0)
            n_sources = int(data_dir.split('_')[-1])
            n_s.append(n_sources)

            if idx == 0:
                score_types = scores.index
                fig, axes = plt.subplots(nrows=len(data_dirs),
                                         ncols=len(score_types),
                                         figsize=(12, 10))
                for ax, score_type in zip(axes[0], score_types):
                    ax.annotate(score_type, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

        for ax, n_source in zip(axes[:,0], n_s):
            txt = 'max ' + str(n_source) + '\n sources'
            ax.annotate(txt, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad,0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center')
            #    score_types
                #for idx_s, score_type in enumerate(score_types):
                #    plt.subplot(len(data_dirs), len(score_types), idx_s+1)
                #    plt.title(score_type)
            #plt.subplot(len(data_dirs), len(score_types),
            #            idx*len(score_types)+1)

        ticks = np.arange(len(scores.columns)) + 1
        for ax in axes[-1,:]:
            ax.set_xticks(ticks)
            ax.set_xticklabels(scores.columns, minor=False, rotation=45,
                               ha='right')

        for axs in axes[:-1,:]:
            [ax.set_xticks(ticks) for ax in axs]
            [ax.set_xticklabels('', minor=False) for ax in axs]


        plt.tight_layout()
        import pdb; pdb.set_trace()

    plot_data = plot_data and visualize_data
    if False and plot_data:
        # plot sources at the activation
        plot_sources_at_activation(X, y, signal_type)

    if plot_data:
        # plot scores
        scores_all = pd.read_pickle(scores_save_file)
        plot_scores(scores_all, file_name='learning_curves', ext='.png')

    if False:  # plot_data: # and False:
        # plot parcels
        display_true_pred_parcels(X, y, data_dir, model=lasso_reweighted,
                                  model_name='lasso lars',
                                  n_samples='all')
    if False:
        display_distances_on_brain(data_dir, subject='sample')

    plot_parcels = False
    if plot_parcels:
        plot_all_parcels(data_dir, 'sample')
