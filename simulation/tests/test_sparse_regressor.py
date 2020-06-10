import pytest

import mne
from mne.datasets import sample

import numpy as np
import numpy.random as random
import os
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import hamming_loss, jaccard_score

from simulation.sparse_regressor import SparseRegressor, ReweightedLasso

SEED = 42


def make_dataset_from_sample():
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    evoked = mne.read_evokeds(raw_fname, condition='Left Auditory',
                              baseline=(None, 0))

    fwd_fname = os.path.join(data_path, 'MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    picks_meg = mne.pick_types(fwd['info'], meg=True, eeg=False, exclude=[])
    lead_field = fwd['sol']['data']
    lead_field = lead_field[picks_meg, :]

    stc = mne.read_source_estimate(fname_stc)

    parcel_indices_lh = np.zeros(len(fwd['src'][0]['inuse']), dtype=int)
    parcel_indices_rh = np.zeros(len(fwd['src'][1]['inuse']), dtype=int)
    for label_name, label_idx in parcel_vertices.items():
        label_id = int(label_name[:-3])
        if '-lh' in label_name:
            parcel_indices_lh[label_idx] = label_id
        else:
            parcel_indices_rh[label_idx] = label_id
    import pdb; pdb.set_trace()
    # return X, y, lead_field, parcel_indices
    return lead_field

def test_sample():
    make_dataset_from_sample()


def make_dataset(n_subjects=1, n_samples_per_subj=2, n_parcels=10,
                 n_sources=300, n_sensors=100, max_true_sources=1):
    rng = np.random.RandomState(SEED)

    electrode_names = [f'e{i}' for i in range(1, n_sensors + 1)]

    L, parcel_indices = [], []
    X = []
    y = []

    for k in range(n_subjects):
        # make random Lead Field
        Lk = rng.randn(n_sensors, n_sources)
        L.append(Lk)

        parcel_indices_k = (np.arange(n_sources) % n_parcels) + 1
        parcel_indices.append(parcel_indices_k)

        mask = np.zeros((n_samples_per_subj, n_sources), dtype=bool)
        yk = np.zeros((n_samples_per_subj, n_parcels), dtype=int)
        for i in range(n_samples_per_subj):
            mask[i, rng.randint(0, n_sources, size=max_true_sources)] = True
            yk[i, np.unique(parcel_indices_k[mask[i]]) - 1] = 1

        beta = rng.randn(n_samples_per_subj, n_sources)
        beta[~mask] = 0

        Xk = (Lk @ beta.T).T

        Xk = pd.DataFrame(Xk, columns=electrode_names)
        Xk['subject_id'] = k
        X.append(Xk)
        y.append(yk)

    X = pd.concat(X, axis=0)
    X['subject'] = X['subject_id'].apply(str)
    y = np.concatenate(y, axis=0)

    return X, y, L, parcel_indices


lasso = linear_model.LassoLars(max_iter=2, normalize=False,
                               fit_intercept=False, alpha=0.01)
rwl1 = ReweightedLasso(alpha_fraction=.01, max_iter=20,
                       max_iter_reweighting=1, tol=1e-4)
rwl10 = ReweightedLasso(alpha_fraction=.01, max_iter=20,
                        max_iter_reweighting=10, tol=1e-4)


@pytest.mark.parametrize('model, hl_max',
                         [(lasso, 0.02),
                          (rwl1, 0),
                          (rwl10, 0.002)
                          ])
def test_sparse_regressor(model, hl_max):
    n_subjects = 1
    n_samples_per_subj = 100
    n_parcels = 10
    n_sources = 500
    n_sensors = 100
    max_true_sources = 2
    X, y, L, parcel_indices = make_dataset(
        n_subjects, n_samples_per_subj, n_parcels, n_sources,
        n_sensors, max_true_sources
    )

    # assert that all the dimensions correspond
    assert X.shape == (n_samples_per_subj * n_subjects, n_sensors + 2)
    assert X['subject_id'].unique() == np.arange(n_subjects)
    assert X.shape[0] == y.shape[0]

    assert len(L) == n_subjects == len(parcel_indices)
    assert L[0].shape == (n_sensors, n_sources)
    assert y.shape[1] == n_parcels
    assert np.mean(y.sum(1) == max_true_sources) > 0.9

    sparse_regressor = SparseRegressor(
        L, parcel_indices, model
    )

    y_pred = sparse_regressor.predict(X)
    hl = hamming_loss(y_pred, y)
    assert hl <= hl_max
