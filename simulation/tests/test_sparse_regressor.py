import pytest

import numpy as np
import numpy.random as random
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import cross_validate, train_test_split

from simulation.sparse_regressor import SparseRegressor

SEED = 42


@pytest.fixture
def make_dataset(n_subjects=1, n_samples_per_subj=2, n_parcels=10,
                 n_sources=300, n_sensors=100, max_true_sources=1):
    # TODO: here each L will be of the same size, change so that sources_no
    # varies between subjects
    rng = np.random.RandomState(SEED)

    electrode_names = [f'e{i}' for i in range(1, n_sensors + 1)]

    L, parcel_indices = [], []
    X = []
    y = []

    # X, y, L, parcel_indices, signal_type_data
    for k in range(n_subjects):
        # make random Lead Field
        Lk = rng.rand(n_sensors, n_sources)
        L.append(Lk)

        # make parcel_indices
        parcel_indices_k = rng.randint(1, n_parcels+1, n_sources)
        parcel_indices.append(parcel_indices_k)

        mask = np.zeros((n_samples_per_subj, n_sources), dtype=bool)
        yk = np.zeros((n_samples_per_subj, n_parcels), dtype=int)
        for i in range(n_samples_per_subj):
            mask[i, rng.randint(0, n_sources, size=2)] = True
            yk[i, np.unique(parcel_indices_k[mask[i]]) - 1] = 1

        beta = rng.randn(n_samples_per_subj, n_sources)
        beta[mask] = 0

        Xk = (Lk @ beta.T).T

        Xk = pd.DataFrame(Xk, columns=electrode_names)
        Xk['subject_id'] = k
        X.append(Xk)
        y.append(yk)

    X = pd.concat(X, axis=0)
    X['subject'] = X['subject_id'].apply(str)
    y = np.concatenate(y, axis=0)

    return X, y, L, parcel_indices


@pytest.mark.parametrize('solver',
                         ['lasso_lars'])
def test_sparse_regressor(make_dataset, solver):
    if solver == 'lasso_lars':
        model = linear_model.LassoLars(max_iter=3, normalize=False,
                                       fit_intercept=False)
    X, y, L, parcel_indices = make_dataset

    # assert that all the dimensions correspond
    assert X[X['subject_id'] == 0].shape[0]  # n_samples_per_subj
    assert X.shape[0] == y.shape[0]
    # n_subjects
    assert len(L) == len(X['subject_id'].unique()) == len(parcel_indices)
    assert L[0].shape[0] == len(X.columns) - 2  # n_sensors
    assert L[0].shape[1] == len(parcel_indices[0])  # n_sources
    assert y.shape[1] == len(np.unique(parcel_indices))  # n_parcels

    sparse_regressor = SparseRegressor(L, parcel_indices, model)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    sparse_regressor.fit(X_train, y_train)
    score = sparse_regressor.score(X_test, y_test)
