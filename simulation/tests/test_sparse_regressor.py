import pytest

import numpy as np
import numpy.random as random
import pandas as pd
from sklearn import linear_model

from simulation.sparse_regressor import SparseRegressor

SEED = 42


@pytest.fixture
def make_dataset(subj_no=1, samples_per_subj=500, parcels_no=112,
                 sources_no=5775, electrode_no=204):
    # TODO: here each L will be of the same size, change so that sources_no
    # varies between subjects
    random.seed(SEED)
    electrode_names = ['e' + str(i) for i in range(1, electrode_no+1)]
    electrode_names.append('subject')
    electrode_names.append('subject_id')

    L, parcel_indices = [], []

    # X, y, L, parcel_indices, signal_type_data
    for subj in range(subj_no):
        # make random Lead Field
        L.append(random.rand(electrode_no, sources_no))

        # make parcel_indices
        parcel_indices.append(random.randint(1, parcels_no+1, sources_no))

    # make X dataset
    X = np.random.rand(samples_per_subj * subj_no, electrode_no + 2)
    X = pd.DataFrame(data=X, columns=electrode_names)
    X['subject_id'] = np.repeat(range(subj_no), samples_per_subj)
    X['subject'] = [str(int(subj)) for subj in X['subject_id']]

    return X, L, parcel_indices


@pytest.mark.parametrize('solver',
                         ['lasso_lars'])
def test_sparse_regressor(make_dataset, solver):
    if solver == 'lasso_lars':
        model = linear_model.LassoLars(max_iter=3, normalize=False,
                                       fit_intercept=False)
    X, L, parcel_indices = make_dataset
    lasso_lars = SparseRegressor(L, parcel_indices, model)
