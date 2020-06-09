import pytest

import numpy.random as random
from sklearn import linear_model

from simulation.sparse_regressor import SparseRegressor

SEED = 42


@pytest.fixture
def make_dataset(subj_no=1, samples_per_subj=100, parcels_no=112,
                 sources_no=5775):
    # TODO: here each L will be of the same size, change so that sources_no
    # varies between subjects
    random.seed(SEED)
    L, parcel_indices = [], []
    for subj in range(subj_no):
        # make random Lead Field
        L.append(random.rand(samples_per_subj, sources_no))

        # make parcel_indices
        parcel_indices.append(random.randint(1, parcels_no+1, sources_no))

    return L, parcel_indices


@pytest.mark.parametrize('solver',
                         ['lasso_lars'])
def test_sparse_regressor(make_dataset, solver):
    if solver == 'lasso_lars':
        model = linear_model.LassoLars(max_iter=3, normalize=False,
                                       fit_intercept=False)
    L, parcel_indices = make_dataset
    lasso_lars = SparseRegressor(L, parcel_indices, model)
