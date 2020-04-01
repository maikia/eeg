import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor


class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, lead_field, parcel_indices, model, n_jobs=1):
        self.lead_field = lead_field
        self.parcel_indices = parcel_indices
        self.model = model
        self.n_jobs = n_jobs

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def decision_function(self, X):
        model = MultiOutputRegressor(self.model, n_jobs=self.n_jobs)
        model.fit(self.lead_field, X.T)
        n_est = len(model.estimators_)
        betas = np.empty([n_est, len(np.unique(self.parcel_indices))])
        for idx in range(n_est):
            est_coef = np.abs(model.estimators_[idx].coef_)
            beta = pd.DataFrame(
                np.abs(est_coef)
            ).groupby(self.parcel_indices).max().transpose()
            betas[idx, :] = beta
        return betas
