import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model

from sklearn.utils.validation import check_is_fitted

import simulation.metrics as met


class LeadCorrelate(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, lead_field, parcel_indices_leadfield):
        self.lead_field = lead_field
        self.parcel_indices_leadfield = parcel_indices_leadfield

    def fit(self, X, y):
        """
        """
        df = self.decision_function(X)
        df = np.array(df)
        assert df.shape == y.shape
        df_mask_flat = df.ravel()[y.ravel() == 1]

        # threshold is selected at 2% of all true
        sorted_mask = np.sort(df_mask_flat)
        thresh_idx = int(len(df_mask_flat)*(0.55))
        self.threshold_ = sorted_mask[thresh_idx]

        # check what is max possible number of parcels
        true_per_row = np.sum(y, axis=1).astype(int)
        self.max_active_sources_ = np.unique(true_per_row)[-1]
        self.n_sources_ = y.shape[1]

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.

            TODO: predict(x): return (lasso.fit(L, x).coef_ != 0).astype(int)
        """
        n_samples, _ = X.shape
        y_pred = np.zeros((n_samples, self.n_sources_), dtype=int)

        check_is_fitted(self, 'n_sources_')

        corr = self.decision_function(X)
        corr = np.array(corr)

        # take values higher than threshold
        corr_poss = corr >= self.threshold_  # boolen
        # leave only max_active_sources_ parcels

        for idx in range(0, len(corr_poss)):
            # check if more than 0 and less than max_active_sources_
            if sum(corr_poss[idx, :]) > self.max_active_sources_:
                # take only self.max_active_sources_ highest possible corr
                corr[idx, np.logical_not(corr_poss[idx, :])] = 0
                y_pred[idx, np.argsort(
                       corr[idx, :])[-self.max_active_sources_:]] = 1
            elif sum(corr_poss[idx, :]) < 1:
                # take a single highest possible corr
                max_corr_idx = np.argsort(corr[idx, :])[-1]
                y_pred[idx, max_corr_idx] = 1
            else:
                # leave corr as selected by corr_poss
                y_pred[idx, corr_poss[idx, :]] = 1
        return y_pred

    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            average number of errors per sample (the more the worse)
        """
        return met.afroc_score(y, self.decision_function(X))

    def decision_function(self, X):
        """ Computes the correlation of the data with the lead field
        Args:
            X: data
        Returns:
            decision: correlation of the signal from each parcel with the given
            data for each sample, dtype = DataFrame
        """
        # ?? do the L has to be normalized for Lasso?
        # L = L / linalg.norm(L, axis=0)  # normalize leadfield column wise

        n_samples, _ = X.shape
        L = self.lead_field
        parcel_indices = self.parcel_indices_leadfield

        clf = linear_model.LassoLarsCV()
        model = MultiOutputRegressor(clf, n_jobs=-1)
        model.fit(L, X.T)

        n_est = len(model.estimators_)
        betas = np.empty([n_est, len(np.unique(parcel_indices))])
        for idx in range(n_est):
            est_coef = model.estimators_[idx].coef_
            beta = pd.DataFrame(
                np.abs(est_coef)).groupby(parcel_indices).max().transpose()
            betas[idx, :] = beta

        return betas
