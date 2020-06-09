import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model

from sklearn.metrics import hamming_loss
# from simulation.emd import emd_score


def solver_lasso(Xw, y, alpha, max_iter):
    model = linear_model.LassoLars(max_iter=max_iter, normalize=False,
                                   fit_intercept=False, alpha=alpha)
    return model.fit(Xw, y).coef_.copy()



class ReweightedLasso(BaseEstimator, RegressorMixin):
    """ Reweighted Lasso estimator with L1 regularizer.

    The optimization objective for Reweighted Lasso is::
        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_0.5

    Where::
        ||W||_0.5 = sum_i sum_j sqrt|w_ij|

    Parameters
    ----------
    alpha : (float or array-like), shape (n_tasks)
        Optional, default ones(n_tasks)
        Constant that multiplies the L0.5 term. Defaults to 1.0
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Parameter vector (W in the cost function formula).
    """
    def __init__(self, alpha_fraction=.01, max_iter=2000,
                 max_iter_reweighting=100, tol=1e-4):
        self.alpha_fraction = alpha_fraction
        self.max_iter = max_iter
        self.max_iter_reweighting = max_iter_reweighting
        self.tol = tol

    def fit(self, X, y):
        n_tasks = len(X)
        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        weights = np.ones_like(self.coef_)
        coef_old = self.coef_.copy()

        self.loss_ = []

        alpha_max = abs(X.T.dot(y)).max() / len(X)
        alpha = self.alpha_fraction * alpha_max

        for i in range(self.max_iter_reweighting):
            Xw = X * weights
            coef_ = solver_lasso(Xw, y, alpha, self.max_iter)
            coef_ = coef_ * weights
            err = abs(coef_ - coef_old).max()
            err /= max(abs(coef_).max(), abs(coef_old).max(), 1.)
            coef_old = coef_.copy()
            weights = 2 * (abs(coef_) ** 0.5 + 1e-10)
            obj = 0.5 * ((X @ coef_ - y) ** 2).sum() / n_samples
            obj += (alpha * abs(coef_) ** 0.5).sum()
            self.loss_.append(obj)
            if err < self.tol and i:
                break

        if i == self.max_iter_reweighting - 1 and i:
            warnings.warn('Reweighted objective did not converge.' +
                          ' You might want to increase ' +
                          'the number of iterations of reweighting.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)
        self.coef_ = coef_

    def predict(self, X):
        return np.dot(X, self.coef_)


def _get_coef(est):
    if hasattr(est, 'steps'):
        return est.steps[-1][1].coef_
    return est.coef_


class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, lead_field, parcel_indices, model, n_jobs=1):
        self.lead_field = lead_field
        self.parcel_indices = parcel_indices
        self.model = model
        self.n_jobs = n_jobs
        # self.data_dir = data_dir # this is required only if EMD score would
        # be used

    def fit(self, X, y):
        return self

    def score(self, X, y):
        # overwites given score with the EMD score (based on the distance)

        y_pred = self.predict(X)

        score = hamming_loss(y, y_pred)
        '''
        subjects = np.unique(X['subject'])
        scores = np.empty(len(subjects))
        X_used = X.reset_index(drop=True)
        for idx, subject in enumerate(subjects):
            subj_idx = X_used[X_used['subject'] == subject].index
            y_subj = y[subj_idx, :]
            y_pred_subj = y_pred[subj_idx, :]
            labels_x = np.load(os.path.join(self.data_dir,
                                            subject + '_labels.npz'),
                               allow_pickle=True)['arr_0']

            score = emd_score(y_subj, y_pred_subj, labels_x)
            scores[idx] = score * (len(y_subj) / len(y))  # normalize

        score = np.sum(scores)
        '''
        return score

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def _run_model(self, model, L, X, fraction_alpha=0.2):
        norms = np.linalg.norm(L, axis=0)
        L = L / norms[None, :]

        est_coefs = np.empty((X.shape[0], L.shape[1]))
        for idx, idx_used in enumerate(X.index.values):
            x = X.iloc[idx].values
            model.fit(L, x)
            est_coef = np.abs(_get_coef(model))
            est_coef /= norms
            est_coefs[idx] = est_coef

        return est_coefs.T

    def decision_function(self, X):
        X = X.reset_index(drop=True)

        n_parcels = max(max(s) for s in self.parcel_indices)
        betas = np.empty((len(X), n_parcels))
        for subj_idx in np.unique(X['subject_id']):
            l_used = self.lead_field[subj_idx]

            X_used = X[X['subject_id'] == subj_idx]
            X_used = X_used.iloc[:, :-2]

            est_coef = self._run_model(self.model, l_used, X_used)

            beta = pd.DataFrame(
                       np.abs(est_coef)
                   ).groupby(
                   self.parcel_indices[subj_idx]).max().transpose()
            betas[X['subject_id'] == subj_idx] = np.array(beta)
        return betas
