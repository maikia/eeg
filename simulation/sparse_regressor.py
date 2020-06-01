import numpy as np
import os
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor
import warnings

from simulation.emd import emd_score


def _get_coef(est):
    if hasattr(est, 'steps'):
        return est.steps[-1][1].coef_
    return est.coef_


class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, lead_field, parcel_indices, model, data_dir, n_jobs=1,
                 weighted_alpha=True):
        self.lead_field = lead_field
        self.parcel_indices = parcel_indices
        self.model = model
        self.n_jobs = n_jobs
        self.data_dir = data_dir
        self.weighted_alpha = weighted_alpha

    def fit(self, X, y):
        return self

    def score(self, X, y):
        # overwites given score with the EMD score (based on the distance)

        y_pred = self.predict(X)
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
        return score

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def residual(self, X, coef_, y):
        """Compute y - X @ coef_."""
        R = y - np.array([x.dot(th) for x, th in zip(X, coef_.T)])
        return R

    def _run_reweighted_model(self, model, L, X, max_iter_reweighting=100):
        # find the best alpha for each case (weighted model)
        # adapted from https://github.com/hichamjanati/mutar
        Xarr = np.array(X)
        tol = 1e-4
        n_tasks = len(X.index.values)
        n_samples, n_features = L.shape
        coef_ = np.zeros((n_tasks, n_features))

        weights = np.ones(coef_.shape)  # np.ones_like(coef_[0, :])
        loss_ = []
        coef_old = coef_.copy()

        alpha_max = abs(L.T.dot(X.T)).max() / len(L)
        alpha = 0.2 * alpha_max
        for i in range(max_iter_reweighting):
            Lw = L[None, :, :]
            weights_ = weights[:, None, :]
            Lw = Lw * weights_

            norms = Lw.std(axis=1)
            Lw = Lw / norms[:, None, :]

            # model.estimator.alpha = alpha
            # model.fit(L, X.T)
            self.model.alpha = alpha
            #import pdb; pdb.set_trace()
            for ii in range(Xarr.shape[0]):
                coef_[ii] = self.model.fit(Lw[ii, :, :], Xarr[ii, :]).coef_

            coef_ = coef_ * weights

            err = abs(coef_ - coef_old).max()
            err /= max(abs(coef_).max(), abs(coef_old).max(), 1.)
            # print(err)
            coef_old = coef_.copy()
            weights = 2 * (abs(coef_) ** 0.5 + 1e-10)

            # import pdb; pdb.set_trace()
            #obj = 0.5 * (utils.residual(X, coef_, y) ** 2).sum() / n_samples
            #obj += (self.alpha[None, :] * abs(coef_) ** 0.5).sum()
            #self.loss_.append(obj)
            if err < tol and i:
                break

        if i == max_iter_reweighting - 1 and i:
            warnings.warn('Reweighted objective did not converge.' +
                          ' You might want to increase ' +
                          'the number of iterations of reweighting.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)
        return coef_.T

    def _run_model(self, model, L, X, fraction_alpha=0.2):
        norms = L.std(axis=0)
        L = L / norms[None, :]

        alpha_max = abs(L.T.dot(X.T)).max() / len(L)
        alpha = fraction_alpha * alpha_max

        model.estimator.alpha = alpha
        model.fit(L, X.T)  # cross validation done here
        est_coefs = np.empty((X.shape[0], L.shape[1]))
        for idx, idx_used in enumerate(X.index.values):
            est_coef = np.abs(_get_coef(model.estimators_[idx]))
            est_coef /= norms
            est_coefs[idx] = est_coef
        return est_coefs.T

    def decision_function(self, X):
        model = MultiOutputRegressor(self.model, n_jobs=self.n_jobs)
        X = X.reset_index(drop=True)

        n_parcels = max(max(s) for s in self.parcel_indices)
        betas = np.empty((len(X), n_parcels))
        for subj_idx in np.unique(X['subject_id']):
            l_used = self.lead_field[subj_idx]

            X_used = X[X['subject_id'] == subj_idx]
            X_used = X_used.iloc[:, :-2]

            if self.weighted_alpha:
                est_coef = self._run_reweighted_model(model, l_used, X_used)
            else:
                est_coef = self._run_model(model, l_used, X_used)

            beta = pd.DataFrame(
                       np.abs(est_coef)
                   ).groupby(
                   self.parcel_indices[subj_idx]).max().transpose()
            betas[X['subject_id'] == subj_idx] = np.array(beta)
        return betas

