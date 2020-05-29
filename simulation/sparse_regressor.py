import numpy as np
import os
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor

from simulation.emd import emd_score


def _get_coef(est):
    if hasattr(est, 'steps'):
        return est.steps[-1][1].coef_
    return est.coef_


class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, lead_field, parcel_indices, model, data_dir, n_jobs=1):
        self.lead_field = lead_field
        self.parcel_indices = parcel_indices
        self.model = model
        self.n_jobs = n_jobs
        self.data_dir = data_dir

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

    def _weighted_model():
        pass

    def decision_function(self, X):
        model = MultiOutputRegressor(self.model, n_jobs=self.n_jobs)
        X = X.reset_index(drop=True)

        betas = np.empty((len(X), 0)).tolist()
        for subj_idx in np.unique(X['subject_id']):
            l_used = self.lead_field[subj_idx]

            X_used = X[X['subject_id'] == subj_idx]
            X_used = X_used.iloc[:, :-2]

            # adapted from https://github.com/hichamjanati/mutar
            X_used = np.array(X_used)

            ##### test only ####
            l_used = np.array([[[3, 1], [2, 0], [1, 0]],\
                     [[0, 2], [-1, 3], [1, -2]]], dtype=float)
            coef = np.array([[1., 1.], [0., -1]])
            # X_used = np.array([x.dot(c) for x, c in zip(l_used, coef.T)])
            # X_used += 0.1
            X_used = np.array([[[0, 0], [1, 1], [1, 0]],[[1, 0], [0, 0], [1, 1]]])
            self.alpha = np.asarray([0.1, 0.2])

            # X_used = X_used[np.newaxis, :]
            max_iter_reweighting = 100
            n_subjects = len(l_used)
            n_samples, n_features = l_used[0].shape

            self.coef_ = np.zeros((n_features, n_subjects))
            weights = np.ones_like(self.coef_)
            coef_old = self.coef_.copy()
            self.loss_ = []
            self.tol=1e-4
            # norms = l_used.std(axis=0)
            # l_used = l_used / norms[None, :]
            for i in range(max_iter_reweighting):
                lw = l_used * weights.T[:, None, :]
                theta = np.zeros((n_features, n_subjects))

                # solver lasso
                for k in range(n_subjects):
                    alpha_lasso = self.alpha.copy()
                    alpha_lasso = np.asarray(alpha_lasso).reshape(n_subjects)
                    #from sklearn.linear_model import Lasso
                    # lasso = Lasso(alpha=alpha_lasso[k], tol=1e-4, max_iter=2000,
                    #               fit_intercept=False, positive=False)
                    model.estimator.alpha = alpha_lasso[k]
                    model.fit(lw[k], X_used[k])
                    # model.estimator.alpha = alpha_lasso

                    theta[:, k] = np.abs(_get_coef(model.estimators_[k]))
                coef_ = theta * weights
                err = abs(coef_ - coef_old).max()
                err /= max(abs(coef_).max(), abs(coef_old).max(), 1.)
                coef_old = coef_.copy()
                weights = 2 * (abs(coef_) ** 0.5 + 1e-10)
                # import pdb; pdb.set_trace()
                # obj = 0.5 * (self.residual(lw, coef_, X_used) ** 2).sum() / n_samples
                # obj += (self.alpha[None, :] * abs(coef_) ** 0.5).sum()
                # self.loss_.append(obj)

                if err < self.tol and i:
                    break

            if i == max_iter_reweighting - 1 and i:
                warnings.warn('Reweighted objective did not converge.' +
                              ' You might want to increase ' +
                          'the number of iterations of reweighting.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)
            import pdb; pdb.set_trace()

            '''
            norms = l_used.std(axis=0)
            l_used = l_used / norms[None, :]

            alpha_max = abs(l_used.T.dot(X_used.T)).max() / len(l_used)
            alpha = 0.2 * alpha_max
            model.estimator.alpha = alpha
            model.fit(l_used, X_used.T)

            for idx, idx_used in enumerate(X_used.index.values):
                est_coef = np.abs(_get_coef(model.estimators_[idx]))
                est_coef /= norms
                beta = pd.DataFrame(
                        np.abs(est_coef)
                        ).groupby(
                        self.parcel_indices[subj_idx]).max().transpose()
                betas[idx_used] = np.array(beta).ravel()
            '''
        betas = np.array(betas)
        return betas
