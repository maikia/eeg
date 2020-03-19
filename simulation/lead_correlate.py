import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class LeadCorrelate(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, lead_field, parcel_indices_leadfield):
        self.L = lead_field
        self.parcel_indices = parcel_indices_leadfield

    def fit(self, X, y):
        """ dummy
        """
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        # X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')
        y = np.zeros(X.shape[0])
        for idx in range(0, len(X)):
            x = X.iloc[[idx]]
            y_pred = pd.DataFrame(self.L.T @ x.T).groupby(self.parcel_indices).max().idxmax().values[0]

            y[idx] = y_pred

        return y

    def score(self):
        # TODO:
        if y_pred == y_true:
            # predicted correctly
            score += 1
        y_true = np.where(y_train[idx])[0][0] + 1
        final_score = score/(idx+1)
        return final_score
