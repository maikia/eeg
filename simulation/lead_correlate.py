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
        # calculate the likelihood to have a number of parcels
        true_per_row = np.sum(y, 1).astype(int)
        self.n_sources = np.unique(true_per_row)
        self.all_sources = np.size(y, 1)
        #occurences = np.bincount(true_per_row)[1:]
        #self.prob_of_occurence = occurences / sum(occurences)

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

        y_pred = np.zeros([X.shape[0], self.all_sources])
        for idx in range(0, len(X)):
            x = X.iloc[[idx]]
            likelihood = pd.DataFrame(self.L.T @ x.T).groupby(
                                            self.parcel_indices).max()
            # consider only n max correlated sources where n is highest number
            # of sources calculated in fit()
            if self.n_sources[-1] == 1:
                y = likelihood.idxmax().values[0]
            else:
                diffs = np.diff(likelihood.nlargest(3, [idx]).values[:,0])
                try:
                    # setting arbitrary threshold
                    cut_at = np.where(diffs < -0.00025)[0][0]
                    y = np.array(likelihood.nlargest(cut_at + 1, [idx]).index)
                except:
                    # take max possible parcels
                    y = np.array(likelihood.nlargest(self.n_sources[-1],
                                                     [idx]).index)

            y_pred[idx, y-1] = 1
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
        y_pred = self.predict(X)
        errors = np.abs(y_pred-y)
        score = np.sum(errors)/len(y)

        return score
