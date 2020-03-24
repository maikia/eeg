import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_selection import f_regression
from sklearn.utils.validation import check_is_fitted
# from sklearn.utils.validation import check_X_y, check_array
# from sklearn.utils.multiclass import unique_labels
# from sklearn.metrics import euclidean_distances


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
        self.threshold_ = df_mask_flat.min()

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
        """
        n_samples, _ = X.shape
        y_pred = np.zeros((n_samples, self.n_sources_), dtype=int)

        check_is_fitted(self, 'n_sources_')

        corr = self.decision_function(X)
        corr = np.array(corr)

        # take values higher than threshold
        corr_poss = corr >= self.threshold_  # boolen
        more, less, good = 0, 0, 0
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

        return self.froc_score(y, self.decision_function(X))

    def decision_function(self, X):
        """ Computes the correlation of the data with the lead field
        Args:
            X: data
        Returns:
            decision: correlation of the signal from each parcel with the given
            data for each sample, dtype = DataFrame
        """
        n_samples, _ = X.shape
        L = self.lead_field
        L = L / linalg.norm(L, axis=0)  # normalize leadfield column wise
        parcel_indices = self.parcel_indices_leadfield

        for idx in range(n_samples):
            x = X.iloc[idx]
            x = x / linalg.norm(x)  # normalize x to take correlations

            corr = pd.DataFrame(np.abs(L.T.dot(x))).groupby(
                         parcel_indices).max().transpose()
            if idx:
                correlation = correlation.append(corr)
            else:
                correlation = corr

        correlation.index = range(n_samples)

        # we don't use the 0 index
        # TODO: remove passing L.idx = 0 all together
        if 0 in correlation:
            correlation = correlation.drop(columns = 0)
        return correlation

    def froc_score(self, y_true, y_score):
        """compute Free response receiver operating characteristic curve (FROC)
        Note: this implementation is restricted to the binary classification task.
        Parameters
        ----------
        y_true : array, shape = [n_samples x n_classes]
                 true binary labels
        y_score : array, shape = [n_samples x n_classes]
                 target scores: probability estimates of the positive class,
                 confidence values
        Returns
        -------
        ts : array, shape = [>2]
            total sensitivity: true positive normalized by sum of all true
            positives
        tfp : array, shape = [>2]
            total false positive: False positive rate divided by length of
            y_true
        thresholds : array, shape = [>2]
            Thresholds on y_score used to compute ts and tfp.
            *Note*: Since the thresholds are sorted from low to high values,
            they are reversed upon returning them to ensure they
            correspond to both fpr and tpr, which are sorted in reversed order
            during their calculation.

        References
        ----------
        http://www.devchakraborty.com/Receiver%20operating%20characteristic.pdf
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3679336/pdf/nihms458993.pdf
        """

        n_samples = len(y_true)
        classes = np.unique(y_true)

        n_pos = float(np.sum(y_true == classes[1]))  # nb of true positive
        n_neg = float(np.sum(y_true == classes[0]))  # nb of true negative

        y_true = np.ravel(y_true)
        y_score = np.ravel(y_score)


        # FROC only for binary classification
        if classes.shape[0] != 2:
            raise ValueError("FROC is defined for binary classification only")

        thresholds = np.unique(y_score)
        neg_value, pos_value = classes[0], classes[1]

        # total sensitivity: true positive normalized by sum of all true
        # positives
        ts = np.zeros(thresholds.size, dtype=np.float)
        # total false positive: False positive rate divided by length of y_true
        tfp = np.zeros(thresholds.size, dtype=np.float)

        current_pos_count = current_neg_count = sum_pos = sum_neg = idx = 0

        signal = np.c_[y_score, y_true]
        sorted_signal = signal[signal[:, 0].argsort(), :][::-1]
        last_score = sorted_signal[0][0]
        for score, value in sorted_signal:
            t = value
            t_est = sorted_signal[:, 0] >= score

            # false positives for this score (threshold)
            unique, counts = np.unique(sorted_signal[:, 1] - t_est, return_counts=True)
            try:
                fps = counts[np.where(unique == -1)][0]
            except IndexError:
                fps = 0
            # true positives for this score (threshold)
            unique, counts = np.unique(sorted_signal[:, 1] + t_est, return_counts=True)
            try:
                tps = counts[np.where(unique == 2)][0]
            except IndexError:
                tps = 0

            ts[idx] = tps
            tfp[idx] = fps

            idx += 1

        tfp = tfp / n_samples
        ts = ts / n_pos

        threshs = thresholds[::-1]

        # TODO: remove:
        plt.figure()
        plt.plot(tfp, ts, 'ro')
        plt.xlabel('total false positives', fontsize=12)
        plt.ylabel('total sensitivity', fontsize=12)
        thresh = threshs.round(5).astype(str)[::400]
        for fp, ts, t in zip(tfp[::400], ts[::400], thresh):
            plt.text(fp, ts-0.025, t, rotation=45)
        plt.title('FROC, max parcels: ' + str(self.n_sources_))
        plt.show()

        return ts, tfp, thresholds[::-1]



    def plotFROC(self):
        """Plots the FROC curve (Free response receiver operating
           characteristic curve)
        """
        # ts, tfp, 
        threshs = thresholds[::-1]
        plt.figure()
        plt.plot(tfp, ts, 'ro')
        plt.xlabel('total false positives', fontsize=12)
        plt.ylabel('total sensitivity', fontsize=12)
        thresh = threshs(5).astype(str)[::100]
        for fp, ts, t in zip(tfp, ts, thresh):
            plt.text(fp, ts-0.025, t, rotation=45)
        plt.title('FROC, max parcels: ' + str(self.n_sources_))
